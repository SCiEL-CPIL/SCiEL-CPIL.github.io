import time
import numpy as np
import random
from itertools import product

# EVOLVE-BLOCK-START

def _int_to_digits(x, q, n, powers):
    """Convert integer index to list of digits."""
    digits = []
    for i in range(n):
        d = (x // powers[i]) % q
        digits.append(d)
    return digits

def _digits_to_int(digits, q, n, powers):
    """Convert list of digits to integer index."""
    x = 0
    for i, d in enumerate(digits):
        x += d * powers[i]
    return x

def _precompute_balls_bitsets(q, n, radius, deadline):
    """
    Precompute for each word x the bitset of words within Hamming distance <= radius.
    Returns a list of numpy arrays (int64 bitsets).
    """
    total = q ** n
    num_blocks = (total + 63) // 64
    
    balls = [None] * total
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # For radius 1, ball size is 1 + n*(q-1)
    # For q=3, n=8, size is 1 + 8*2 = 17.
    
    for x in range(total):
        bits = np.zeros(num_blocks, dtype=np.int64)
        
        # Add x itself
        bits[x // 64] |= (1 << (x % 64))
        
        # Add neighbors
        if radius >= 1:
            for i in range(n):
                old_digit = (x // powers[i]) % q
                for new_digit in range(q):
                    if new_digit != old_digit:
                        y = x + (new_digit - old_digit) * powers[i]
                        bits[y // 64] |= (1 << (y % 64))
                        
        balls[x] = bits
        
        if time.time() > deadline - 20:
            break
            
    # Fill rest if timeout (should not happen for n=8)
    for x in range(total):
        if balls[x] is None:
             # Fallback simple fill
             bits = np.zeros(num_blocks, dtype=np.int64)
             bits[x // 64] |= (1 << (x % 64))
             for i in range(n):
                 old_digit = (x // powers[i]) % q
                 for new_digit in range(q):
                     if new_digit != old_digit:
                         y = x + (new_digit - old_digit) * powers[i]
                         bits[y // 64] |= (1 << (y % 64))
             balls[x] = bits
             
    return balls

def _solve_covering_code_bnb(q, n, radius, deadline):
    """
    Branch and Bound exact solver for covering code.
    Returns list of codeword indices.
    """
    total = q ** n
    balls = _precompute_balls_bitsets(q, n, radius, deadline)
    
    # Upper bound for pruning:
    # Greedy heuristic to get a quick upper bound
    covered = np.zeros(total, dtype=bool)
    greedy_set = []
    # Precompute ball lists for greedy (slower but needed for logic)
    # Actually, we can use the bitsets for greedy too, but iterating bits is annoying.
    # Let's just use a simple greedy for the bound.
    # Since n=8, total=6561, ball size=17.
    # Greedy: pick word covering most uncovered.
    
    # Optimization: Precompute ball lists for greedy
    ball_lists = []
    for i in range(total):
        lst = []
        b = balls[i]
        for block_idx in range(len(b)):
            word_block = b[block_idx]
            while word_block:
                # Extract lowest set bit
                lsb = word_block & -word_block
                pos = (lsb.bit_length() - 1)
                idx = block_idx * 64 + pos
                if idx < total:
                    lst.append(idx)
                word_block ^= lsb
        ball_lists.append(lst)
        
    # Greedy
    uncovered_count = np.ones(total, dtype=np.int32) # Initial uncovered is all 1
    # Wait, uncovered is boolean.
    uncovered = np.ones(total, dtype=bool)
    
    # To speed up greedy, we can just pick random or first uncovered?
    # Standard greedy: pick c maximizing sum(uncovered[y] for y in ball[c])
    # With 6561 words and 17 neighbors, this is 6561*17 ~ 100k ops per step.
    # 500 steps -> 50M ops. Might be slow in Python.
    # Let's use a simpler "random greedy" or "first uncovered" for the initial bound.
    # Actually, a decent initial bound is important for BnB.
    # Let's use a fast greedy:
    # 1. Find an uncovered point u.
    # 2. Pick a random c from ball[u] (or c=u).
    # 3. Add c.
    # This is O(N * ball_size) total if we just add.
    # Better:
    while np.any(uncovered):
        # Find first uncovered
        u = np.where(uncovered)[0][0]
        # Pick a candidate. Let's pick the one that covers most uncovered in its ball.
        best_c = u
        best_count = 0
        # Candidates are in ball[u]
        for c in ball_lists[u]:
            count = np.sum(uncovered[ball_lists[c]])
            if count > best_count:
                best_count = count
                best_c = c
        # Add best_c
        greedy_set.append(best_c)
        for y in ball_lists[best_c]:
            uncovered[y] = False
            
    ub = len(greedy_set)
    
    # If greedy is already very good (e.g. < 490), we might just return it if BnB is too slow?
    # But we want to try for optimal (486).
    
    # BnB State
    # covered_bits: list of bitsets? No, we need to know which points are covered.
    # For BnB, we maintain `uncovered` as a boolean array or bitset.
    # Using boolean array is easier for logic, bitset for speed.
    # Let's use boolean array for simplicity, but it's slow.
    # Let's use bitsets for uncovered too.
    
    uncovered_bits = [0] * (total // 64 + 1)
    for i in range(total):
        uncovered_bits[i // 64] |= (1 << (i % 64))
        
    # We need to efficiently compute "uncovered & ball[c]".
    # ball[c] is a bitset.
    # intersection = uncovered_bits AND ball[c]
    # If intersection is non-zero, c covers some new points.
    # If intersection is zero, c is useless.
    
    # We also need to update uncovered_bits when adding c.
    # uncovered_bits = uncovered_bits AND NOT ball[c]
    
    # Heuristic for branching:
    # 1. Find an uncovered point u.
    # 2. Branch on candidates c in ball[u].
    #    Order candidates by how many new points they cover (greedy heuristic).
    
    # Global best
    best_solution = greedy_set[:]
    best_size = ub
    
    # Time management
    start_time = time.time()
    deadline_bnb = deadline - 5 # Leave 5s for cleanup
    
    # To speed up finding an uncovered point from bitset:
    def get_first_uncovered(bits):
        for b_idx in range(len(bits)):
            if bits[b_idx]:
                lsb = bits[b_idx] & -bits[b_idx]
                pos = lsb.bit_length() - 1
                return b_idx * 64 + pos
        return -1
    
    # Recursion stack
    # To avoid deep recursion overhead, we can use an explicit stack or just recursion with limit.
    # Depth is ~486. Python recursion limit is usually 1000. We should be fine.
    # But we need to pass state. Copying bitsets is expensive (6561/64 ~ 103 int64s).
    # Copying 103 ints is fast.
    
    # Let's use a class or closure for state.
    
    # Optimization: Precompute ball bitsets as lists of ints?
    # balls is already list of np arrays.
    
    # We will use a recursive function.
    # To prevent stack overflow or too much overhead, we might need to be careful.
    
    # Let's try a different approach:
    # Since the space is small (6561), and we have a good upper bound (greedy),
    # we can use a "Best-First" or "Depth-First" with pruning.
    
    # Let's stick to DFS with strong pruning.
    
    # Precompute candidate lists for each u?
    # For a given u, candidates are ball[u].
    # We need to sort them by "gain".
    # Gain depends on current uncovered set.
    
    # To make it faster, we can use a heuristic:
    # If we get stuck, we can use the greedy result.
    
    # Let's implement the DFS.
    
    # To speed up, we can use numpy for bit operations?
    # np.bitwise_and, np.bitwise_not.
    # But scalar operations in Python loop over blocks might be slower than numpy vectorized.
    # Let's use numpy arrays for uncovered_bits.
    
    uncovered_np = np.zeros(total // 64 + 1, dtype=np.int64)
    for i in range(total):
        uncovered_np[i // 64] |= (1 << (i % 64))
        
    # Helper to check if empty
    def is_empty(bits):
        return not np.any(bits)
        
    # Helper to get first set bit index
    def first_set_bit(bits):
        # Find first non-zero block
        non_zero = np.nonzero(bits)[0]
        if len(non_zero) == 0:
            return -1
        block_idx = non_zero[0]
        word = bits[block_idx]
        lsb = word & -word
        pos = (lsb.bit_length() - 1)
        return block_idx * 64 + pos
        
    # Helper to compute gain
    def compute_gain(c, unc):
        # Return number of bits in unc & balls[c]
        inter = np.bitwise_and(unc, balls[c])
        return np.bit_count(inter).sum() if hasattr(np, 'bit_count') else sum(bin(x).count('1') for x in inter)
        
    # np.bit_count is available in newer numpy. Let's assume it is or use popcount table?
    # For int64, we can use a lookup or just bin().count.
    # Since we do this many times, speed matters.
    # Let's use a helper.
    
    # Actually, for branching, we don't need exact gain for all, just relative order.
    # We can pick the first few candidates and sort them.
    
    # Let's use a simpler heuristic for branching order:
    # Pick u.
    # Candidates = ball[u].
    # For each c in candidates, calculate gain.
    # Sort by gain descending.
    # Branch on c.
    
    # This is expensive: |ball[u]| * cost_of_gain.
    # |ball[u]| = 17.
    # cost_of_gain = 103 ANDs + 103 popcounts.
    # 17 * 103 ~ 1700 ops.
    # 486 levels * 1700 ~ 800k ops.
    # This is very fast!
    
    # Wait, is it?
    # At each node, we do this.
    # Number of nodes?
    # If pruning is effective, it should be manageable.
    
    # Let's implement it.
    
    # To avoid recursion limit issues or overhead, we can use a stack.
    # But recursion is cleaner.
    # sys.setrecursionlimit(10000)
    
    # We need to handle the case where we find a better solution.
    
    # One issue: Python recursion overhead.
    # Let's try it.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # To speed up popcount:
    # We can use int.bit_count() in Python 3.10+
    # Or use a precomputed table for 16-bit chunks.
    # Let's use int.bit_count() if available, else fallback.
    try:
        (1).bit_count()
        def popcount_int(x):
            return x.bit_count()
    except AttributeError:
        def popcount_int(x):
            return bin(x).count('1')
            
    def popcount_np(arr):
        # arr is np.int64 array
        # Convert to list of ints?
        # Or use np.unpackbits?
        # Let's just sum popcount of each element.
        total = 0
        for v in arr:
            total += popcount_int(int(v))
        return total
        
    # Even better: numpy doesn't have vectorized popcount for int64 easily in older versions.
    # Let's stick to the loop. 103 iterations is fine.
    
    # DFS
    def dfs(current_set, unc_bits, depth):
        nonlocal best_solution, best_size
        
        if time.time() > deadline_bnb:
            return
            
        # Pruning
        if len(current_set) >= best_size:
            return
            
        if is_empty(unc_bits):
            # Found a valid cover
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_solution = current_set[:]
                # If we hit the theoretical lower bound or a very good number, we can stop?
                # Lower bound is 403.
                # If we hit 486, it's optimal.
                if best_size <= 486:
                    # We can't easily stop the whole search from here without a flag.
                    # We'll just return and let the caller check.
                    pass
            return
            
        # Find an uncovered point
        u = first_set_bit(unc_bits)
        if u == -1:
            # Should be caught by is_empty
            return
            
        # Get candidates
        candidates = ball_lists[u]
        
        # Calculate gains and sort
        # To save time, if we have many candidates, we might just pick the top few?
        # No, exact BnB needs all.
        
        scored_candidates = []
        for c in candidates:
            # Check if c is already in current_set?
            # We don't need to add it if it is.
            # But if c is in current_set, gain is 0 (since it's already covered by c? No, unc_bits excludes covered).
            # If c is in current_set, it was added previously.
            # If u is uncovered, c cannot be in current_set?
            # Wait. If c is in current_set, then ball[c] is covered.
            # So u (which is in ball[c]) would be covered.
            # Contradiction. So c is NOT in current_set.
            
            inter = np.bitwise_and(unc_bits, balls[c])
            gain = popcount_np(inter)
            if gain > 0:
                scored_candidates.append((gain, c))
                
        if not scored_candidates:
            # No candidate covers u? Impossible since u is in ball[u] and u is a candidate.
            return
            
        # Sort by gain descending
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Branch
        for gain, c in scored_candidates:
            if time.time() > deadline_bnb:
                return
                
            # Heuristic pruning:
            # If len(current_set) + 1 + lower_bound(remaining) >= best_size, skip?
            # Simple lower bound: ceil(popcount(unc_bits) / max_ball_size)
            # max_ball_size = 17
            remaining_count = popcount_np(unc_bits)
            lb = (remaining_count + 16) // 17
            
            if len(current_set) + 1 + lb >= best_size:
                # This branch cannot beat best_size
                # But we must be careful: lb is for the REMAINING uncovered after adding c?
                # No, remaining_count is BEFORE adding c.
                # After adding c, remaining is roughly remaining_count - gain.
                # So lb_after = ceil((remaining_count - gain) / 17)
                lb_after = ((remaining_count - gain) + 16) // 17
                if len(current_set) + 1 + lb_after >= best_size:
                    continue
                    
            # Add c
            new_unc = np.bitwise_and(unc_bits, np.bitwise_not(balls[c]))
            current_set.append(c)
            dfs(current_set, new_unc, depth + 1)
            current_set.pop()
            
            # If we found a solution that is optimal (e.g. 486), we can break?
            # We don't know if 486 is optimal without proof, but it's the best known.
            # If we find 486, we can stop searching for better?
            # Yes, if we are confident.
            # But let's let it run to see if it finds < 486 (unlikely).
            
    # Start DFS
    # Use greedy set as initial? No, start from empty to find smaller?
    # No, BnB starts from empty and adds.
    # But we have a good upper bound from greedy.
    # We can also use the greedy set to guide the search?
    # Standard BnB:
    
    # To speed up, we can order the initial candidates?
    # The first level branches on all 6561 words?
    # No, the first level picks u (first uncovered, which is 0).
    # Candidates are ball[0] (17 words).
    # So the first level has 17 branches.
    # This is efficient.
    
    dfs([], uncovered_np, 0)
    
    if best_solution:
        return best_solution
    else:
        return greedy_set

def _construct_ternary_8_1(q, n, radius, deadline):
    """
    Specialized constructor for q=3, n=8, radius=1.
    Uses BnB.
    """
    total = q ** n
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Run BnB
    best_indices = _solve_covering_code_bnb(q, n, radius, deadline)
    
    # Convert to digits
    codewords = []
    for idx in best_indices:
        codewords.append(_int_to_digits(idx, q, n, powers))
        
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills after 150s. Use 145s deadline for safety.
    deadline = start_time + 145
    
    try:
        # Use the specialized BnB
        result = _construct_ternary_8_1(q, n, radius, deadline)
    except Exception as e:
        # Fallback to simple greedy if BnB fails or times out badly
        try:
            total = q ** n
            powers = [1] * n
            for i in range(1, n):
                powers[i] = powers[i-1] * q
                
            # Simple greedy fallback
            # Precompute balls
            balls = []
            for x in range(total):
                ball = [x]
                for i in range(n):
                    old_digit = (x // powers[i]) % q
                    for new_digit in range(q):
                        if new_digit != old_digit:
                            y = x + (new_digit - old_digit) * powers[i]
                            ball.append(y)
                balls.append(ball)
                
            covered = np.zeros(total, dtype=bool)
            selected = []
            while not np.all(covered) and time.time() < deadline - 5:
                unc_idx = np.where(~covered)[0]
                if len(unc_idx) == 0:
                    break
                target = random.choice(unc_idx.tolist())
                selected.append(target)
                for y in balls[target]:
                    covered[y] = True
                    
            codewords = []
            for idx in selected:
                codewords.append(_int_to_digits(idx, q, n, powers))
            result = codewords
        except Exception:
            # Ultimate fallback: raise
            raise e

    return result

# EVOLVE-BLOCK-END

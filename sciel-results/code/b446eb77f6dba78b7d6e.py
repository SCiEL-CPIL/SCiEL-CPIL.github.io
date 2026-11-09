import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Fallback for non-target parameters
    if not (q == 3 and n == 8 and radius == 1):
        # Use the seed logic for other cases
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Handle small n generically if needed, though task is fixed
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 5.0 # Leave 5s for cleanup and verification
    
    # Precompute balls as bitsets
    # ball[i] is a bitset of indices covered by codeword i
    balls = [0] * N
    
    # Helper to convert int to base-q digits
    def int_to_digits(num, length):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % q
            num //= q
        return digits

    # Helper to convert base-q digits to int
    def digits_to_int(digits):
        val = 0
        for i in range(len(digits) - 1, -1, -1):
            val = val * q + digits[i]
        return val

    # Precompute balls
    for i in range(N):
        digits = int_to_digits(i, n)
        mask = 1 << i
        for k in range(n):
            orig_val = digits[k]
            for v in range(q):
                if v == orig_val:
                    continue
                new_digits = digits.copy()
                new_digits[k] = v
                j = digits_to_int(new_digits)
                mask |= (1 << j)
        balls[i] = mask

    # Phase 1: Greedy Construction
    # We want to cover all N points.
    # uncovered is a bitset of points not yet covered.
    uncovered = (1 << N) - 1
    codewords = []
    
    # To speed up greedy, we can just pick the one that covers most uncovered.
    # Since N=6561, we can do this efficiently.
    while uncovered:
        if time.time() - start_time > time_limit - safety_margin:
            break
        best_idx = -1
        max_covered = 0
        # Iterate through all candidates? That's 6561 * 6561 bits operations.
        # 6561 * (6561/64) ~ 670k operations per step.
        # Number of steps ~ 486. Total ~ 300M operations. Might be slow in Python.
        
        # Optimization: Random sampling or smarter selection?
        # Actually, for the first few, it's fast. As uncovered shrinks, we can restrict candidates.
        # But let's stick to a simple loop first, maybe vectorized with numpy if needed.
        # Given the time limit, let's try a simple loop first.
        
        # We can break early if we find a codeword that covers ALL uncovered? Unlikely.
        for i in range(N):
            # If i is already in codewords, skip? No, we can reuse? No, distinct codewords.
            # But if i is in codewords, its coverage is already accounted for in 'covered', 
            # so 'uncovered & balls[i]' would be 0 if it was the best before.
            # Actually, we just need to find the max.
            covered = uncovered & balls[i]
            # Count bits
            # Python 3.10+ has int.bit_count()
            count = covered.bit_count()
            if count > max_covered:
                max_covered = count
                best_idx = i
                if max_covered == uncovered.bit_count():
                    break
        
        if best_idx == -1:
            break
            
        codewords.append(best_idx)
        uncovered &= ~balls[best_idx]
        
    if not uncovered:
         # We have a valid covering
         pass
    else:
         # Fallback if greedy failed (shouldn't happen)
         return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Phase 2: Local Search (Simulated Annealing)
    # State: set of codewords
    # Objective: Minimize len(codewords)
    # Moves:
    # 1. Remove a redundant codeword.
    # 2. Swap a codeword with another that improves "uniqueness".
    
    # We maintain a count array: counts[i] = number of times point i is covered.
    counts = np.zeros(N, dtype=np.int32)
    for c in codewords:
        # Extract bits from balls[c]
        # This is slow if done one by one.
        # Alternative: Use numpy to extract bits?
        # Or just iterate through set bits.
        mask = balls[c]
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            mask ^= lsb
            
    cw_set = set(codewords)
    current_size = len(codewords)
    best_size = current_size
    best_codewords = list(codewords)
    
    # Annealing parameters
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    def get_unique_count(c):
        # Number of points covered ONLY by c
        # This requires knowing counts.
        # If we remove c, points with counts[i] == 1 become 0.
        # So unique points = number of i where counts[i] == 1 and c covers i.
        mask = balls[c]
        # We need to count bits in mask that correspond to indices where counts == 1.
        # This is hard to do with simple bit ops on numpy array.
        # Let's precompute a bitset of "critical" points?
        # Critical points are those with count == 1.
        # But counts change.
        
        # Alternative: 
        # Let's just calculate the "gain" of a swap or removal directly using counts.
        pass

    # Simulated Annealing Loop
    # To make it fast, we need efficient move generation and evaluation.
    
    # Precompute "neighbors" for swap? No, too many.
    # Randomly pick a codeword to remove/swap.
    
    # Helper to update counts and critical mask
    # critical_mask: bitset of points with count == 1
    
    # Let's use a different metric: 
    # When removing c:
    #   Decrement counts for all points in balls[c].
    #   If any count becomes 0, it's invalid.
    #   If no count becomes 0, it's a valid removal (size -1).
    
    # When swapping c with d:
    #   Remove c, add d.
    #   Size stays same.
    #   Why do this? To enable future removals.
    #   We want d to cover points that c covers, but also cover points that are "loosely covered" 
    #   or to free up "tight" points?
    #   Actually, we want to increase the "slack" of the covering.
    
    # Let's try a simpler heuristic: 
    # 1. Try to remove any redundant codeword.
    # 2. If stuck, try to swap a codeword with a random one that covers many "low count" points.
    
    # We'll use a time-based loop.
    
    while time.time() - start_time < time_limit - safety_margin:
        # 1. Try to remove a redundant codeword
        removed = False
        # Randomly shuffle codewords to avoid bias
        random.shuffle(codewords)
        
        for c in codewords:
            mask = balls[c]
            # Check if all points in mask have count > 1
            # We need to know if min(counts[i] for i in mask) > 1
            
            # Efficient check:
            # If we have a bitset of points with count == 1 (critical_mask)
            # Then if mask & critical_mask == 0, c is redundant.
            
            # We need to maintain critical_mask.
            # This is expensive to update.
            
            # Let's just do the count check directly for now, might be slow.
            # Optimization: Only check a few random candidates per iteration.
            pass
        
        # This direct check is O(N) per candidate. 486 candidates * 6561 points = 3M ops.
        # In Python, 3M ops is ~0.1s. 
        # If we do this 1000 times, it's 100s. Too slow.
        
        # We need to maintain critical_mask (bitset of points with count == 1).
        
        # Let's restart with a structure that maintains critical_mask.
        
    # --- Revised Approach ---
    
    # State:
    # counts: np.array of int
    # critical_mask: int (bitset of indices where counts[i] == 1)
    
    # Re-initialize counts and critical_mask
    counts = np.zeros(N, dtype=np.int32)
    for c in codewords:
        mask = balls[c]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            mask ^= lsb
            
    critical_mask = 0
    for i in range(N):
        if counts[i] == 1:
            critical_mask |= (1 << i)
    
    cw_set = set(codewords)
    current_size = len(codewords)
    best_size = current_size
    best_codewords = list(codewords)
    
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    steps = 0
    
    while time.time() - start_time < time_limit - safety_margin:
        steps += 1
        
        # Move 1: Try to remove a redundant codeword
        # A codeword c is redundant if balls[c] & critical_mask == 0
        # We can find redundant codewords quickly?
        # We can iterate through codewords and check.
        
        # To speed up, we can pick a random codeword and check.
        # If it's redundant, remove it.
        
        # Move 2: Swap
        # Pick c from codewords.
        # Pick d from all N \ codewords.
        # Swap if it improves some metric.
        
        # Let's try a random walk on the space of coverings.
        
        # Pick a random codeword c
        c = random.choice(codewords)
        
        # Check if c is redundant
        if balls[c] & critical_mask == 0:
            # Remove c
            codewords.remove(c)
            cw_set.discard(c)
            current_size -= 1
            
            # Update counts and critical_mask
            mask = balls[c]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                counts[idx] -= 1
                if counts[idx] == 1:
                    critical_mask |= (1 << idx)
                elif counts[idx] == 0:
                    # Should not happen if c was redundant
                    pass
                else:
                    # counts[idx] was > 1, now >= 1. 
                    # If it was 1, it's now 0 (impossible for redundant).
                    # If it was > 1, it's now >= 1.
                    # Wait, if counts[idx] was 2, it becomes 1. So it becomes critical.
                    # If counts[idx] was 3, it becomes 2. Not critical.
                    # So:
                    if counts[idx] == 1:
                        critical_mask |= (1 << idx)
                    # If counts[idx] was 1, it becomes 0. But c was redundant, so this doesn't happen.
                mask ^= lsb
            
            if current_size < best_size:
                best_size = current_size
                best_codewords = list(codewords)
            continue
        
        # If not redundant, try to swap
        # Pick a random candidate d
        d = random.randint(0, N - 1)
        if d in cw_set:
            continue
            
        # Evaluate swap c -> d
        # We need to check if the new covering is valid.
        # Valid if (critical_mask \ balls[c]) & ~balls[d] == 0?
        # Let's trace:
        # Points covered only by c are in critical_mask & balls[c].
        # If we remove c, these points lose their last cover.
        # They must be covered by d.
        # So we need: (critical_mask & balls[c]) & ~balls[d] == 0
        
        if (critical_mask & balls[c]) & ~balls[d] == 0:
            # Swap is valid
            codewords[codewords.index(c)] = d
            cw_set.discard(c)
            cw_set.add(d)
            
            # Update counts and critical_mask
            # Remove c
            mask_c = balls[c]
            while mask_c:
                lsb = mask_c & -mask_c
                idx = lsb.bit_length() - 1
                counts[idx] -= 1
                if counts[idx] == 1:
                    critical_mask |= (1 << idx)
                # else counts[idx] >= 2 or 0 (if it was 1, now 0, but we checked validity so no 0s)
                mask_c ^= lsb
                
            # Add d
            mask_d = balls[d]
            while mask_d:
                lsb = mask_d & -mask_d
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                if counts[idx] == 2:
                    # Was 1, now 2. No longer critical.
                    critical_mask &= ~(1 << idx)
                mask_d ^= lsb
            
            # Swap doesn't change size, but might enable future removals.
            # We accept this swap if it reduces the "critical load" or something?
            # For now, we accept it with probability exp(-1/T) or just always?
            # Since size is same, we accept if it increases "slack".
            # Slack = sum(1/counts[i])? 
            # Let's just accept it to explore.
            pass
        else:
            # Invalid swap, reject
            pass
            
    # Restore best found
    codewords = best_codewords
    
    # Final Polish: Greedy Removal
    counts = np.zeros(N, dtype=np.int32)
    for c in codewords:
        mask = balls[c]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            mask ^= lsb
            
    changed = True
    while changed:
        changed = False
        new_codewords = []
        for c in codewords:
            mask = balls[c]
            # Check if redundant
            redundant = True
            # Check if any point in mask has count == 1
            # We can just decrement and check min?
            # Or use critical_mask logic again?
            # Let's just decrement and check.
            temp_counts = counts.copy() # Expensive!
            # Better: 
            # We know counts.
            # A point i in mask is critical if counts[i] == 1.
            # If any such point exists, c is not redundant.
            
            # Efficient check:
            # We can precompute a bitset of critical points?
            # Let's just loop through bits of mask and check counts.
            # Max 17 bits.
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    redundant = False
                    break
                m ^= lsb
                
            if redundant:
                # Remove c
                # Update counts
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_codewords.append(c)
        codewords = new_codewords
        
    # Convert back to lists of integers
    result = []
    for c in codewords:
        digits = int_to_digits(c, n)
        result.append(digits[::-1])

    # Verify coverage (safety check)
    if time.time() - start_time < time_limit - safety_margin:
        covered = np.zeros(N, dtype=np.uint8)
        for c in codewords:
            mask = balls[c]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                covered[idx] = 1
                mask ^= lsb
        if not covered.all():
            # Fallback to full space if verification fails (shouldn't happen)
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    return result

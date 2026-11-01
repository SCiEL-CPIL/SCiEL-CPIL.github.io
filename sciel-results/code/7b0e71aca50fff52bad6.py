import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Specific optimization for the known task: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
        
    # Fallback for other parameters: simple greedy + pruning
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    N = 2187  # 3^7
    # Precompute powers of 3 for fast index conversion
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    # Precompute ball_indices: for each word index, list of indices within Hamming distance 1
    ball_indices = []
    for i in range(N):
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        
        ball = [i]
        for pos in range(7):
            orig = digits[pos]
            for val in range(3):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    ball.append(new_i)
        ball_indices.append(ball)
    
    # Convert to numpy arrays for vectorized operations
    # We need a fixed-size array for vectorization in local search? 
    # Ball size is always 1 + 7*2 = 15.
    ball_indices_np = np.array(ball_indices, dtype=np.int32)
    
    # Precompute "covers" as bitsets? 
    # N=2187 fits in a Python int (arbitrary precision).
    # This allows very fast set intersection/union operations.
    covers = [0] * N
    for i in range(N):
        mask = 0
        for j in ball_indices[i]:
            mask |= (1 << j)
        covers[i] = mask

    # Helper: Greedy Set Cover initialization
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = (1 << N) - 1  # Full mask
        selected = []
        selected_set = set()
        
        while uncovered:
            best_idx = -1
            best_count = -1
            best_candidates = []
            
            # To make it randomized, we can shuffle indices or just pick random ties.
            # For speed, we'll just pick the first best or random among bests.
            
            # Iterate all candidates
            # Optimization: Skip already selected
            # Since N=2187 is small, a full scan is okay, but we can do better.
            # Let's just scan all.
            
            candidates = list(range(N))
            rng.shuffle(candidates)
            
            for i in candidates:
                if i in selected_set:
                    continue
                    
                # Count uncovered bits covered by i
                cnt = (uncovered & covers[i]).bit_count()
                
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    best_candidates = [i]
                elif cnt == best_count and cnt > 0:
                    # Random tie-break: keep a list of bests and pick one at the end?
                    # Or just probabilistically switch?
                    # To avoid O(N^2) list appends in worst case, let's just pick the first one found in random order?
                    # No, we want the *maximum* count.
                    # If we find a strictly better one, we reset.
                    # If we find an equal one, we can store it.
                    # But storing all ties is expensive.
                    # Let's just pick the first one with max count in the shuffled order.
                    # This effectively randomizes the tie-breaking.
                    pass 
                    
            if best_idx == -1 or best_count == 0:
                break # Should not happen if uncovered.any()
                
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~covers[best_idx]
            
            if time.time() > deadline - 10.0:
                break
                
        return selected

    # Helper: Check if valid
    def is_valid(code):
        if not code:
            return N == 0
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt[ball_indices_np[c]] += 1
        return (cnt > 0).all()

    # Helper: Local Search (Delete + Swap)
    def local_search(code, seed):
        rng = random.Random(seed)
        code_set = set(code)
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt[ball_indices_np[c]] += 1
        
        best_code = code.copy()
        best_size = len(code)
        no_improve = 0
        
        while time.time() < deadline - 5.0 and no_improve < 30:
            # Phase 1: Deletion
            changed = False
            # Try to remove random codewords
            indices_to_try = list(range(len(code)))
            rng.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                if idx >= len(code): continue
                c = code[idx]
                # Check if all neighbors are covered by >= 2
                if cnt[ball_indices_np[c]].min() >= 2:
                    # Safe to remove
                    cnt[ball_indices_np[c]] -= 1
                    code.pop(idx)
                    code_set.remove(c)
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
            
            if not changed:
                no_improve += 1
                if no_improve >= 30:
                    break
            else:
                no_improve = 0
            
            # Phase 2: Swap
            if time.time() > deadline - 5.0:
                break
                
            # Try a few swaps
            swaps_attempted = 0
            max_swaps = 150
            while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
                swaps_attempted += 1
                if not code:
                    break
                
                # Pick a random codeword to remove
                c = rng.choice(code)
                
                # Find words that would become uncovered
                # U = {w in ball(c) | cnt[w] == 1}
                ball_c = ball_indices_np[c]
                U = ball_c[cnt[ball_c] == 1]
                
                if len(U) == 0:
                    # c is redundant
                    cnt[ball_c] -= 1
                    code.remove(c)
                    code_set.remove(c)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
                    continue
                
                # Find candidates c' that cover all words in U
                # c' must be in ball(w) for all w in U.
                # Intersection of balls of U.
                
                # Get set of candidates from first word in U
                candidates = set(ball_indices_np[U[0]].tolist())
                
                for w in U[1:]:
                    candidates &= set(ball_indices_np[w].tolist())
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                    
                # Filter out candidates already in code
                valid_candidates = [c_prime for c_prime in candidates if c_prime not in code_set]
                
                if not valid_candidates:
                    continue
                    
                # Pick a random valid candidate
                c_prime = rng.choice(valid_candidates)
                
                # Perform swap
                # Remove c
                cnt[ball_indices_np[c]] -= 1
                code.remove(c)
                code_set.remove(c)
                
                # Add c_prime
                cnt[ball_indices_np[c_prime]] += 1
                code.append(c_prime)
                code_set.add(c_prime)
                
                # Immediate check for deletions after swap?
                # The outer loop will handle it, but doing it immediately might help escape local minima faster.
                # Let's do a quick pass.
                # Actually, just letting the outer loop run is fine.
                
        return best_code

    # Main Logic
    best_code = None
    best_size = N + 1
    
    # 1. Deterministic Greedy
    code = greedy_init(0)
    if is_valid(code):
        code = local_search(code, 0)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
    
    # 2. Randomized Greedies + Local Search
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        code = greedy_init(attempts)
        if not is_valid(code):
            continue
            
        code = local_search(code, attempts)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            # If we reached the known optimum 186, we can stop early?
            # The problem says best known is 186.
            if best_size <= 186:
                break
                
    if best_code is None:
        # Fallback: return a large valid code
        # Just return all words? No, too big.
        # Return product code
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = []
        for h in tetracode:
            for tail in np.ndindex(3, 3, 3):
                code.append(h + list(tail))
        best_code = code
        best_size = len(code)

    # Convert indices back to lists of ints
    result = []
    for idx in best_code:
        digits = [0] * 7
        tmp = idx
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        result.append(digits)
        
    return result

def _construct_general(q, n, radius, deadline):
    # Simple fallback for other parameters
    # Greedy Set Cover
    N = q ** n
    if N > 100000:
        # Too large for simple greedy in 120s? 
        # Just return a small random sample? No, must be valid.
        # For general case, we might not find a good code.
        # But the task is specific to q=3, n=7.
        pass
    
    # Generate all words
    words = []
    for t in __import__('itertools').product(range(q), repeat=n):
        words.append(list(t))
        
    # Map word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute balls
    balls = []
    for i, w in enumerate(words):
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = w.copy()
                    new_w[pos] = val
                    ball.append(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    # Greedy
    uncovered = set(range(N))
    selected = []
    while uncovered:
        best_idx = -1
        best_count = -1
        for i in range(N):
            if i in selected: continue
            count = len(uncovered & set(balls[i]))
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1: break
        selected.append(best_idx)
        uncovered -= set(balls[best_idx])
        
    return [words[i] for i in selected]

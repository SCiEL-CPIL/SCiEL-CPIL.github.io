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
    # Also precompute digits for each index to speed up ball generation if needed, 
    # but here we can compute ball directly using index arithmetic.
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
    ball_indices_np = np.array(ball_indices, dtype=np.int32)
    
    # Precompute "covers" as bitsets? No, N=2187 is too large for Python ints in a list 
    # to be efficient for bit_count in a tight loop without overhead. 
    # Instead, we maintain a coverage count array.
    
    # Helper: Greedy Set Cover initialization
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = np.ones(N, dtype=bool)
        selected = []
        # Precompute coverage masks as lists of indices for each codeword? 
        # We have ball_indices_np which is exactly that.
        
        while uncovered.any():
            best_idx = -1
            best_count = -1
            # Sample a subset of candidates to speed up greedy? 
            # Or iterate all. 2187 iterations * 15 checks = ~30k ops per step.
            # Max steps ~ 200. Total ~ 6M ops. Fast enough.
            
            # To make it randomized, we can shuffle indices or just pick random ties.
            # For speed, we'll just pick the first best or random among bests.
            
            # Vectorized count of covered words for all candidates?
            # counts[i] = sum(uncovered[ball_indices_np[i]])
            # This is a matrix-vector product if we had a binary matrix.
            # Let's just loop, it's small enough.
            
            # Optimization: Only check candidates that cover at least one uncovered word.
            # We can maintain a list of "active" candidates.
            # Simpler: Just loop all N.
            
            # Using numpy sum is fast
            # counts = np.zeros(N)
            # for i in range(N):
            #    counts[i] = uncovered[ball_indices_np[i]].sum()
            
            # Even faster: 
            # We can use the fact that ball size is fixed.
            
            best_count = -1
            candidates = list(range(N))
            rng.shuffle(candidates)
            
            # Check a random subset first to find a good candidate quickly?
            # Let's just check all, it's only 2187.
            for i in candidates:
                # If i is already selected, skip? 
                # Actually, adding a duplicate is useless. 
                # We should skip selected ones.
                if i in selected: # O(N) check, bad.
                    continue
                    
                # Use a set for selected?
                # Let's maintain selected_set
                
                cnt = uncovered[ball_indices_np[i]].sum()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                elif cnt == best_count and cnt > 0:
                    # Random tie-break
                    if rng.random() < 0.5:
                        best_idx = i
                        
            if best_idx == -1 or best_count == 0:
                break # Should not happen if uncovered.any()
                
            selected.append(best_idx)
            # Update uncovered
            uncovered[ball_indices_np[best_idx]] = False
            
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
        
        # Precompute for swap logic:
        # For a given word w, which codewords cover it?
        # This is expensive to maintain dynamically.
        # Instead, for swap:
        # 1. Pick c to remove.
        # 2. Find U = {w in ball(c) | cnt[w] == 1}. These are the words that would be lost.
        # 3. Find c' not in code_set such that U is a subset of ball(c').
        #    i.e., for all w in U, w is in ball(c').
        #    This means c' must be in the intersection of balls of all w in U? No.
        #    c' must cover all w in U.
        #    So c' must be in ball(w) for all w in U.
        #    So c' is in Intersection_{w in U} ball(w).
        #    If U is empty, any c' works.
        
        while time.time() < deadline - 5.0 and no_improve < 20:
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
                if no_improve >= 20:
                    break
            else:
                no_improve = 0
            
            # Phase 2: Swap
            if time.time() > deadline - 5.0:
                break
                
            # Try a few swaps
            swaps_attempted = 0
            max_swaps = 100
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
                    # c is redundant, but we didn't remove it in Phase 1?
                    # Maybe Phase 1 missed it due to order? 
                    # Just remove it.
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
                # Start with ball(U[0])
                # Since |U| is small (<= 15), and ball size is 15, intersection is small.
                
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
                
                # Check if this swap allowed further deletions?
                # We don't check immediately, the next iteration of the while loop will do Phase 1.
                # But we should check if the swap itself improved the size? No, size is same.
                # The improvement comes from subsequent deletions.
                
                # To avoid infinite loops of bad swaps, we can track if we made progress.
                # But since we don't know if deletions will happen, we just let the loop run.
                # To prevent cycling, we can limit swaps per "stagnation" period.
                
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

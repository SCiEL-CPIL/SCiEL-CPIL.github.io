import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Constructs a covering code using:
    1. Greedy Set Cover with randomization
    2. Aggressive Local Search with Pruning, Swaps, and Add-Remove moves
    """
    start_time = time.time()
    timeout = 115  # Safety margin from 120s limit
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]
    
    total_words = 3 ** 7
    n_words = total_words
    
    # Precompute digits and balls
    digits = np.zeros((n_words, n), dtype=int)
    for i in range(n_words):
        temp = i
        for k in range(n):
            digits[i, k] = temp % 3
            temp //= 3
            
    # Precompute balls as lists of indices
    balls = []
    powers3 = [1, 3, 9, 27, 81, 243, 729]
    for i in range(n_words):
        ball = [i]
        for pos in range(n):
            orig_digit = digits[i, pos]
            p = powers3[pos]
            base_val = i - orig_digit * p
            for new_digit in range(3):
                if new_digit == orig_digit:
                    continue
                j = base_val + new_digit * p
                ball.append(j)
        balls.append(ball)
        
    def greedy_cover(seed):
        """Run greedy set cover with randomization."""
        rng = random.Random(seed)
        uncovered = np.ones(n_words, dtype=bool)
        code = []
        
        while np.any(uncovered):
            if time.time() - start_time > timeout:
                break
            best_idx = -1
            max_cov = 0
            best_candidates = []
            
            for i in range(n_words):
                cov = 0
                for j in balls[i]:
                    if uncovered[j]:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_candidates = [i]
                    if max_cov == 15:
                        break
                elif cov == max_cov and max_cov > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
            best_idx = rng.choice(best_candidates)
            code.append(best_idx)
            for j in balls[best_idx]:
                uncovered[j] = False
                
        return code
    
    def local_search(code, seed):
        """
        Apply local search with:
        1. Pruning
        2. Swaps
        3. Add-Remove (Grow then Prune)
        """
        if time.time() - start_time > timeout:
            return code
            
        count = np.zeros(n_words, dtype=int)
        for c in code:
            for j in balls[c]:
                count[j] += 1
                
        code_set = set(code)
        rng = random.Random(seed)
        
        # Helper to prune
        def prune():
            nonlocal code, count, code_set
            new_code = []
            changed = False
            for c in code:
                redundant = True
                for j in balls[c]:
                    if count[j] < 2:
                        redundant = False
                        break
                if redundant:
                    for j in balls[c]:
                        count[j] -= 1
                    code_set.discard(c)
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                code = new_code
            return changed

        prune()
        
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            
            # 1. Prune
            if prune():
                improved = True
                continue
            
            # 2. Try Swaps
            swap_done = False
            for _ in range(50): # More swap attempts
                if time.time() - start_time > timeout:
                    return code
                if len(code) == 0:
                    break
                
                c_idx = rng.randint(0, len(code) - 1)
                c = code[c_idx]
                w = rng.randint(0, n_words - 1)
                
                if w == c or w in code_set:
                    continue
                    
                # Check validity: B(c) \ B(w) must be covered by others
                valid = True
                for j in balls[c]:
                    if j not in balls[w] and count[j] < 2:
                        # j is covered only by c, and w doesn't cover j
                        # So we cannot remove c
                        valid = False
                        break
                        
                if valid:
                    # Perform swap
                    for j in balls[c]:
                        count[j] -= 1
                    for j in balls[w]:
                        count[j] += 1
                    code[c_idx] = w
                    code_set.discard(c)
                    code_set.add(w)
                    improved = True
                    swap_done = True
                    break # Restart loop to prune
            
            if swap_done:
                continue
            
            # 3. Try Add-Remove
            # Add a random word, then prune. If size decreases, keep.
            if len(code) < n_words:
                w = rng.randint(0, n_words - 1)
                if w not in code_set:
                    # Save state to revert if needed
                    old_code = code
                    old_count = count.copy()
                    old_set = code_set.copy()
                    
                    # Add w
                    code.append(w)
                    code_set.add(w)
                    for j in balls[w]:
                        count[j] += 1
                    
                    # Prune
                    prune()
                    
                    # Check if size improved
                    if len(code) < len(old_code):
                        improved = True
                        # Keep the new state
                    else:
                        # Revert
                        code = old_code
                        count = old_count
                        code_set = old_set
                        # We might accept equal size to change structure, but rarely
                        # if rng.random() < 0.1:
                        #     improved = True
                        #     code = old_code # Wait, we reverted.
                        #     # Actually, if we want to accept equal size, we shouldn't revert
                        #     # But for now, let's stick to strict improvement to avoid noise
                        #     # Or we can accept with probability
                        #     if rng.random() < 0.05:
                        #         improved = True
                        #         # Re-apply the move? 
                        #         # This is getting complex. Let's just revert.
                        pass
                        
        return code

    best_code = None
    best_size = float('inf')
    
    num_trials = 15
    for trial in range(num_trials):
        if time.time() - start_time > timeout:
            break
            
        seed = 100 + trial
        code = greedy_cover(seed)
        
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
            
        code = local_search(code, seed)
        
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
            
        # If we have time, run local search on the current best again
        if best_size <= 200 and time.time() - start_time < timeout - 10:
            code = local_search(list(best_code), seed + 1000)
            if len(code) < best_size:
                best_code = code
                best_size = len(code)
    
    if best_code is None:
        best_code = list(range(n_words))
        
    result = [list(digits[c]) for c in best_code]
    return result

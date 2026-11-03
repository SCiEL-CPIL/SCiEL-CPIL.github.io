import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118  # 2s buffer for safety
    
    N = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors as bitmasks (Python ints)
    # N=6561, so we need ~103 64-bit chunks, but Python ints handle arbitrary length efficiently.
    neighbor_masks = [0] * N
    
    # Precompute word indices for each codeword's neighborhood
    # This is needed for cover count updates
    neighbor_indices = [[] for _ in range(N)]
    
    for c_idx in range(N):
        # Generate all words within Hamming distance 'radius'
        # For radius=1, n=8, q=3: 1 + 8*2 = 17 neighbors
        mask = 0
        indices = []
        
        # The word itself
        mask |= (1 << c_idx)
        indices.append(c_idx)
        
        if radius == 1:
            c_word = idx_to_word(c_idx)
            for i in range(n):
                orig = c_word[i]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = c_word.copy()
                        new_word[i] = new_val
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        indices.append(w_idx)
        else:
            # General case (not expected for this task, but good for robustness)
            # We skip complex precomputation for radius > 1 to save time, 
            # assuming task is radius=1 as per description "differs ... in at most one position"
            pass
            
        neighbor_masks[c_idx] = mask
        neighbor_indices[c_idx] = indices

    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
            
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered word to be the "target"
            # To make it efficient, we can pick the word with fewest potential coders
            # But for speed, let's just pick a random uncovered word and find the best coder for it.
            
            # Get a random uncovered word
            # We need to iterate to find one.
            temp = uncovered
            # Pick a random bit
            # Efficient way: get bit length, pick random position
            # But simpler: just loop. N=6561 is small.
            # Actually, let's use the criticality heuristic for better initial quality.
            
            # Find word with min criticality among uncovered
            # Precompute criticality (number of codewords covering each word)
            # This is static, so compute once outside? 
            # No, criticality is static (depends on geometry, not current code).
            # Criticality[w] = number of c such that w in neighbor_indices[c]
            
            # We didn't precompute criticality. Let's do it now or on the fly.
            # On the fly is too slow. Let's precompute it.
            
            # Wait, I can't precompute it in the function scope easily without extra memory/time.
            # Let's just use a standard greedy: pick word with max new coverage? No, that's for max coverage.
            # For set cover, pick word with min candidates.
            
            # Let's just pick a random uncovered word for diversity in restarts.
            # To pick a random index from a bitmask:
            # 1. Count bits
            # 2. Pick random k
            # 3. Find k-th set bit
            # This is O(N) per pick. 6561 * 500 picks = 3M ops. Fine.
            
            # Simpler: just iterate and pick the first one? No, that's deterministic.
            # Let's use the "min criticality" heuristic.
            
            # We need criticality array.
            # Let's compute it once.
            
            # (Computation of criticality moved to outer scope)
            
            min_crit = N + 1
            target_word = -1
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                w_idx = lsb.bit_length() - 1
                crit = criticality[w_idx]
                if crit < min_crit:
                    min_crit = crit
                    target_word = w_idx
                if min_crit == 1:
                    break
                temp ^= lsb
            
            if target_word == -1:
                break
                
            # Find best coder for target_word
            best_c = -1
            best_new_count = -1
            
            for c_idx in word_to_coders[target_word]:
                if c_idx in code_set:
                    continue
                new_count = (uncovered & neighbor_masks[c_idx]).bit_count()
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_c = c_idx
            
            if best_c == -1:
                # Fallback: pick any
                for c_idx in word_to_coders[target_word]:
                    if c_idx not in code_set:
                        best_c = c_idx
                        break
                        
            if best_c == -1:
                break
                
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]
            
        return code

    # Precompute word_to_coders and criticality
    word_to_coders = [[] for _ in range(N)]
    for c_idx in range(N):
        for w in neighbor_indices[c_idx]:
            word_to_coders[w].append(c_idx)
            
    criticality = np.array([len(word_to_coders[w]) for w in range(N)], dtype=np.int32)

    def local_search_remove(code):
        """Remove redundant codewords"""
        if not code:
            return code
            
        code_set = set(code)
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_indices[c]:
                cover_counts[w] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
                has_unique = False
                for w in neighbor_indices[c]:
                    if cover_counts[w] == 1:
                        has_unique = True
                        break
                        
                if not has_unique:
                    code.pop(i)
                    for w in neighbor_indices[c]:
                        cover_counts[w] -= 1
                    code_set.discard(c)
                    improved = True
                    break
                    
        return code

    def simulated_annealing(code, max_iters=1000):
        """SA to minimize size, allowing temporary invalid states."""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_indices[c]:
                cover_counts[w] += 1
                
        uncovered_count = np.sum(cover_counts == 0)
        
        # Objective: Size + Penalty * Uncovered
        penalty = 50.0 
        current_obj = len(code) + penalty * uncovered_count
        
        best_code = list(code)
        best_obj = current_obj
        
        temp = 10.0
        cooling = 0.99
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if temp < 0.1:
                break
            
            move_type = random.random()
            
            if move_type < 0.4 and code:
                # Remove
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                
                newly_uncovered = 0
                for w in neighbor_indices[c_remove]:
                    if cover_counts[w] == 1:
                        newly_uncovered += 1
                        
                delta_obj = -1 + penalty * newly_uncovered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or random.random() < np.exp(-delta_obj / temp):
                    code.pop(idx)
                    for w in neighbor_indices[c_remove]:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    current_obj = new_obj
                    uncovered_count += newly_uncovered
                    
            elif move_type < 0.7:
                # Add
                c_add = random.randint(0, N - 1)
                if c_add in code_set:
                    continue
                
                newly_covered = 0
                for w in neighbor_indices[c_add]:
                    if cover_counts[w] == 0:
                        newly_covered += 1
                        
                delta_obj = 1 - penalty * newly_covered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or random.random() < np.exp(-delta_obj / temp):
                    code.append(c_add)
                    for w in neighbor_indices[c_add]:
                        cover_counts[w] += 1
                    code_set.add(c_add)
                    current_obj = new_obj
                    uncovered_count -= newly_covered
                    
            else:
                # Swap
                if not code:
                    continue
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                c_add = random.randint(0, N - 1)
                if c_add in code_set:
                    continue
                    
                # Compute delta
                uncovered_delta = 0
                for w in neighbor_indices[c_remove]:
                    if cover_counts[w] == 1:
                        uncovered_delta += 1
                for w in neighbor_indices[c_add]:
                    if cover_counts[w] == 0:
                        uncovered_delta -= 1
                        
                delta_obj = penalty * uncovered_delta
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or random.random() < np.exp(-delta_obj / temp):
                    code[idx] = c_add
                    for w in neighbor_indices[c_remove]:
                        cover_counts[w] -= 1
                    for w in neighbor_indices[c_add]:
                        cover_counts[w] += 1
                    code_set.discard(c_remove)
                    code_set.add(c_add)
                    current_obj = new_obj
                    uncovered_count += uncovered_delta
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_code = list(code)
                
            temp *= cooling
            
        # Repair if best_code is invalid
        cover_counts_best = np.zeros(N, dtype=np.int32)
        for c in best_code:
            for w in neighbor_indices[c]:
                cover_counts_best[w] += 1
                
        if np.any(cover_counts_best == 0):
            # Greedy repair
            uncovered_mask = (1 << N) - 1
            for c in best_code:
                uncovered_mask &= ~neighbor_masks[c]
                
            while uncovered_mask and time.time() < deadline:
                best_c = -1
                best_count = -1
                # Find codeword covering most uncovered words
                # Optimization: Only check candidates that cover at least one uncovered word
                # We can iterate through all N, N is 6561, this is fast.
                for c in range(N):
                    if c in best_code:
                        continue
                    count = (uncovered_mask & neighbor_masks[c]).bit_count()
                    if count > best_count:
                        best_count = count
                        best_c = c
                if best_c == -1:
                    break
                best_code.append(best_c)
                uncovered_mask &= ~neighbor_masks[best_c]
                
        return best_code

    # Main Loop
    best_code = []
    best_size = float('inf')
    
    # Initial run
    code = greedy_cover()
    code = local_search_remove(code)
    code = simulated_annealing(code, max_iters=500)
    code = local_search_remove(code)
    
    if len(code) < best_size:
        best_code = code
        best_size = len(code)
        
    # Restarts
    restart = 1
    while time.time() < deadline - 5:
        code = greedy_cover(seed=restart * 101)
        code = local_search_remove(code)
        
        # Shorter SA for restarts
        code = simulated_annealing(code, max_iters=200)
        code = local_search_remove(code)
        
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
            
        restart += 1
        
    # Final check
    if not best_code:
        # Fallback: Full greedy
        best_code = greedy_cover()
        
    # Convert indices to words
    result = [idx_to_word(i) for i in best_code]
    return result

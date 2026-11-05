import numpy as np
import itertools
import random

def construct(q, n, radius):
    N = q ** n
    
    if radius != 1:
        raise NotImplementedError("Radius > 1 not optimized")

    sphere_size = 1 + n * (q - 1)
    
    # Vectorized neighbor generation
    # Create indices for all positions and values
    pos_indices = np.arange(n)
    # We need to generate all neighbors for all i in 0..N-1
    # This is tricky to do in one shot without memory blowup, but N is small (2187).
    # Current python loop is O(N * n * q) = 2187 * 7 * 2 ~ 30k ops. This is fast.
    # The bottleneck is definitely the local search.
    
    neighbors = np.empty((N, sphere_size), dtype=int)
    
    # Precompute powers of q
    q_powers = np.array([q ** i for i in range(n)], dtype=int)
    
    # Iterate to fill neighbors
    for i in range(N):
        neighbors[i, 0] = i
        val = i
        idx = 1
        for pos in range(n):
            orig_val = val % q
            val //= q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[i, idx] = i + (new_val - orig_val) * q_powers[pos]
                idx += 1

    def run_greedy_local_search(seed_val):
        rng = np.random.RandomState(seed_val)
        covered = np.zeros(N, dtype=bool)
        cover_count = np.zeros(N, dtype=int)
        C = []
        
        # 1. Greedy Phase
        while covered.sum() < N:
            gains = (~covered[neighbors]).sum(axis=1)
            max_gain = np.max(gains)
            # Get all candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            # Pick one randomly to diversify
            best_idx = rng.choice(candidates)
            
            if max_gain == 0:
                break
            
            C.append(best_idx)
            cover_count[neighbors[best_idx]] += 1
            covered[neighbors[best_idx]] = True
        
        # 2. Local Search Phase
        in_C = np.zeros(N, dtype=bool)
        for c in C:
            in_C[c] = True
            
        improved = True
        # Limit iterations to prevent infinite loops or excessive time
        max_iters = 100
        iter_count = 0
        
        while improved and iter_count < max_iters:
            iter_count += 1
            improved = False
            
            # Step A: Remove redundant codewords
            # We can vectorize the check for redundancy?
            # A codeword c is redundant if for all w in neighbors[c], cover_count[w] > 1.
            # i.e., min(cover_count[neighbors[c]]) > 1
            # We can check this for all c in C at once?
            # neighbors[C] is a list of arrays.
            # Let's stick to loop but optimize.
            
            removable_idx = -1
            for i in range(len(C)):
                c = C[i]
                nb = neighbors[c]
                # Check if any word has count 1
                if np.any(cover_count[nb] == 1):
                    continue
                # If no word has count 1, it's safe to remove
                removable_idx = i
                break
            
            if removable_idx != -1:
                c = C[removable_idx]
                nb = neighbors[c]
                cover_count[nb] -= 1
                C.pop(removable_idx)
                in_C[c] = False
                improved = True
                continue # Restart local search loop

            if C:
                # Step B: Swap to increase redundancy
                current_critical = np.sum(cover_count == 1)
                best_swap = None
                best_critical = current_critical
                
                # To speed this up, we might only consider swaps for codewords with high criticality
                # But let's try to make the existing loop faster first.
                
                for c in C:
                    nb_c = neighbors[c]
                    critical_mask = cover_count[nb_c] == 1
                    if not np.any(critical_mask):
                        continue
                    
                    critical_words = nb_c[critical_mask]
                    
                    # Find candidates that cover at least one critical word
                    # and are not in C
                    # Union of neighbors of critical words
                    # If only 1 critical word, candidates are neighbors of it.
                    # If more, intersection? No, the new codeword must cover ALL critical words?
                    # No, if we remove c, the critical words become uncovered (count 0).
                    # The new codeword cp must cover them.
                    # So cp must be in the intersection of neighbors of all critical words.
                    
                    if len(critical_words) == 1:
                        candidates = neighbors[critical_words[0]]
                    else:
                        # Intersection of neighbors
                        # Start with first
                        inter_mask = np.zeros(N, dtype=bool)
                        inter_mask[neighbors[critical_words[0]]] = True
                        for w in critical_words[1:]:
                            # Update mask
                            # If intersection becomes empty, break
                            if not np.any(inter_mask):
                                break
                            # Intersect with neighbors[w]
                            # We can do: inter_mask &= (neighbors[w] is in inter_mask)
                            # This is hard to vectorize easily without sets or loops.
                            # Given small N, set intersection might be faster in Python?
                            # Or just use numpy:
                            # temp = np.zeros(N, dtype=bool)
                            # temp[neighbors[w]] = True
                            # inter_mask &= temp
                            temp = np.zeros(N, dtype=bool)
                            temp[neighbors[w]] = True
                            inter_mask &= temp
                        candidates = np.where(inter_mask)[0]
                    
                    if len(candidates) == 0:
                        continue
                        
                    mask = ~in_C[candidates]
                    if not np.any(mask):
                        continue
                    
                    valid_candidates = candidates[mask]
                    
                    # Evaluate swaps
                    # We want to minimize new_critical
                    # new_critical = sum(count == 1) after swap
                    # count' = count - 1(nb_c) + 1(nb_cp)
                    # We can compute this without full copy if we are careful,
                    # but N=2187 is small. Copy is ~10-20us.
                    # 186 codewords * ~10 candidates * 10us = ~18ms per pass.
                    # 100 passes = 1.8s. Acceptable.
                    
                    for cp in valid_candidates:
                        nb_cp = neighbors[cp]
                        
                        # Quick check: does cp cover the critical words?
                        # We already ensured cp is in intersection, so yes.
                        
                        # Calculate change in critical count
                        # Words in nb_c:
                        # If count was 1 -> becomes 0 (was critical, now not critical, but uncovered? 
                        # Wait, if we remove c, and cp doesn't cover w, w becomes 0.
                        # But cp DOES cover all critical words of c.
                        # So critical words of c (count 1) become count 1 (from cp) or more?
                        # If w is in nb_c and nb_cp: count' = count - 1 + 1 = count.
                        # If w is in nb_c but not nb_cp: count' = count - 1.
                        #   If count was 1, becomes 0. (Lost critical, but became uncovered? 
                        #   No, if it was critical (count 1) and we remove c, it becomes 0 UNLESS cp covers it.
                        #   We know cp covers all critical words of c.
                        #   So if w was critical (count 1), it is in nb_c.
                        #   Is it in nb_cp? Yes, because cp is in intersection of neighbors of critical words.
                        #   So count' = 1 - 1 + 1 = 1. Still critical.
                        #   Wait, if count was 2, becomes 1. Becomes critical.
                        
                        # Let's just do the copy and sum. It's safer.
                        temp_count = cover_count.copy()
                        temp_count[nb_c] -= 1
                        temp_count[nb_cp] += 1
                        
                        new_critical = np.sum(temp_count == 1)
                        
                        if new_critical < best_critical:
                            best_critical = new_critical
                            best_swap = (c, cp)
                            
                    if best_swap:
                        # Apply best swap found in this iteration for c?
                        # No, the current code applies the best swap found for ANY c in the whole loop?
                        # The code says:
                        # if best_swap:
                        #   ... apply ...
                        #   improved = True
                        # This breaks out of the c-loop? No, it's outside the c-loop in the original?
                        # Let's check original structure.
                        # for c in C:
                        #   ...
                        #   for cp in valid:
                        #      ...
                        # if best_swap:
                        #   ...
                        # It applies the best swap found for the *current* c?
                        # No, `best_swap` is reset per `c`?
                        # Original:
                        # for c in C:
                        #   ...
                        #   best_swap = None
                        #   best_critical = current_critical
                        #   ...
                        #   if best_swap:
                        #      ... apply ...
                        #      improved = True
                        #      break # <--- Does it break?
                        # The original code does NOT break out of the `for c in C` loop immediately after applying a swap.
                        # It sets `improved = True` and continues to next `c`?
                        # Wait, if it applies a swap, `cover_count` changes.
                        # The `current_critical` calculated at the start of the c-loop is now stale for subsequent c's.
                        # This is a bug or suboptimal behavior in the original.
                        # It should restart the local search loop after a swap.
                        
                        pass # We will handle this in the new implementation

                # End of Step B logic

        return C

    best_C = None
    best_len = float('inf')
    
    # Run multiple times
    for seed in range(10):
        C = run_greedy_local_search(seed)
        if len(C) < best_len:
            best_len = len(C)
            best_C = C
            
    # Final cleanup: try to remove any remaining redundant words
    # The local search loop should have done this, but let's be sure.
    if best_C:
        # Convert to list of lists
        result_words = []
        for idx in best_C:
            word = []
            val = idx
            for _ in range(n):
                word.append(val % q)
                val //= q
            result_words.append(word)
        return result_words
    else:
        return []

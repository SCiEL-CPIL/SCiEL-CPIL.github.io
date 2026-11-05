import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    N = q ** n
    
    if radius != 1:
        raise NotImplementedError("Radius > 1 not optimized")

    sphere_size = 1 + n * (q - 1)
    neighbors = np.empty((N, sphere_size), dtype=int)
    
    # Precompute neighbors
    for i in range(N):
        idx = 0
        neighbors[i, idx] = i
        idx += 1
        val = i
        for pos in range(n):
            orig_val = val % q
            val //= q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[i, idx] = i + (new_val - orig_val) * (q ** pos)
                idx += 1

    best_C = None
    best_len = N + 1
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    while time.time() - start_time < time_limit:
        covered = np.zeros(N, dtype=bool)
        cover_count = np.zeros(N, dtype=int)
        C = []
        
        # 1. Greedy Phase with Randomization
        while covered.sum() < N:
            gains = (~covered[neighbors]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                # Randomly pick one to explore different basins
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_idx = candidates[0]
                
            C.append(best_idx)
            cover_count[neighbors[best_idx]] += 1
            covered[neighbors[best_idx]] = True
        
        if len(C) >= best_len:
            continue
            
        # 2. Local Search Phase
        in_C = np.zeros(N, dtype=bool)
        for c in C:
            in_C[c] = True
            
        improved = True
        while improved:
            improved = False
            
            # Step A: Remove redundant codewords (vectorized)
            for i in range(len(C)):
                c = C[i]
                nb = neighbors[c]
                cover_count[nb] -= 1
                
                if np.any(cover_count[nb] == 0):
                    cover_count[nb] += 1
                else:
                    C.pop(i)
                    in_C[c] = False
                    improved = True
                    break
            
            if not improved and C:
                # Step B: Efficient Swap using vectorized set operations
                best_swap = None
                best_delta = 0
                
                for c in C:
                    nb_c = neighbors[c]
                    critical_mask = (cover_count[nb_c] == 1)
                    critical_words = nb_c[critical_mask]
                    
                    if len(critical_words) == 0:
                        continue
                    
                    # Find candidates that cover all critical words
                    if len(critical_words) == 1:
                        candidates = neighbors[critical_words[0]]
                    else:
                        # Use bitwise AND on neighbor masks
                        mask0 = np.zeros(N, dtype=bool)
                        mask0[neighbors[critical_words[0]]] = True
                        for w in critical_words[1:]:
                            mask_w = np.zeros(N, dtype=bool)
                            mask_w[neighbors[w]] = True
                            mask0 &= mask_w
                            if not mask0.any():
                                break
                        if not mask0.any():
                            continue
                        candidates = np.where(mask0)[0]
                    
                    mask = ~in_C[candidates]
                    if not np.any(mask):
                        continue
                    
                    valid_candidates = candidates[mask]
                    
                    # Vectorized delta calculation
                    for cp in valid_candidates:
                        nb_cp = neighbors[cp]
                        
                        # Compute symmetric difference efficiently
                        in_c = np.zeros(N, dtype=bool)
                        in_c[nb_c] = True
                        in_cp = np.zeros(N, dtype=bool)
                        in_cp[nb_cp] = True
                        
                        only_c_mask = in_c & ~in_cp
                        only_cp_mask = in_cp & ~in_c
                        
                        # Check validity: no word in only_c has count 1
                        if np.any(cover_count[only_c_mask] == 1):
                            continue
                        
                        # Calculate delta
                        delta = np.sum(cover_count[only_c_mask] == 2) - np.sum(cover_count[only_cp_mask] == 1)
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_swap = (c, cp)
                
                if best_swap:
                    c, cp = best_swap
                    nb_c = neighbors[c]
                    nb_cp = neighbors[cp]
                    
                    cover_count[nb_c] -= 1
                    cover_count[nb_cp] += 1
                    
                    C.remove(c)
                    C.append(cp)
                    in_C[c] = False
                    in_C[cp] = True
                    improved = True
                    continue

                # Step C: 2-for-1 Swap / 2-Removal
                K = len(C)
                for i in range(K):
                    if improved:
                        break
                    c1 = C[i]
                    nb1 = neighbors[c1]
                    set_nb1 = set(nb1)
                    
                    for j in range(i + 1, K):
                        c2 = C[j]
                        nb2 = neighbors[c2]
                        set_nb2 = set(nb2)
                        
                        critical_words = []
                        
                        # Check words in nb1
                        for w in nb1:
                            if w in set_nb2:
                                # Covered by both c1 and c2
                                if cover_count[w] == 2:
                                    critical_words.append(w)
                            else:
                                # Covered only by c1 (among c1,c2)
                                if cover_count[w] == 1:
                                    critical_words.append(w)
                                    
                        # Check words in nb2 that are not in nb1
                        for w in nb2:
                            if w not in set_nb1:
                                if cover_count[w] == 1:
                                    critical_words.append(w)
                            
                        if not critical_words:
                            # Can remove both c1 and c2
                            cover_count[nb1] -= 1
                            cover_count[nb2] -= 1
                            C.pop(j)
                            C.pop(i)
                            in_C[c1] = False
                            in_C[c2] = False
                            improved = True
                            break
                        else:
                            # Try to find c3 covering all critical_words
                            # c3 must be in intersection of neighbors of critical_words
                            # If critical_words is small, find intersection
                            if len(critical_words) > 0:
                                # Start with neighbors of first critical word
                                mask = np.zeros(N, dtype=bool)
                                mask[neighbors[critical_words[0]]] = True
                                for w in critical_words[1:]:
                                    mask_w = np.zeros(N, dtype=bool)
                                    mask_w[neighbors[w]] = True
                                    mask &= mask_w
                                    if not mask.any():
                                        break
                                
                                if mask.any():
                                    # Find any candidate c3
                                    candidates = np.where(mask)[0]
                                    # Prefer c3 not in C? Or any c3?
                                    # If c3 is in C, we are removing 2 and adding 1 existing.
                                    # If c3 is not in C, we are removing 2 and adding 1 new.
                                    # Both reduce size by 1.
                                    # We should pick one that doesn't cause issues?
                                    # Adding c3 increases cover_count for its neighbors.
                                    # Removing c1, c2 decreases.
                                    # We already checked validity for critical words.
                                    # Are there other words that become uncovered?
                                    # No, only words in nb1 U nb2 were at risk.
                                    # Words outside nb1 U nb2 are unaffected by removal.
                                    # Words in nb1 U nb2 are covered by c3 if in critical_words.
                                    # What about words in nb1 U nb2 that are NOT in critical_words?
                                    # If w in nb1 U nb2 and not critical, it means:
                                    # If w in nb1 only: cover_count[w] >= 2. After removal, count >= 1. Covered by others.
                                    # If w in nb2 only: cover_count[w] >= 2. After removal, count >= 1. Covered by others.
                                    # If w in both: cover_count[w] >= 3. After removal, count >= 1. Covered by others.
                                    # So validity is guaranteed.
                                    
                                    # Pick first candidate
                                    c3 = candidates[0]
                                    
                                    # Update state
                                    cover_count[nb1] -= 1
                                    cover_count[nb2] -= 1
                                    cover_count[neighbors[c3]] += 1
                                    
                                    C.pop(j)
                                    C.pop(i)
                                    C.append(c3)
                                    
                                    in_C[c1] = False
                                    in_C[c2] = False
                                    in_C[c3] = True
                                    
                                    improved = True
                                    break

        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()
            
    # Final optimization: try to remove more codewords
    if best_C:
        in_C = np.zeros(N, dtype=bool)
        for c in best_C:
            in_C[c] = True
        cover_count = np.zeros(N, dtype=int)
        for c in best_C:
            cover_count[neighbors[c]] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(best_C)):
                c = best_C[i]
                nb = neighbors[c]
                cover_count[nb] -= 1
                if np.any(cover_count[nb] == 0):
                    cover_count[nb] += 1
                else:
                    best_C.pop(i)
                    in_C[c] = False
                    improved = True
                    break
    
    result_words = []
    for idx in best_C:
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        result_words.append(word)
        
    return result_words

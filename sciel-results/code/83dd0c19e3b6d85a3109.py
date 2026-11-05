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

    # Precompute coverers (inverse of neighbors)
    coverers = [set() for _ in range(N)]
    for c in range(N):
        for w in neighbors[c]:
            coverers[w].add(c)
            
    # Convert to lists for faster iteration if needed, or keep sets for intersection
    # Sets are good for intersection.

    # Partition into cosets of a 2D subspace (indices 0, 1 vary)
    coset_id = np.zeros(N, dtype=int)
    words_in_coset = [[] for _ in range(q ** (n - 2))]
    
    for i in range(N):
        val = i
        coords = [0] * n
        for pos in range(n):
            coords[pos] = val % q
            val //= q
        
        key = 0
        for pos in range(2, n):
            key = key * q + coords[pos]
            
        coset_id[i] = key
        words_in_coset[key].append(i)

    best_C = None
    best_len = N + 1
    start_time = time.time()
    time_limit = 110
    
    while time.time() - start_time < time_limit:
        covered = np.zeros(N, dtype=bool)
        in_C = np.zeros(N, dtype=bool)
        C = []
        
        # 1. Partition-Based Constructive Phase
        cosets = list(range(len(words_in_coset)))
        random.shuffle(cosets)
        
        for cid in cosets:
            words_in_c = words_in_coset[cid]
            
            # Check if all covered
            if all(covered[w] for w in words_in_c):
                continue
                
            while True:
                # Find uncovered words in this coset
                uncovered_in_c = [w for w in words_in_c if not covered[w]]
                if not uncovered_in_c:
                    break
                
                # Find best codeword
                best_cand = -1
                max_local_gain = -1
                max_global_gain = -1
                
                # Candidate pool: All c not in C
                # We can iterate through all N
                for c in range(N):
                    if in_C[c]:
                        continue
                    
                    # Count local gain
                    local_gain = 0
                    for w in uncovered_in_c:
                        if c in coverers[w]:
                            local_gain += 1
                            
                    if local_gain <= max_local_gain:
                        continue
                        
                    # Calculate global gain (uncovered words globally covered by c)
                    global_gain = 0
                    for w in neighbors[c]:
                        if not covered[w]:
                            global_gain += 1
                            
                    # Prefer high local gain, then high global gain
                    if local_gain > max_local_gain or (local_gain == max_local_gain and global_gain > max_global_gain):
                        max_local_gain = local_gain
                        max_global_gain = global_gain
                        best_cand = c
                        
                if best_cand == -1:
                    break # Should not happen if coverers is correct
                    
                # Add best_cand
                C.append(best_cand)
                in_C[best_cand] = True
                for w in neighbors[best_cand]:
                    covered[w] = True
        
        # 2. Local Search Phase
        cover_count = np.zeros(N, dtype=int)
        for c in C:
            cover_count[neighbors[c]] += 1
            
        improved = True
        while improved:
            improved = False
            
            # Step A: Remove redundant
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
                # Step B: Swap
                best_swap = None
                # We want to find a swap that maintains coverage.
                # To improve, we can look for swaps that reduce the number of "critical" words
                # or just accept any valid swap and hope for better removals.
                
                for c in C:
                    nb_c = neighbors[c]
                    # Words that would be uncovered if c is removed, unless covered by c'
                    # Critical words: cover_count[w] == 1
                    critical_words = nb_c[cover_count[nb_c] == 1]
                    
                    if len(critical_words) == 0:
                        continue # c is redundant? Should have been removed in Step A.
                        
                    # Find c' that covers all critical_words
                    # c' must be in intersection of coverers[w] for w in critical_words
                    # And c' not in C
                    # And for all w in nb_c not in nb_c', cover_count[w] > 1 (so they stay covered)
                    
                    # Intersection of coverers
                    candidates = coverers[critical_words[0]].copy()
                    for w in critical_words[1:]:
                        candidates.intersection_update(coverers[w])
                        
                    # Remove existing C members
                    candidates -= set(C)
                    
                    if not candidates:
                        continue
                        
                    for c_prime in candidates:
                        nb_cp = neighbors[c_prime]
                        
                        # Check validity: 
                        # Words in nb_c not in nb_cp must have cover_count > 1
                        # i.e., cover_count[w] >= 2 for all w in nb_c \ nb_cp
                        # Note: cover_count currently includes c.
                        # If w in nb_c \ nb_cp, it is covered by c.
                        # If cover_count[w] == 1, it is ONLY covered by c.
                        # If we remove c, w is uncovered.
                        # So we need cover_count[w] > 1.
                        
                        # Efficient check:
                        # Create mask for nb_c
                        mask_c = np.zeros(N, dtype=bool)
                        mask_c[nb_c] = True
                        mask_cp = np.zeros(N, dtype=bool)
                        mask_cp[nb_cp] = True
                        
                        only_c = mask_c & ~mask_cp
                        
                        if np.any(cover_count[only_c] == 1):
                            continue
                            
                        # Valid swap found
                        # Perform swap
                        cover_count[nb_c] -= 1
                        cover_count[nb_cp] += 1
                        
                        C.remove(c)
                        C.append(c_prime)
                        in_C[c] = False
                        in_C[c_prime] = True
                        improved = True
                        break
                        
                    if improved:
                        break
        
        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()
            
    # Final cleanup
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

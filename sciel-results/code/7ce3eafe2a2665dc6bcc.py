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
            
            # Step A: Remove redundant codewords
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
                # Step B: Efficient Swap
                current_critical = np.sum(cover_count == 1)
                best_swap = None
                best_delta = 0  # We want negative delta (reduction in critical)
                
                for c in C:
                    nb_c = neighbors[c]
                    # Critical words covered ONLY by c
                    critical_mask = (cover_count[nb_c] == 1)
                    critical_words = nb_c[critical_mask]
                    
                    if len(critical_words) == 0:
                        # If no critical words, we can just remove? No, we check removal in Step A.
                        # If we reach here, it wasn't removable? 
                        # Actually, if no critical words, cover_count[nb_c] >= 2 for all.
                        # Then removing c would make them >= 1. So it IS removable.
                        # Why didn't Step A catch it? 
                        # Step A checks `np.any(cover_count[nb] == 0)` AFTER decrement.
                        # If cover_count[nb_c] were all >= 2, decrement makes them >= 1. So no zeros.
                        # So it should have been removed.
                        # Thus, if we are here, critical_words is not empty.
                        pass
                    
                    # Candidates must cover all critical_words
                    # To be efficient, we can iterate through all non-C words and check?
                    # Or intersect neighborhoods of critical words.
                    
                    if len(critical_words) == 1:
                        w = critical_words[0]
                        candidates = neighbors[w]
                    else:
                        inter = set(neighbors[critical_words[0]].tolist())
                        for w in critical_words[1:]:
                            inter.intersection_update(neighbors[w].tolist())
                            if not inter:
                                break
                        if not inter:
                            continue
                        candidates = np.array(list(inter))
                    
                    mask = ~in_C[candidates]
                    if not np.any(mask):
                        continue
                    
                    valid_candidates = candidates[mask]
                    
                    for cp in valid_candidates:
                        nb_cp = neighbors[cp]
                        
                        # Calculate Delta Critical
                        # Words in S_c \ S_cp: count drops by 1
                        # Words in S_cp \ S_c: count rises by 1
                        
                        # We need to know counts for these specific words.
                        # Validity: No word in S_c \ S_cp can have count 1 (would become 0).
                        # Delta: 
                        # +1 for each w in S_c \ S_cp with count 2 (becomes 1)
                        # -1 for each w in S_cp \ S_c with count 1 (becomes 2)
                        
                        # S_c \ S_cp
                        # We can compute this using boolean masks or sets.
                        # For N=2187, SphereSize=19. Sets are fast enough.
                        
                        set_c = set(nb_c.tolist())
                        set_cp = set(nb_cp.tolist())
                        
                        only_c = set_c - set_cp
                        only_cp = set_cp - set_c
                        
                        # Check validity
                        valid = True
                        delta = 0
                        for w in only_c:
                            if cover_count[w] == 1:
                                valid = False
                                break
                            elif cover_count[w] == 2:
                                delta += 1
                        if not valid:
                            continue
                            
                        for w in only_cp:
                            if cover_count[w] == 1:
                                delta -= 1
                            
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
        
        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()
            
    result_words = []
    for idx in best_C:
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        result_words.append(word)
        
    return result_words

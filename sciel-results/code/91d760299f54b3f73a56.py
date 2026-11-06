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
    
    # Helper: Greedy cover to fill holes
    def greedy_fill(covered, cover_count, C):
        while covered.sum() < N:
            gains = (~covered[neighbors]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_idx = candidates[0]
            C.append(best_idx)
            cover_count[neighbors[best_idx]] += 1
            covered[neighbors[best_idx]] = True
        return C

    # Helper: Local Search
    def local_search(C):
        in_C = np.zeros(N, dtype=bool)
        for c in C:
            in_C[c] = True
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
                best_delta = 0
                for c in C:
                    nb_c = neighbors[c]
                    critical_mask = (cover_count[nb_c] == 1)
                    critical_words = nb_c[critical_mask]
                    if len(critical_words) == 0:
                        continue
                    
                    if len(critical_words) == 1:
                        candidates = neighbors[critical_words[0]]
                    else:
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
                    
                    for cp in valid_candidates:
                        nb_cp = neighbors[cp]
                        in_c = np.zeros(N, dtype=bool)
                        in_c[nb_c] = True
                        in_cp = np.zeros(N, dtype=bool)
                        in_cp[nb_cp] = True
                        
                        only_c_mask = in_c & ~in_cp
                        only_cp_mask = in_cp & ~in_c
                        
                        if np.any(cover_count[only_c_mask] == 1):
                            continue
                        
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
        return C

    # Initial Greedy
    covered = np.zeros(N, dtype=bool)
    cover_count = np.zeros(N, dtype=int)
    C = []
    greedy_fill(covered, cover_count, C)
    
    C = local_search(C)
    
    if len(C) < best_len:
        best_len = len(C)
        best_C = C.copy()
        
    # Iterated Local Search
    while time.time() - start_time < time_limit:
        # Perturb best_C
        C = best_C.copy()
        # Remove k random codewords
        k = min(5, len(C))
        if k > 0:
            remove_indices = random.sample(range(len(C)), k)
            remove_indices.sort(reverse=True)
            for idx in remove_indices:
                c = C.pop(idx)
                # Note: We don't update covered/cover_count here immediately, 
                # greedy_fill will handle the "holes"
                
        # Recalculate covered and cover_count for the perturbed C
        covered = np.zeros(N, dtype=bool)
        cover_count = np.zeros(N, dtype=int)
        for c in C:
            nb = neighbors[c]
            covered[nb] = True
            cover_count[nb] += 1
            
        # Fill holes
        greedy_fill(covered, cover_count, C)
        
        # Local Search
        C = local_search(C)
        
        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()

    # Final optimization on best_C
    if best_C:
        C = best_C.copy()
        C = local_search(C)
        if len(C) < best_len:
            best_len = len(C)
            best_C = C

    result_words = []
    for idx in best_C:
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        result_words.append(word)
        
    return result_words

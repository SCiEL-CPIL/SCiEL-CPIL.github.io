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
    
    def greedy_construct(randomize=False):
        covered = np.zeros(N, dtype=bool)
        C = []
        while covered.sum() < N:
            gains = (~covered[neighbors]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            if randomize and len(candidates) > 1:
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_idx = candidates[0]
            C.append(best_idx)
            covered[neighbors[best_idx]] = True
        return C

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
                # Only check a random subset of C to speed up? No, check all but optimize inner loop.
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
                    
                    # Optimize: Calculate delta for all valid_candidates at once?
                    # delta = (new coverage) - (lost coverage)
                    # Lost: words in nb_c not in nb_cp with count 1
                    # Gained: words in nb_cp not in nb_c with count 1? No, we want to minimize size.
                    # Swap reduces size if we replace 1 with 1? No, size stays same.
                    # We want to swap if it allows future removals?
                    # The current code calculates delta based on counts.
                    
                    # Let's stick to the current logic but maybe break early if we find a good swap.
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
                            if best_delta < -10: # Good enough improvement
                                break
                    if best_delta < -10:
                        break
                
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

    # ILS Loop
    while time.time() - start_time < time_limit:
        # Greedy
        C = greedy_construct(randomize=True)
        
        # Local Search
        C = local_search(C)
        
        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()
            
        # Perturb if we have time
        if time.time() - start_time < time_limit - 5:
            # Randomly swap 2-3 codewords in a copy of best_C
            if best_C:
                C_perturbed = best_C.copy()
                for _ in range(3):
                    if len(C_perturbed) > 1:
                        i = random.randint(0, len(C_perturbed) - 1)
                        # Pick a random new codeword
                        new_c = random.randint(0, N - 1)
                        C_perturbed[i] = new_c
                C_perturbed = local_search(C_perturbed)
                if len(C_perturbed) < best_len:
                    best_len = len(C_perturbed)
                    best_C = C_perturbed.copy()

    # Final cleanup
    if best_C:
        C = local_search(best_C)
        if len(C) < best_len:
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

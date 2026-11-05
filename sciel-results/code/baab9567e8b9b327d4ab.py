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
                
    # Precompute adjacency matrix for fast vectorized swaps
    # Adj[i, j] is True if j is a neighbor of i (including i)
    Adj = np.zeros((N, N), dtype=bool)
    Adj[np.arange(N)[:, None], neighbors] = True

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
        stall_count = 0
        max_stalls = 20 # Limit local search duration
        
        while improved and stall_count < max_stalls:
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
            
            if improved:
                stall_count = 0
                continue # Restart loop to check for more removals
            
            if not C:
                break
                
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
                # Candidates must be in intersection of neighbors of all critical words
                # AND not in C
                mask0 = np.ones(N, dtype=bool)
                for w in critical_words:
                    mask0 &= Adj[w]
                    if not mask0.any():
                        break
                if not mask0.any():
                    continue
                
                candidates = np.where(mask0 & ~in_C)[0]
                if len(candidates) == 0:
                    continue
                
                # Vectorized delta calculation
                # In_CP_batch: (M, N)
                In_CP_batch = Adj[candidates]
                In_C_vec = Adj[c]
                
                Only_C_batch = In_C_vec[None, :] & ~In_CP_batch
                Only_CP_batch = In_CP_batch & ~In_C_vec[None, :]
                
                # Check validity: no word in Only_C has count 1
                # cover_count[Only_C_batch] -> (M, N)
                # We need to check if ANY word in Only_C for a given candidate has count 1
                # invalid_mask = np.any(cover_count[Only_C_batch] == 1, axis=1)
                
                # Optimize:
                # If cover_count[w] == 1, and w is in Only_C, it's invalid.
                # So we can precompute a mask of words with count 1?
                # But cover_count changes.
                
                # Let's compute delta only for valid candidates
                # First, find valid candidates
                # A candidate is valid if no word in Only_C has count 1.
                # Only_C words are those in In_C_vec but not in In_CP_batch.
                # So for a candidate cp, invalid if there exists w such that Adj[c,w]=1, Adj[cp,w]=0, and cover_count[w]==1.
                
                # Let's just compute delta for all and check validity in the delta step?
                # No, validity is a hard constraint.
                
                # Vectorized validity check:
                # Count_1_mask = (cover_count == 1) # (N,)
                # For each cp, we need: sum(Count_1_mask & Only_C_batch[i]) == 0
                # Count_1_Only_C = Count_1_mask[None, :] & Only_C_batch
                # valid_mask = np.sum(Count_1_Only_C, axis=1) == 0
                
                Count_1_mask = (cover_count == 1)
                Count_1_Only_C = Count_1_mask[None, :] & Only_C_batch
                valid_mask = np.sum(Count_1_Only_C, axis=1) == 0
                
                if not np.any(valid_mask):
                    continue
                
                valid_candidates = candidates[valid_mask]
                valid_Only_C_batch = Only_C_batch[valid_mask]
                valid_Only_CP_batch = Only_CP_batch[valid_mask]
                
                Delta_batch = np.sum(cover_count[valid_Only_C_batch] == 2, axis=1) - np.sum(cover_count[valid_Only_CP_batch] == 1, axis=1)
                
                min_delta_idx = np.argmin(Delta_batch)
                min_delta = Delta_batch[min_delta_idx]
                
                if min_delta < best_delta:
                    best_delta = min_delta
                    best_swap = (c, valid_candidates[min_delta_idx])
            
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
                stall_count = 0
            else:
                stall_count += 1
        
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

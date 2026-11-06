import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Precompute APs and Incidence Matrix
    # APs are defined by start s, step d.
    # We include all APs of length >= 2.
    # Note: Length 1 APs have sum +/- 1, so max is at least 1.
    
    ap_terms = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            # AP: s, s+d, s+2d, ... < N
            terms = list(range(s, N, d))
            if len(terms) > 1:
                ap_terms.append(terms)
                
    num_aps = len(ap_terms)
    
    # Build M (num_aps x N)
    M = np.zeros((num_aps, N), dtype=np.float32)
    # Also store the list of AP indices for each position for fast updates
    pos_to_aps = [[] for _ in range(N)]
    
    for j, terms in enumerate(ap_terms):
        for i in terms:
            M[j, i] = 1.0
            pos_to_aps[i].append(j)
            
    # Convert pos_to_aps to numpy arrays for indexing
    pos_to_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]
    
    # Compute C = M.T @ M
    # C[i, k] = number of APs containing both i and k
    # C[i, i] = number of APs containing i (K_i)
    C = M.T @ M
    K = np.diag(C).astype(np.float32)
    
    # Helper to get hard max
    def get_hard_max(sums):
        return int(np.max(np.abs(sums)))
    
    # Soft-max surrogate: smooth approximation to max|sums|
    # Used as auxiliary objective to guide search toward reducing the hard max
    def soft_max_obj(sums, beta=3.0):
        abs_sums = np.abs(sums)
        m = np.max(abs_sums)
        if m == 0:
            return 0.0
        # log-sum-exp trick for stability
        return (m + np.log(np.sum(np.exp(abs_sums - m)))) / beta

    best_signs = None
    best_hard_max = N # Worst case
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        
        # Initialize with random signs
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.float32)
        
        # Initial sums
        sums = M @ signs
        
        # Initial G = C @ signs
        G = C @ signs
        
        current_hard_max = get_hard_max(sums)
        
        # SA Parameters
        T = 10.0
        T_min = 0.1
        
        steps = 0
        best_local_hard_max = current_hard_max
        best_local_signs = signs.copy()
        
        while time.time() < deadline - 2:
            # Pick random index to flip
            k = rng.randint(0, N - 1)
            
            # Delta for signs
            delta = -2.0 * signs[k]
            
            # Change in J = sum S_P^2
            delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]
            
            # Combined objective: primarily sum-of-squares, with soft-max penalty
            # The soft-max term encourages reducing the worst AP sums
            idxs = pos_to_aps_arr[k]
            
            # Estimate change in soft-max by looking at the affected APs
            # For efficiency, only compute full soft-max change every 10 steps
            if steps % 10 == 0:
                old_soft = soft_max_obj(sums)
                # Compute new sums for affected APs only (approximation)
                new_sums_approx = sums.copy()
                new_sums_approx[idxs] += delta
                new_soft = soft_max_obj(new_sums_approx)
                delta_soft = new_soft - old_soft
                # Scale soft-max change to be comparable to delta_J
                delta_combined = delta_J + 5.0 * delta_soft
            else:
                delta_combined = delta_J
            
            # Metropolis Criterion
            accept = False
            if delta_combined < 0:
                accept = True
            else:
                if delta_combined / T > 100:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_combined / T)
                    
            if accept:
                # Apply move
                signs[k] = -signs[k]
                
                # Update sums for all APs containing k
                idxs = pos_to_aps_arr[k]
                sums[idxs] += delta
                
                # Update G
                G += delta * C[:, k]
                
                steps += 1
                
                # Check hard max periodically
                if steps % 30 == 0:
                    current_hard_max = get_hard_max(sums)
                    if current_hard_max < best_local_hard_max:
                        best_local_hard_max = current_hard_max
                        best_local_signs = signs.copy()
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                        
            # Annealing
            if steps % 500 == 0:
                T *= 0.97
                if T < T_min:
                    T = T_min
                    
            # Early stop if we hit 1
            if best_hard_max <= 1:
                break
        
        # Polish phase: exact local search to reduce hard max
        if time.time() < deadline - 1:
            polish_signs = best_local_signs.copy()
            polish_sums = M @ polish_signs
            polish_hard_max = get_hard_max(polish_sums)
            
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for k in range(N):
                    delta = -2.0 * polish_signs[k]
                    idxs = pos_to_aps_arr[k]
                    # Try flip
                    old_sums = polish_sums.copy()
                    polish_sums[idxs] += delta
                    new_hard_max = get_hard_max(polish_sums)
                    if new_hard_max < polish_hard_max:
                        polish_signs[k] = -polish_signs[k]
                        polish_hard_max = new_hard_max
                        improved = True
                        if polish_hard_max < best_hard_max:
                            best_hard_max = polish_hard_max
                            best_signs = polish_signs.copy()
                            if best_hard_max <= 1:
                                break
                    else:
                        # Revert
                        polish_sums = old_sums
                if best_hard_max <= 1:
                    break
    
    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]

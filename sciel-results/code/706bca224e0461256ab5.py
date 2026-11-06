import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Precompute APs and Incidence Matrix
    ap_terms = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            terms = list(range(s, N, d))
            if len(terms) > 1:
                ap_terms.append(terms)
                
    num_aps = len(ap_terms)
    
    # Build M (num_aps x N)
    M = np.zeros((num_aps, N), dtype=np.float32)
    pos_to_aps = [[] for _ in range(N)]
    
    for j, terms in enumerate(ap_terms):
        for i in terms:
            M[j, i] = 1.0
            pos_to_aps[i].append(j)
            
    pos_to_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]
    
    # Compute C = M.T @ M
    C = M.T @ M
    K = np.diag(C).astype(np.float32)
    
    def get_hard_max(sums):
        return int(np.max(np.abs(sums)))
    
    def get_soft_max(sums, beta):
        """Log-sum-exp based soft max of |sums|"""
        abs_sums = np.abs(sums)
        max_val = np.max(abs_sums)
        if max_val == 0:
            return 0.0
        # Use log-sum-exp trick for stability
        exp_sums = np.exp(beta * (abs_sums - max_val))
        soft_max = max_val + np.log(np.sum(exp_sums)) / beta
        return soft_max
    
    def get_weighted_soft_max(sums, beta, weight):
        """Weighted soft max that emphasizes larger values"""
        abs_sums = np.abs(sums)
        max_val = np.max(abs_sums)
        if max_val == 0:
            return 0.0
        exp_sums = np.exp(beta * (abs_sums - max_val))
        soft_max = max_val + np.log(np.sum(exp_sums)) / beta
        return weight * soft_max
    
    best_signs = None
    best_hard_max = N
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        rng = random.Random(seed)
        
        # Initialize with random signs
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.float32)
        
        # Initial sums
        sums = M @ signs
        
        # Initial G = C @ signs
        G = C @ signs
        
        current_hard_max = get_hard_max(sums)
        
        # SA Parameters - tune for better max reduction
        T = 12.0
        beta = 3.0  # Higher temperature for soft-max to be more sensitive
        weight_soft = 0.5  # Weight for soft-max objective
        
        steps = 0
        no_improve_steps = 0
        best_hard_this_run = current_hard_max
        
        while time.time() < deadline - 5:
            # Pick random index to flip
            k = rng.randint(0, N - 1)
            
            delta = -2.0 * signs[k]
            
            # Compute delta_J using sum-of-squares
            delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]
            
            # Compute delta for soft-max objective
            idxs = pos_to_aps_arr[k]
            affected_sums = sums[idxs]
            
            old_soft = get_soft_max(affected_sums, beta)
            new_sums_affected = affected_sums + delta
            new_soft = get_soft_max(new_sums_affected, beta)
            delta_soft = new_soft - old_soft
            
            # Combined objective: primarily sum-of-squares, with soft-max as tiebreaker
            delta_obj = delta_J + weight_soft * delta_soft
            
            accept = False
            if delta_obj < 0:
                accept = True
            else:
                if delta_obj / T > 100:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_obj / T)
                    
            if accept:
                signs[k] = -signs[k]
                sums[idxs] += delta
                G += delta * C[:, k]
                
                steps += 1
                no_improve_steps += 1
                
                # Check hard max more frequently
                if steps % 10 == 0:
                    current_hard_max = get_hard_max(sums)
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                        no_improve_steps = 0
                    elif current_hard_max < best_hard_this_run:
                        best_hard_this_run = current_hard_max
                        no_improve_steps = 0
                        
            # Annealing - faster cooling
            if steps % 300 == 0:
                T *= 0.95
                if T < 0.05:
                    T = 0.05
                beta *= 1.08
                if beta > 30.0:
                    beta = 30.0
                weight_soft = min(1.0, weight_soft * 1.02)  # Increase weight over time
                
            if best_hard_max <= 1:
                break
                
            # If stuck, try a more aggressive phase
            if no_improve_steps > 5000:
                # Restart with current best
                if best_signs is not None:
                    signs = best_signs.copy()
                    sums = M @ signs
                    G = C @ signs
                    current_hard_max = get_hard_max(sums)
                    best_hard_this_run = current_hard_max
                    no_improve_steps = 0
                    T = max(2.0, T * 0.5)
                else:
                    break
                    
    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    # Final polish: try to reduce max by focusing on the worst APs
    if time.time() < deadline - 2:
        signs = best_signs.copy()
        sums = M @ signs
        G = C @ signs
        current_hard_max = get_hard_max(sums)
        
        # Find the APs with the largest absolute sums
        abs_sums = np.abs(sums)
        worst_aps = np.where(abs_sums >= current_hard_max - 1)[0]
        
        polish_steps = 0
        polish_T = 1.0
        
        while time.time() < deadline - 1 and polish_steps < 50000:
            # Focus on positions that affect the worst APs
            candidate_positions = set()
            for ap_idx in worst_aps[:20]:  # Top 20 worst APs
                candidate_positions.update(ap_terms[ap_idx])
            
            if not candidate_positions:
                break
                
            k = rng.choice(list(candidate_positions))
            
            delta = -2.0 * signs[k]
            
            idxs = pos_to_aps_arr[k]
            affected_sums = sums[idxs]
            
            old_max = np.max(np.abs(affected_sums))
            new_sums_affected = affected_sums + delta
            new_max = np.max(np.abs(new_sums_affected))
            
            # Only accept if it reduces the max or keeps it same but reduces sum of squares
            if new_max < old_max:
                accept = True
            elif new_max == old_max:
                delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]
                accept = delta_J < 0 or rng.random() < math.exp(-delta_J / polish_T)
            else:
                accept = False
            
            if accept:
                signs[k] = -signs[k]
                sums[idxs] += delta
                G += delta * C[:, k]
                polish_steps += 1
                
                if polish_steps % 50 == 0:
                    current_hard_max = get_hard_max(sums)
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                    
                    # Update worst APs
                    abs_sums = np.abs(sums)
                    worst_aps = np.where(abs_sums >= current_hard_max - 1)[0]
                    
                    if len(worst_aps) == 0:
                        break
                        
            # Cool down
            if polish_steps % 1000 == 0:
                polish_T *= 0.9
                if polish_T < 0.1:
                    polish_T = 0.1
                    
    return [int(s) for s in best_signs]

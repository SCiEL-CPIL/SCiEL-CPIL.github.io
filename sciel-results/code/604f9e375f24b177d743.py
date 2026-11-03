"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute exact QAP cost."""
    n = len(p)
    D = b[np.ix_(p, p)]
    return int(np.sum(a * D))

def _local_search(a, b, p, deadline):
    """
    Fast local search with 1-0 interchanges and 2-0 swaps.
    deadline: absolute time (time.time()) to stop by.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = _compute_cost(a, b, p)
    
    # Precompute for fast delta calculations
    # For swapping positions i and j (2-0 interchange):
    # Delta = sum_{k != i,j} (a[i,k] + a[k,i]) * (D[j,k] - D[i,k]) 
    #         + (a[i,j] + a[j,i]) * (D[j,i] - D[i,j])
    #         + (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
    # Using A_sym = A + A.T:
    # Delta = sum_{k} A_sym[i,k] * D[j,k] - sum_{k} A_sym[i,k] * D[i,k]
    #         + sum_{k} A_sym[j,k] * D[i,k] - sum_{k} A_sym[j,k] * D[j,k]
    #         + correction for k=i and k=j terms
    A_sym = a + a.T
    
    while time.time() < deadline:
        improved = False
        
        # 2-0 swap local search
        for i in range(n - 1):
            if time.time() > deadline:
                return p, current_cost
            
            # For all j > i, compute delta for swapping p[i] and p[j]
            js = np.arange(i + 1, n)
            
            # Vectorized: for each j in js, delta = 
            #   sum_k A_sym[i,k]*(D[j,k]-D[i,k]) + sum_k A_sym[j,k]*(D[i,k]-D[j,k])
            # = sum_k (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
            
            # But this includes k=i and k=j which need correction.
            # Let's use the standard formula:
            # delta(i,j) = (A_sym[i] - A_sym[j]) @ (D[j] - D[i]) 
            #              + (A[i,i] - A[j,j]) * (D[j,j] - D[i,i])
            #              + (A[i,j] - A[j,i]) * (D[j,i] - D[i,j])
            
            # Vectorized version:
            # For each j: (A_sym[i] - A_sym[j]) @ (D[j] - D[i])
            # = A_sym[i] @ D[j] - A_sym[i] @ D[i] - A_sym[j] @ D[i] + A_sym[j] @ D[j]
            
            # Precompute A_sym[i] @ D[i] (scalar)
            AiD_i = A_sym[i] @ D[i]
            
            # For each j: A_sym[i] @ D[j] - A_sym[i] @ D[i] - A_sym[j] @ D[i] + A_sym[j] @ D[j]
            # = (A_sym[i] @ D[js].T) - AiD_i - (D[i] @ A_sym[js].T) + np.sum(D[js] * A_sym[js], axis=1)
            
            term1 = D[js] @ A_sym[i]  # shape: (len(js),)
            term2 = A_sym[js] @ D[i]  # shape: (len(js),)
            term3 = np.sum(D[js] * A_sym[js], axis=1)  # shape: (len(js),)
            
            base_delta = term1 - AiD_i - term2 + term3
            
            # Correction for diagonal and cross terms
            # The base formula double-counts or miscounts some terms.
            # Correct delta for swap(i,j):
            # = sum_{k!=i,j} (A_sym[i,k]-A_sym[j,k])*(D[j,k]-D[i,k])
            #   + (A_sym[i,i]-A_sym[j,i])*(D[j,i]-D[i,i])  -- but this is part of the sum
            #   
            # Actually, let's use the verified formula:
            # delta = (A_sym[i] - A_sym[j]) @ (D[j] - D[i])
            #         + (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            #         + (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            
            # (A_sym[i] - A_sym[j]) @ (D[j] - D[i]) for each j:
            # = A_sym[i] @ D[j] - A_sym[i] @ D[i] - A_sym[j] @ D[i] + A_sym[j] @ D[j]
            # This is what we have as base_delta.
            
            # But we need to add:
            # (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            diag_corr = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
            
            # (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            cross_corr = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = base_delta + diag_corr + cross_corr
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    deadline = start_time + time_budget
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Multiple random restarts with local search
    # Since row sums are weak predictors, use diverse random starts
    num_initials = 8
    time_per_initial = (time_budget * 0.25) / num_initials
    
    for idx in range(num_initials):
        if time.time() > start_time + time_budget * 0.25:
            break
        
        p_init = np.random.permutation(n)
        dl = min(start_time + time_per_initial * (idx + 1), deadline - 5)
        p_ls, cost_ls = _local_search(a, b, p_init, dl)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    if best_p is None:
        best_p = np.random.permutation(n)
        best_cost = _compute_cost(a, b, best_p)
    
    # Phase 2: Iterated Local Search
    consecutive_no_improve = 0
    base_perturb = 4
    
    while time.time() < deadline - 3:
        # Adaptive perturbation: increase when stuck
        if consecutive_no_improve < 5:
            perturb_k = base_perturb
        elif consecutive_no_improve < 15:
            perturb_k = base_perturb + 2
        elif consecutive_no_improve < 30:
            perturb_k = base_perturb + 5
        else:
            perturb_k = base_perturb + 10
        
        p_pert = _perturb(best_p, perturb_k)
        
        # Give LS a reasonable time chunk
        remaining = deadline - 3 - time.time()
        if remaining < 2:
            break
        
        # Use up to 40% of remaining time for this LS run, min 1s
        ls_time = min(remaining * 0.4, max(1.0, remaining * 0.25))
        ls_deadline = time.time() + ls_time
        
        p_ls, cost_ls = _local_search(a, b, p_pert, ls_deadline)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
    
    return best_p.tolist()
# EVOLVE-BLOCK-END

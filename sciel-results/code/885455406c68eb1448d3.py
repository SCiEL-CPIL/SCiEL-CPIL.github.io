"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    # Safety margin: 120s limit, aim to finish by 118s
    deadline = time.time() + 118.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        # S1 = sum_k (a[i, k] - a[j, k]) * (b[v, p[k]] - b[u, p[k]])
        # S2 = sum_k (a[k, i] - a[k, j]) * (b[p[k], v] - b[p[k], u])
        
        diff_a_rows = a[i, :] - a[j, :]
        diff_b_rows = b[v, p] - b[u, p]
        s1 = np.dot(diff_a_rows, diff_b_rows)
        
        diff_a_cols = a[:, i] - a[:, j]
        diff_b_cols = b[p, v] - b[p, u]
        s2 = np.dot(diff_a_cols, diff_b_cols)
        
        # Correction for k=i and k=j
        ai_i = a[i, i]
        aj_i = a[j, i]
        ai_j = a[i, j]
        aj_j = a[j, j]
        
        bu_u = b[u, u]
        bu_v = b[u, v]
        bv_u = b[v, u]
        bv_v = b[v, v]
        
        corr_1 = (ai_i - aj_i) * (bv_u - bu_u)
        corr_2 = (ai_i - aj_i) * (bu_v - bu_u)
        corr_3 = (ai_j - aj_j) * (bv_v - bu_v)
        corr_4 = (ai_j - aj_j) * (bu_v - bv_v)
        
        # Correct interaction term for i, j
        corr_interaction = (ai_j - aj_i) * (bv_u - bu_v)
        
        return s1 + s2 - (corr_1 + corr_2 + corr_3 + corr_4) + corr_interaction

    def local_search(p, current_cost, deadline):
        improved = True
        max_iter = 500 
        iter_count = 0
        
        while improved and time.time() < deadline - 2.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Scan all pairs to find the steepest descent move
            for i in range(n - 1):
                if time.time() >= deadline - 2.0:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() >= deadline - 2.0:
                        return p, current_cost
                    d = swap_delta(i, j, p)
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                iter_count += 1
            else:
                break
                
        return p, current_cost

    # --- Initialization Strategies ---
    inits = []

    # 1. Rank-based initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    p_rank_rev = np.empty(n, dtype=np.int64)
    p_rank_rev[heavy] = central[::-1]
    
    inits.extend([p_rank, p_rank_rev])

    # 2. Hungarian Algorithm Initialization (Linear Relaxation)
    # Construct cost matrix C where C[i, j] approximates the contribution of placing i at j.
    # C[i, j] = sum_k a[i, k] * b[j, k] is a good linear proxy if we assume k is distributed uniformly or similar.
    # More accurately, the linear relaxation often uses C[i, j] = sum_k a[i, k] * (b[j, k] - mean_b) or similar.
    # Let's use C[i, j] = - sum_k a[i, k] * b[j, k] to maximize affinity.
    # Note: This is O(n^3) to compute, which is 150^3 = 3.3M ops, very fast.
    
    try:
        if time.time() < deadline - 5.0:
            # Compute the linearized cost matrix
            # We want to minimize sum a[i,k] b[p[i], p[k]].
            # A common linearization is to fix the "average" location for each facility index k.
            # Let's try C[i, j] = sum_k a[i, k] * b[j, k]
            # We want to minimize this, so we pass C to linear_sum_assignment.
            C = a @ b
            
            # Solve the assignment problem
            # row_ind, col_ind = linear_sum_assignment(C)
            # p = np.empty(n, dtype=np.int64)
            # p[row_ind] = col_ind
            
            # However, linear_sum_assignment minimizes the cost.
            # So we pass C directly.
            row_ind, col_ind = linear_sum_assignment(C)
            p_hung = np.empty(n, dtype=np.int64)
            p_hung[row_ind] = col_ind
            inits.append(p_hung)
            
            # Also try the negative (maximization) just in case sign conventions vary in interpretation, 
            # though for minimization of sum a*b, minimizing sum a[i,k]b[j,k] is the direct linear relaxation.
    except Exception:
        pass

    # 3. Spectral initialization
    try:
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        
        a_scale = np.sqrt(np.mean(a_centered ** 2)) + 1e-10
        b_scale = np.sqrt(np.mean(b_centered ** 2)) + 1e-10
        
        a_norm = a_centered / a_scale
        b_norm = b_centered / b_scale
        
        # Symmetrized combined matrix
        C = 0.5 * (a_norm + a_norm.T) + 0.5 * (b_norm + b_norm.T)
        C = 0.5 * (C + C.T)
        
        # Get top k eigenvectors
        k = 5
        if time.time() < deadline - 30.0:
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Take the last k eigenvectors (largest eigenvalues)
            eigvecs = eigenvectors[:, n-k:]
            
            for idx in range(k):
                eigvec = eigvecs[:, idx]
                facility_order = np.argsort(eigvec, kind="stable")
                location_order = np.argsort(-eigvec, kind="stable")
                
                p = np.empty(n, dtype=np.int64)
                p[facility_order] = location_order
                inits.append(p)
    except Exception:
        pass

    # 4. Random initializations
    rng = np.random.RandomState(42)
    for _ in range(3):
        inits.append(rng.permutation(n))

    # --- Main Search Loop ---
    
    best_cost = np.inf
    best_p = None
    
    # Run local search from each initialization
    for p_init in inits:
        if time.time() > deadline - 15.0:
            break
        
        p_work = p_init.copy()
        cost = compute_cost(p_work)
        
        p_work, cost = local_search(p_work, cost, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_work.copy()
    
    if best_p is None:
        best_p = inits[0].copy()
        best_cost = compute_cost(best_p)

    # --- Iterated Local Search (ILS) with Adaptive Perturbation ---
    
    no_improve_count = 0
    base_perturb = 3
    max_perturb = 15
    
    while time.time() < deadline - 10.0:
        p = best_p.copy()
        
        # Adaptive perturbation size
        perturb_size = base_perturb + min(no_improve_count, 10)
        perturb_size = min(perturb_size, max_perturb)
        
        # Occasionally do a large jump (20% chance)
        if rng.rand() < 0.2:
            perturb_size = max_perturb + rng.randint(0, 10)
        
        num_swaps = max(2, perturb_size)
        for _ in range(num_swaps):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        current_cost = compute_cost(p)
        p, current_cost = local_search(p, current_cost, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Reset counter if it gets too high to allow exploration
        if no_improve_count > 20:
            no_improve_count = 10

    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()
    
    return list(range(n))

# EVOLVE-BLOCK-END

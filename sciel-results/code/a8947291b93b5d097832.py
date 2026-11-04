"""Improved QAP solver with Hungarian init, fast 2-opt LS, and ILS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _local_search(a, b, p, time_limit, start_time, max_passes=5):
    """
    Fast local search with 2-opt swaps using vectorized delta computation.
    Returns (p, cost).
    """
    n = len(p)
    p = np.array(p, dtype=np.int64).copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = float(np.sum(a * D))
    
    # Precompute a + a.T for delta computation (symmetric part)
    # Actually for QAP with potentially asymmetric a, we need to be careful.
    # The standard delta for swapping i and j:
    # delta = sum_k [(a[i,k]+a[j,k])*(D[j,k]-D[i,k])] 
    #         - (a[i,i]+a[i,j])*(D[j,i]-D[i,i]) 
    #         - (a[j,i]+a[j,j])*(D[j,j]-D[i,j])
    #         + (a[i,i]-a[j,j])*(D[j,j]-D[i,i])
    #         + (a[i,j]-a[j,i])*(D[j,i]-D[i,j])
    
    pass_count = 0
    improved = True
    while improved and pass_count < max_passes and (time.time() - start_time) < time_limit:
        improved = False
        pass_count += 1
        
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return p, current_cost
            
            js = np.arange(i+1, n)
            if len(js) == 0:
                continue
            
            a_i = a[i]
            a_js = a[js]  # shape (len(js), n)
            
            D_i = D[i]
            D_js = D[js]  # shape (len(js), n)
            
            # Main term: sum_k (a[i,k] + a[j,k]) * (D[j,k] - D[i,k])
            diff_D = D_js - D_i[None, :]  # (len(js), n)
            sum_a = a_i[None, :] + a_js  # (len(js), n)
            main_term = np.sum(sum_a * diff_D, axis=1)  # (len(js),)
            
            # Correction terms
            a_ii = a[i, i]
            a_ij = a[i, js]  # (len(js),)
            Dj_i = D[js, i]  # (len(js),)
            Di_i = D[i, i]
            corr1 = -(a_ii + a_ij) * (Dj_i - Di_i)
            
            a_ji = a[js, i]  # (len(js),)
            a_jj = np.diag(a[js])  # (len(js),)
            Dj_j = np.diag(D_js)  # (len(js),)
            Di_j = D_i[js]  # (len(js),)
            corr2 = -(a_ji + a_jj) * (Dj_j - Di_j)
            
            corr3 = (a_ii - a_jj) * (Dj_j - Di_i)
            
            corr4 = (a_ij - a_ji) * (Dj_i - Di_j)
            
            deltas = main_term + corr1 + corr2 + corr3 + corr4
            
            min_idx = np.argmin(deltas)
            if deltas[min_idx] < -1e-9:
                j = js[min_idx]
                delta = deltas[min_idx]
                
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D: swap rows i and j, then swap cols i and j
                D[i, :], D[j, :] = D[j, :].copy(), D[i, :].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += delta
                improved = True
    
    return p, current_cost

def _hungarian_init(a, b, n, rng):
    """
    Generate an initial solution using Hungarian algorithm on a surrogate cost matrix.
    We use the idea that high-flow facilities should be placed at low-distance locations.
    """
    # Compute row sums of a (total flow involving each facility)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Create a cost matrix for assignment
    # We want to assign facilities with high flow to locations with low distance sum
    # This is a linear assignment problem
    # Cost[i, j] = a_row_sums[i] * b_row_sums[j]
    # But we can also add some noise for diversity
    
    cost_matrix = np.outer(a_row_sums, b_row_sums)
    
    # Add small random perturbation for diversity
    cost_matrix += rng.uniform(0, 1, size=(n, n)) * 0.01 * np.max(np.abs(cost_matrix))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # row_ind[i] = facility i, col_ind[i] = location assigned to facility i
    # p[facility] = location
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _diversified_random_init(a, b, n, rng):
    """Generate a random initial solution with some structure."""
    # Sort facilities by row sum
    a_row_sums = a.sum(axis=1)
    a_order = np.argsort(a_row_sums)[::-1]
    
    # Shuffle locations with some bias
    b_row_sums = b.sum(axis=1)
    b_order = np.argsort(b_row_sums)
    
    # Partially shuffle
    p = np.empty(n, dtype=np.int64)
    p[a_order] = b_order
    p = rng.permutation(p)
    
    return p

def _perturb(p, k, rng):
    """Apply k random swaps to p."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 140.0  # leave 10s buffer for safety
    
    rng = np.random.default_rng(42)
    
    # Generate initial solutions
    initial_solutions = []
    
    # Hungarian-based init
    initial_solutions.append(_hungarian_init(a, b, n, rng))
    
    # Diversified random inits
    for _ in range(3):
        initial_solutions.append(_diversified_random_init(a, b, n, rng))
    
    # Pure random
    initial_solutions.append(rng.permutation(n))
    
    best_p = None
    best_cost = np.inf
    
    # Local search on each initial solution
    for idx, p_init in enumerate(initial_solutions):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_init, remaining - 3, start_time, max_passes=3)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Iterated Local Search
    # Use adaptive perturbation size
    no_improve_count = 0
    max_no_improve = 50
    
    while (time.time() - start_time) < time_budget - 10:
        # Adaptive perturbation: start small, increase if stuck
        if no_improve_count < 10:
            k_perturb = 2
        elif no_improve_count < 30:
            k_perturb = 4
        elif no_improve_count < 50:
            k_perturb = 6
        else:
            k_perturb = 8
            no_improve_count = 0  # reset after large perturbation
        
        p_pert = _perturb(best_p, k_perturb, rng)
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_pert, remaining - 3, start_time, max_passes=3)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_p.tolist()
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    # p is array of indices 0..n-1
    # cost = sum a[i,j] * b[p[i], p[j]]
    return np.sum(a * b[np.ix_(p, p)])

def _fast_2opt(a, b, p, time_budget=10.0):
    """
    Fast 2-opt local search.
    p: numpy array of permutation
    Returns updated p and cost.
    """
    n = len(p)
    p = p.copy().astype(np.int64)
    best_cost = _qap_cost(a, b, p)
    start_time = time.time()
    
    # Precompute row/col sums of A to speed up delta? 
    # The delta formula for swapping i and j (locations in permutation indices):
    # Let pi = p[i], pj = p[j]
    # Delta = sum_{k != i,j} [ a[i,k] (b[pj, p[k]] - b[pi, p[k]]) 
    #                        + a[k,i] (b[p[k], pj] - b[p[k], pi])
    #                        + a[j,k] (b[pi, p[k]] - b[pj, p[k]])
    #                        + a[k,j] (b[p[k], pi] - b[p[k], pj]) ]
    #       + a[i,i] (b[pj, pj] - b[pi, pi])
    #       + a[j,j] (b[pi, pi] - b[pj, pj])
    #       + a[i,j] (b[pj, pi] - b[pi, pj])
    #       + a[j,i] (b[pi, pj] - b[pj, pi])
    
    # We can simplify if A and B are symmetric. QAPLIB instances usually are.
    # Let's assume symmetry for speed but keep the general formula structure 
    # by using the full matrix.
    
    # To make this fast, we iterate i, and for each i, we find the best j > i.
    # This is Best-Improvement 2-opt.
    
    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_i, best_j = -1, -1
        
        # Check time periodically
        for i in range(n - 1):
            if time.time() - start_time > time_budget:
                break
            
            pi = p[i]
            # We want to find j > i that minimizes delta
            # We can vectorize the calculation of delta for all j > i
            # But the mask changes for each i.
            
            # Let's do a simpler loop for j but vectorize over k
            # This is O(n^3) but with low constant factors in NumPy
            
            for j in range(i + 1, n):
                if time.time() - start_time > time_budget:
                    break
                
                pj = p[j]
                
                # Compute delta
                # Self terms
                delta = a[i, i] * (b[pj, pj] - b[pi, pi])
                delta += a[j, j] * (b[pi, pi] - b[pj, pj])
                # Cross terms
                delta += a[i, j] * (b[pj, pi] - b[pi, pj])
                delta += a[j, i] * (b[pi, pj] - b[pj, pi])
                
                # Bulk terms: sum over k != i, j
                # Create mask
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                pk = p[mask]
                
                # Vectorized sums
                # Term 1: sum a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
                delta += np.sum(a[i, mask] * (b[pj, pk] - b[pi, pk]))
                # Term 2: sum a[k,i] * (b[p[k], pj] - b[p[k], pi])
                delta += np.sum(a[mask, i] * (b[pk, pj] - b[pk, pi]))
                # Term 3: sum a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
                delta += np.sum(a[j, mask] * (b[pi, pk] - b[pj, pk]))
                # Term 4: sum a[k,j] * (b[p[k], pi] - b[p[k], pj])
                delta += np.sum(a[mask, j] * (b[pk, pi] - b[pk, pj]))
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < -1e-9:
            # Swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            best_cost += int(round(best_delta))
            improved = True
    
    return p, best_cost

def _frank_wolfe(a, b, p_init, max_iters=30, time_budget=10.0):
    """
    Frank-Wolfe algorithm on Birkhoff polytope.
    Starts from permutation p_init.
    Returns best permutation found.
    """
    n = len(p_init)
    # Initialize X as the permutation matrix
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p_init] = 1.0
    
    best_p = p_init.copy()
    best_cost = _qap_cost(a, b, p_init)
    
    start_time = time.time()
    
    for t in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Gradient of f(X) = sum a[i,j] b[p_i, p_j]
        # In matrix form: f(X) = tr(A X B X^T) = tr(B X^T A X)
        # Gradient G = A X B + A^T X B^T
        # If A, B symmetric: G = A X B + A X B = 2 A X B
        # Let's use general formula
        grad = a @ X @ b + a.T @ X @ b.T
        grad = (grad + grad.T) / 2.0  # Symmetrize to handle numerical issues
        
        # Solve linear assignment problem: min <G, Y> s.t. Y in Birkhoff
        # This is equivalent to finding the permutation that minimizes sum G[i, p[i]]
        # Use Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(grad)
        
        # Y is the permutation matrix corresponding to row_ind, col_ind
        Y = np.zeros((n, n), dtype=np.float64)
        Y[row_ind, col_ind] = 1.0
        
        # Get the permutation from Y
        p_new = col_ind.copy() # p_new[i] = col_ind[i] where row_ind[i]=i? 
        # Wait, linear_sum_assignment returns row_ind, col_ind such that Y[row_ind, col_ind]=1
        # We need p such that Y[i, p[i]] = 1.
        # Since row_ind is a permutation of 0..n-1, we can map:
        # p_new[row_ind] = col_ind
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        # Step size
        # Standard FW step: 2/(t+2)
        step = 2.0 / (t + 2)
        
        # Update X
        X = (1.0 - step) * X + step * Y
        
        # Evaluate cost of the new permutation p_new
        cost = _qap_cost(a, b, p_new)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_new.copy()
            
        # Optional: Check convergence
        # If step is small and no improvement, break?
        # Let's just rely on max_iters and time budget.
        
    return best_p, best_cost

def _greedy_initial(a, b):
    """Greedy rank-matching initial solution."""
    n = len(a)
    # Sort facilities by total flow (row sum of A)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    # Sort locations by total distance (row sum of B)
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    
    # Ensure float64 for faster computation
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # 1. Initial Solution: Greedy + 2-opt
    p0 = _greedy_initial(a, b)
    p0, cost0 = _fast_2opt(a, b, p0, time_budget=10.0)
    
    best_p = p0.copy()
    best_cost = cost0
    
    # 2. Diversification with Frank-Wolfe
    # We will run FW from the current best, or perturbed versions, to find new basins.
    # To avoid getting stuck, we can perturb the starting point for FW occasionally.
    
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 15.0:
        # Create a starting point for FW
        # 50% chance to start from best, 50% from a perturbed best
        if rng.rand() < 0.5:
            p_start = best_p.copy()
        else:
            p_start = best_p.copy()
            # Perturb: swap 2-5 random elements
            num_swaps = rng.randint(2, 6)
            for _ in range(num_swaps):
                i, j = rng.randint(0, n, 2)
                if i != j:
                    p_start[i], p_start[j] = p_start[j], p_start[i]
        
        # Run FW
        # Allocate time based on remaining time
        remaining = deadline - time.time()
        fw_budget = min(10.0, remaining * 0.4)
        if fw_budget < 1.0:
            break
            
        p_fw, cost_fw = _frank_wolfe(a, b, p_start, max_iters=50, time_budget=fw_budget)
        
        # Polish with 2-opt
        remaining = deadline - time.time()
        opt_budget = min(15.0, remaining * 0.5)
        if opt_budget < 1.0:
            break
            
        p_2opt, cost_2opt = _fast_2opt(a, b, p_fw, time_budget=opt_budget)
        
        if cost_2opt < best_cost:
            best_cost = cost_2opt
            best_p = p_2opt.copy()
            
    return best_p.tolist()

# EVOLVE-BLOCK-END

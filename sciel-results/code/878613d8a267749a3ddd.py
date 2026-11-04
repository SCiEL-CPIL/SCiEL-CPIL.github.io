"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    return p

def _eigen_init(a, b, n, rng):
    """
    Generate initial permutation using eigenvalue-based heuristic.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        
        k = min(32, n)
        idx_a = np.argsort(np.abs(eigvals_a))[::-1][:k]
        idx_b = np.argsort(np.abs(eigvals_b))[::-1][:k]
        
        score_a = np.zeros(n)
        for idx in idx_a:
            score_a += np.abs(eigvals_a[idx]) * np.abs(eigvecs_a[:, idx])
        
        score_b = np.zeros(n)
        for idx in idx_b:
            score_b += np.abs(eigvals_b[idx]) * np.abs(eigvecs_b[:, idx])
        
        fac_order = np.argsort(score_a)[::-1]
        loc_order = np.argsort(score_b)
        
        p = np.zeros(n, dtype=int)
        for i in range(n):
            p[fac_order[i]] = loc_order[i]
        
        p_rev = np.zeros(n, dtype=int)
        for i in range(n):
            p[fac_order[i]] = loc_order[n - 1 - i]
        
        cost1 = np.sum(a * b[np.ix_(p, p)])
        cost2 = np.sum(a * b[np.ix_(p_rev, p_rev)])
        
        if cost1 <= cost2:
            return p
        else:
            return p_rev
    except:
        return rng.permutation(n)

def _sa_search(a, b, p_init, start_time, time_budget, rng):
    """
    Simulated Annealing with 2-swap moves, optimized for speed.
    Uses vectorized delta calculation.
    """
    n = len(p_init)
    p = p_init.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    best_cost = current_cost
    best_p = p.copy()
    
    a_sym = a + a.T
    
    # Temperature schedule
    T = current_cost * 0.01
    T_min = 1e-6
    alpha = 0.99
    deadline = start_time + time_budget
    
    # Precompute indices for random selection? No, random is faster.
    
    while time.time() < deadline - 0.5:
        # Batch process moves? 
        # For now, single step but very tight.
        
        # Pick two random positions
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
            
        # Vectorized delta for swap i, j
        # delta = a_sym[i] @ (D[j] - D[i]) + a_sym[:,i] @ (D[:,j] - D[:,i])
        #         - a[i,i]*(D[j,j]-D[i,i]) - a[j,j]*(D[i,i]-D[j,j])
        #         + (a[i,j]-a[j,i])*(D[i,j]-D[j,i])
        
        dD_row = D[j] - D[i]
        dD_col = D[:, j] - D[:, i]
        
        delta = np.dot(a_sym[i], dD_row) + np.dot(a_sym[:, i], dD_col)
        delta -= a[i, i] * (D[j, j] - D[i, i])
        delta -= a[j, j] * (D[i, i] - D[j, j])
        delta += (a[i, j] - a[j, i]) * (D[i, j] - D[j, i])
        
        delta = int(delta)
        
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, T_min)):
            p[i], p[j] = p[j], p[i]
            D[i], D[j] = D[j].copy(), D[i].copy()
            D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_p, best_cost

def _local_search_polish(a, b, p, time_budget, start_time):
    """
    Fast first-improvement 2-swap local search.
    Optimized to be O(n^2) per pass, with early exit.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    a_sym = a + a.T
    
    deadline = start_time + time_budget
    
    improved = True
    while improved and time.time() < deadline - 0.2:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline - 0.2:
                return p, current_cost
            
            # Vectorized delta for all j > i
            js = np.arange(i + 1, n)
            
            # delta(i, j) = a_sym[i] @ (D[j] - D[i]) + a_sym[:,i] @ (D[:,j] - D[:,i])
            #              - a[i,i]*(D[j,j]-D[i,i]) - a[j,j]*(D[i,i]-D[j,j])
            #              + (a[i,j]-a[j,i])*(D[i,j]-D[j,i])
            
            diff_D_rows = D[js] - D[i] # (m, n)
            term1 = np.sum(a_sym[i] * diff_D_rows, axis=1) # (m,)
            
            diff_D_cols = D[:, js] - D[:, i] # (n, m)
            term2 = np.sum(a_sym[:, i] * diff_D_cols, axis=0) # (m,)
            
            diag_i = a[i, i]
            diag_j = a[js, js]
            D_ii = D[i, i]
            D_jj = D[js, js]
            D_ij = D[i, js]
            D_ji = D[js, i]
            a_ij = a[i, js]
            a_ji = a[js, i]
            
            term3 = -diag_i * (D_jj - D_ii)
            term4 = -diag_j * (D_ii - D_jj)
            term5 = (a_ij - a_ji) * (D_ij - D_ji)
            
            deltas = term1 + term2 + term3 + term4 + term5
            deltas = deltas.astype(np.int64)
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                p[i], p[j] = p[j], p[i]
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                current_cost += int(min_delta)
                improved = True
                
    return p, best_cost, best_p # Wait, return signature must match? No, internal function.

def solve(a, b):
    n = len(a)
    start_time = time.time()
    # 120s limit. Use 115s to be safe.
    time_budget = 115.0
    
    rng = np.random.default_rng(42)
    
    # 1. Generate initial solutions
    p_hung = _hungarian_init(a, b)
    p_eigen = _eigen_init(a, b, n, rng)
    p_random = rng.permutation(n)
    
    initial_solutions = [p_hung, p_eigen, p_random]
    
    best_p = None
    best_cost = np.inf
    
    # 2. Polish each initial solution with local search
    # Allocate 20% of time to polishing
    polish_time = time_budget * 0.2
    time_per_polish = polish_time / len(initial_solutions)
    
    for p_init in initial_solutions:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 5:
            break
            
        p_ls, cost_ls = _local_search_polish(a, b, p_init, time_per_polish, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
    if best_p is None:
        best_p = p_hung.copy()
        best_cost = _compute_cost(a, b, best_p)
        
    # 3. Run SA from the best polished solution
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed - 2.0 # Leave 2s for final check
    if remaining > 10:
        p_sa, cost_sa = _sa_search(a, b, best_p, start_time, remaining, rng)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
            
    # 4. Final polish
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed - 0.5
    if remaining > 2:
        p_final, cost_final = _local_search_polish(a, b, best_p, remaining, start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END

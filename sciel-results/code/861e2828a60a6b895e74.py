import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    def get_cost(p):
        """Compute exact QAP cost for permutation p."""
        B_perm = b[:, p]
        S = a @ B_perm
        return int(np.sum(S[np.arange(n), p]))
    
    def compute_delta_swap(i, k, p):
        """Compute delta cost for swapping facilities i and k."""
        u, v = p[i], p[k]
        if u == v:
            return 0
        
        # Precompute row/col differences
        diff_row = b[v, p] - b[u, p]
        diff_col = b[p, v] - b[p, u]
        
        # Total contribution change
        term_i_row = np.dot(a[i, :], diff_row)
        term_k_row = np.dot(a[k, :], diff_row)
        term_i_col = np.dot(a[:, i], diff_col)
        term_k_col = np.dot(a[:, k], diff_col)
        
        total = (term_i_row - term_k_row) + (term_i_col - term_k_col)
        
        # Corrections for j=i and j=k
        term_i_val = (a[i,i]-a[k,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[i,k])*(b[u,v]-b[u,u])
        term_k_val = (a[i,k]-a[k,k])*(b[v,v]-b[u,v]) + (a[k,i]-a[k,k])*(b[v,v]-b[v,u])
        
        off_diag = total - term_i_val - term_k_val
        
        diag_ii = a[i,i] * (b[v,v] - b[u,u])
        diag_kk = a[k,k] * (b[u,u] - b[v,v])
        cross_ik = a[i,k] * (b[v,u] - b[u,v])
        cross_ki = a[k,i] * (b[u,v] - b[v,u])
        
        delta = diag_ii + diag_kk + cross_ik + cross_ki + off_diag
        return delta
    
    # Phase 1: Multiple restarts with Hungarian initialization
    # Use different linearization strategies to get diverse starting points
    
    def hungarian_init(method='standard'):
        """Initialize using Hungarian algorithm with different linearizations."""
        if method == 'standard':
            W = a @ b.T
        elif method == 'symmetric':
            # Symmetric combination
            W = 0.5 * (a @ b.T + b @ a.T)
        elif method == 'diagonal_weighted':
            # Weight by diagonal dominance
            diag_a = np.diag(a)
            diag_b = np.diag(b)
            W = a @ b.T + np.outer(diag_a, diag_b)
        elif method == 'row_sum':
            # Use row sums as weights
            row_sums_a = a.sum(axis=1)
            col_sums_b = b.sum(axis=0)
            W = a @ b.T + np.outer(row_sums_a, col_sums_b)
        else:
            W = a @ b.T
        
        row_ind, col_ind = linear_sum_assignment(W)
        p = np.full(n, -1, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    
    # Generate multiple diverse initial solutions
    initial_solutions = []
    methods = ['standard', 'symmetric', 'diagonal_weighted', 'row_sum']
    
    for method in methods:
        p = hungarian_init(method)
        cost = get_cost(p)
        initial_solutions.append((cost, p.copy()))
    
    # Sort by cost
    initial_solutions.sort(key=lambda x: x[0])
    
    # Take top 3 solutions for further optimization
    best_cost = initial_solutions[0][0]
    best_p = initial_solutions[0][1].copy()
    
    # Phase 2: Simulated Annealing from each good initial solution
    for sol_idx in range(min(3, len(initial_solutions))):
        if time.time() - start_time > time_limit:
            break
        
        p = initial_solutions[sol_idx][1].copy()
        current_cost = initial_solutions[sol_idx][0]
        
        # Estimate temperature
        sample_deltas = []
        for _ in range(50):
            i, k = np.random.choice(n, 2, replace=False)
            d = compute_delta_swap(i, k, p)
            sample_deltas.append(abs(d))
        
        T = np.mean(sample_deltas) * 2.0 if sample_deltas else 10000.0
        T_min = 1.0
        alpha = 0.99985
        
        sa_start = time.time()
        sa_time_limit = 25.0
        
        while time.time() - sa_start < sa_time_limit:
            i, k = np.random.choice(n, 2, replace=False)
            delta = compute_delta_swap(i, k, p)
            
            if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
                p[i], p[k] = p[k], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            T *= alpha
            if T < T_min:
                T = T_min
        
        # Local search after SA
        p = best_p.copy()
        current_cost = best_cost
        improved = True
        max_passes = 8
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                for k in range(i+1, n):
                    delta = compute_delta_swap(i, k, p)
                    if delta < 0:
                        p[i], p[k] = p[k], p[i]
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
    
    # Phase 3: Intensive local search with multiple passes
    p = best_p.copy()
    current_cost = best_cost
    
    # Use random restarts within local search to escape local optima
    for ls_round in range(5):
        if time.time() - start_time > time_limit - 5:
            break
        
        improved = True
        max_passes = 15
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            # Randomize the order slightly to avoid cycling
            indices = list(range(n))
            np.random.shuffle(indices)
            for i_idx in indices:
                i = indices[i_idx]
                for k_idx in range(i_idx+1, n):
                    k = indices[k_idx]
                    if i > k:
                        continue
                    delta = compute_delta_swap(i, k, p)
                    if delta < 0:
                        p[i], p[k] = p[k], p[i]
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
    
    # Phase 4: Final fine-tuning with very small perturbations
    if time.time() - start_time < time_limit - 2:
        p = best_p.copy()
        current_cost = best_cost
        
        # Try all pairs once more
        for i in range(n):
            for k in range(i+1, n):
                delta = compute_delta_swap(i, k, p)
                if delta < 0:
                    p[i], p[k] = p[k], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p.tolist()

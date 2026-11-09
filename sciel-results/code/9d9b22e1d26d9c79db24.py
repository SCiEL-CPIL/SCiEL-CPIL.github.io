# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _verify_solution(domain, values, K):
    """Verify that the solution satisfies all constraints efficiently."""
    if len(domain) != len(values):
        return False
    if len(domain) < 1:
        return False
    if np.any(domain <= 0) or np.any(domain > K):
        return False
    if not np.all(np.isfinite(values)):
        return False
    
    clipped = np.clip(values, -10, 10)
    S = np.sum(clipped / domain.astype(np.float64))
    
    has_1 = 1 in domain
    if has_1:
        f1_idx = np.where(domain == 1)[0]
        f1_val = clipped[f1_idx[0]] - S
        if f1_val < -10 or f1_val > 10:
            return False
    else:
        if -S < -10 or -S > 10:
            return False
    
    f = np.zeros(K + 1)
    for i, k in enumerate(domain):
        f[k] = clipped[i]
    
    if has_1:
        f[1] = f[1] - S
    else:
        f[1] = -S
    
    max_x = 10 * K
    
    # Efficient M(x) computation using difference array technique
    M_diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for k in range(1, K + 1):
        fk = f[k]
        if fk == 0:
            continue
        j_max = max_x // k
        for j in range(1, j_max + 1):
            start = j * k
            end = min((j + 1) * k - 1, max_x)
            M_diff[start] += fk * j
            M_diff[end + 1] -= fk * j
    
    M = np.cumsum(M_diff)
    
    if np.any(M[1:] > 1.0001):
        return False
    
    return True

def _compute_metric(domain, values, K):
    """Compute the metric a = -sum f(k)*log(k)/k"""
    clipped = np.clip(values, -10, 10)
    S = np.sum(clipped / domain.astype(np.float64))
    
    f = np.zeros(K + 1)
    for i, k in enumerate(domain):
        f[k] = clipped[i]
    
    has_1 = 1 in domain
    if has_1:
        f[1] = f[1] - S
    else:
        f[1] = -S
    
    a = 0.0
    for k in range(2, K + 1):
        if f[k] != 0:
            a -= f[k] * math.log(k) / k
    return a

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    from scipy.optimize import linprog
    
    best_domain = None
    best_values = None
    best_metric = -np.inf
    
    K_candidates = [3000, 2000, 1500, 1000]
    
    for K in K_candidates:
        if time.time() > deadline - 150:
            break
        
        n_vars = K - 1
        if n_vars <= 0:
            continue
        
        k_vals = np.arange(2, K + 1, dtype=np.float64)
        c = np.log(k_vals) / k_vals
        
        x_set = set()
        max_x = 10 * K
        
        # Include x=1 (critical constraint)
        x_set.add(1)
        
        # Include small x values densely
        for x in range(1, min(200, max_x) + 1):
            x_set.add(x)
        
        # Include x values that are multiples of small k
        for k in range(2, min(K, 400) + 1):
            for m in range(1, min(max_x // k, 30) + 1):
                x = m * k
                if x <= max_x:
                    x_set.add(x)
        
        # Include x = k*j - 1 (boundary points where floor changes)
        for k in range(2, min(K, 250) + 1):
            for j in range(1, min(max_x // k, 25) + 1):
                x = k * j - 1
                if 1 <= x <= max_x:
                    x_set.add(x)
        
        x_vals = np.array(sorted(x_set), dtype=np.int64)
        n_constraints = len(x_vals)
        
        max_constraints = 8000
        if n_constraints > max_constraints:
            x_vals_sorted = np.sort(x_vals)
            n_small = min(200, len(x_vals_sorted))
            x_small = x_vals_sorted[:n_small]
            x_large = x_vals_sorted[n_small:]
            if len(x_large) > 0:
                n_large = min(max_constraints - n_small, len(x_large))
                idx_large = np.random.choice(len(x_large), size=n_large, replace=False)
                x_large = x_large[np.sort(idx_large)]
                x_vals = np.concatenate([x_small, x_large])
            else:
                x_vals = x_small
            x_vals = np.sort(x_vals)
            n_constraints = len(x_vals)
        
        # Correct constraint: M(x) = sum_{k>=2} f(k) * (floor(x/k) - x/k) <= 1
        A = np.floor(x_vals[:, None].astype(np.float64) / k_vals[None, :]) - x_vals[:, None].astype(np.float64) / k_vals[None, :]
        b = np.ones(n_constraints)
        bounds = [(-10, 10)] * n_vars
        
        time_limit = min(120, deadline - time.time() - 80)
        if time_limit <= 10:
            continue
        
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs', 
                         options={'time_limit': time_limit, 'presolve': True,
                                  'dual_feasibility_tolerance': 1e-10,
                                  'primal_feasibility_tolerance': 1e-10})
            
            if res.success and res.x is not None:
                g = res.x
                f1 = -np.sum(g / k_vals)
                
                if abs(f1) > 10:
                    continue
                
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = np.zeros(K, dtype=np.float64)
                values[0] = f1
                values[1:] = g
                
                if _verify_solution(domain, values, K):
                    metric = _compute_metric(domain, values, K)
                    if metric > best_metric:
                        best_metric = metric
                        best_domain = domain.copy()
                        best_values = values.copy()
                        
        except Exception as e:
            continue
    
    if best_domain is not None:
        return best_domain, best_values
    
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END

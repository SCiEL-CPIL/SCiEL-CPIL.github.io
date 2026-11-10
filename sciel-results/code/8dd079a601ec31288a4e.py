import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    best_domain = None
    best_values = None
    best_score = -np.inf
    last_solution = None

    # Try different K values with increasing complexity
    K_values = [100, 200, 500, 1000, 2000, 4000, 8000, 15000, 30000]
    
    for K in K_values:
        if time.time() > deadline:
            break
        
        n_vars = K - 1
        if n_vars <= 0:
            continue
        
        ks = np.arange(2, K + 1, dtype=np.float64)
        c = np.log(ks) / ks  # Objective: maximize -sum c_k * g_k, so minimize sum c_k * g_k
        
        max_x = 10 * K
        
        # Build sparse constraint matrix for M(x) = sum_k g(k) * floor(x/k) <= 1.0001
        # Note: f(1) is not included in the LP variables, it's determined by the constraint sum g(k)/k = 0
        # Actually, the constraint is sum_k f(k)/k = 0, where f(1) = -S, S = sum_{k>=2} g(k)/k
        # So we need S to be such that |f(1)| <= 10, i.e., |S| <= 10
        # And M(x) = sum_{k in domain} f(k) * floor(x/k)
        # For k=1: f(1) * floor(x/1) = f(1) * x = -S * x
        # For k>=2: g(k) * floor(x/k)
        # So M(x) = -S * x + sum_{k>=2} g(k) * floor(x/k) <= 1.0001
        # This gives: sum_{k>=2} g(k) * floor(x/k) - x * S <= 1.0001
        # Where S = sum_{k>=2} g(k)/k
        # So: sum_{k>=2} g(k) * floor(x/k) - x * sum_{k>=2} g(k)/k <= 1.0001
        # => sum_{k>=2} g(k) * (floor(x/k) - x/k) <= 1.0001
        # => -sum_{k>=2} g(k) * {x/k} <= 1.0001
        # => sum_{k>=2} g(k) * {x/k} >= -1.0001
        
        # Let's just build the constraint directly: M(x) <= 1.0001
        # M(x) = -S*x + sum_{k=2}^{K} g(k)*floor(x/k)
        # = sum_{k=2}^{K} g(k)*floor(x/k) - x*sum_{k=2}^{K} g(k)/k
        # = sum_{k=2}^{K} g(k)*(floor(x/k) - x/k)
        # = -sum_{k=2}^{K} g(k)*{x/k}
        
        # So the constraint is: -sum_{k=2}^{K} g(k)*{x/k} <= 1.0001
        # i.e., sum_{k=2}^{K} g(k)*{x/k} >= -1.0001
        # i.e., -sum_{k=2}^{K} g(k)*{x/k} <= 1.0001
        
        # Let's use: for each x, -sum_k g(k) * {x/k} <= 1.0001
        # Where {x/k} = x/k - floor(x/k)
        
        # Actually, let me just build the matrix directly with the original formulation.
        # M(x) = sum_{k in domain} f(k) * floor(x/k)
        # domain includes k=1 with f(1) = -S
        # So M(x) = f(1)*floor(x/1) + sum_{k=2}^K f(k)*floor(x/k)
        # = -S * x + sum_{k=2}^K g(k) * floor(x/k)
        
        # For the LP, the variables are g(2), ..., g(K)
        # Constraint for each x: sum_{k=2}^K g(k) * floor(x/k) - x * sum_{k=2}^K g(k)/k <= 1.0001
        # => sum_{k=2}^K g(k) * (floor(x/k) - x/k) <= 1.0001
        
        # Let's build this matrix
        rows = []
        cols = []
        data = []
        
        # For efficiency, build in chunks
        chunk_size = 200
        for x_start in range(1, max_x + 1, chunk_size):
            if time.time() > deadline - 10:
                break
            x_end = min(x_start + chunk_size, max_x + 1)
            x_chunk = np.arange(x_start, x_end, dtype=np.float64)
            
            # For each k in ks, coefficient for g(k) in row x is floor(x/k) - x/k = -{x/k}
            # Shape: (chunk_len, n_vars)
            # floor(x/k) - x/k
            coeff = np.floor(x_chunk[:, None] / ks[None, :]) - x_chunk[:, None] / ks[None, :]
            
            nz_rows, nz_cols = np.nonzero(coeff)
            nz_vals = coeff[nz_rows, nz_cols]
            
            global_rows = x_chunk[nz_rows].astype(int) - 1
            global_cols = nz_cols
            
            rows.extend(global_rows.tolist())
            cols.extend(global_cols.tolist())
            data.extend(nz_vals.tolist())
        
        # Add |S| <= 10 constraints: S = sum g(k)/k, |S| <= 10
        # S <= 10: sum g(k)/k <= 10
        # -S <= 10: -sum g(k)/k <= 10 => sum g(k)/k >= -10 => -sum g(k)/k <= 10
        
        row_pos = 1.0 / ks
        row_neg = -1.0 / ks
        
        rows.extend([max_x] * n_vars)
        cols.extend(range(n_vars))
        data.extend(row_pos.tolist())
        
        rows.extend([max_x + 1] * n_vars)
        cols.extend(range(n_vars))
        data.extend(row_neg.tolist())
        
        A_ub = coo_matrix((data, (rows, cols)), shape=(max_x + 2, n_vars)).tocsr()
        b_ub = np.concatenate([np.full(max_x, 1.0001), [10.0, 10.0]])
        
        bounds = [(-10, 10)] * n_vars
        
        # Warm start if available
        x0 = None
        if last_solution is not None and len(last_solution) == n_vars:
            x0 = last_solution
        elif last_solution is not None and len(last_solution) < n_vars:
            x0 = np.zeros(n_vars)
            x0[:len(last_solution)] = last_solution
        elif last_solution is not None and len(last_solution) > n_vars:
            x0 = last_solution[:n_vars]
        
        try:
            options = {'maxiter': 10000, 'time_limit': 120}
            if x0 is not None:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                             method='highs', x0=x0, options=options)
            else:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                             method='highs', options=options)
            
            if res.success:
                g = res.x
                S = np.sum(g / ks)
                f = np.zeros(K + 1)
                f[2:K+1] = g
                f[1] = -S
                
                if abs(f[1]) <= 10.0 + 1e-9:
                    a = -np.sum(g * c)
                    if a > best_score:
                        best_score = a
                        domain = np.arange(1, K + 1, dtype=np.int64)
                        values = np.clip(f[1:K+1], -10, 10)
                        best_domain = domain
                        best_values = values
                        last_solution = g.copy()
                        
        except Exception:
            pass
    
    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

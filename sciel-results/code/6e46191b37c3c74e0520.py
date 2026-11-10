import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    
    def deadline_ok():
        return time.time() - start_time < 580
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Two-stage approach:
    # Stage 1: Identify critical x values where the constraint is most likely binding
    # Stage 2: Solve reduced LP on those x values
    
    for K in [200, 500, 1000, 2000, 5000, 10000]:
        if not deadline_ok():
            break
        
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks
        max_x = 10 * K
        
        # Stage 1: Identify critical x values
        # The constraint is sum_k g(k) * {x/k} >= -1.0001
        # Critical x are where many {x/k} values are simultaneously small (for positive g) 
        # or large (for negative g)
        
        # Use x values that are multiples of small integers (where many {x/k} = 0)
        # and x values near multiples (where {x/k} is small)
        critical_x = set()
        
        # x = 1, 2, ..., min(K, 200) - small x values
        for x in range(1, min(K, 200) + 1):
            critical_x.add(x)
        
        # x = multiples of small k: x = m*k for m=1,2,..., floor(10K/k)
        for k in range(2, min(K, 50) + 1):
            for m in range(1, min(10 * K // k, 100) + 1):
                x = m * k
                if 1 <= x <= max_x:
                    critical_x.add(x)
        
        # x = multiples of larger k (sparser)
        for k in range(50, min(K, 200) + 1, 5):
            for m in range(1, min(10 * K // k, 50) + 1):
                x = m * k
                if 1 <= x <= max_x:
                    critical_x.add(x)
        
        # x near max_x
        for x in range(max(1, max_x - 100), max_x + 1):
            critical_x.add(x)
        
        # Add some random samples
        if deadline_ok():
            n_random = min(5000, max_x)
            random_x = np.random.randint(1, max_x + 1, size=n_random)
            for x in random_x:
                critical_x.add(int(x))
        
        x_vals = np.array(sorted(critical_x), dtype=np.float64)
        n_constraints = len(x_vals)
        
        if not deadline_ok():
            continue
        
        # Stage 2: Build reduced constraint matrix
        # A_ub[i,j] = -{x_i / k_j} for the constraint sum_k g(k)*{x/k} >= -1.0001
        # which is -sum_k g(k)*{x/k} <= 1.0001
        
        # Build in chunks to avoid memory issues
        A_ub_parts = []
        chunk_size = min(1000, n_constraints)
        for i in range(0, n_constraints, chunk_size):
            j = min(i + chunk_size, n_constraints)
            x_chunk = x_vals[i:j]
            A_chunk = -(x_chunk[:, None] % ks[None, :]) / ks[None, :]
            A_ub_parts.append(A_chunk)
        
        if not deadline_ok():
            continue
        
        A_ub = np.vstack(A_ub_parts)
        b_ub = np.ones(n_constraints)
        
        # Add bounds on f(1) = -sum(g/k)
        row_pos = 1.0 / ks
        row_neg = -1.0 / ks
        
        A_ub_full = np.vstack([A_ub, row_pos, row_neg])
        b_ub_full = np.concatenate([b_ub, [10.0, 10.0]])
        
        bounds = [(-10, 10)] * n_vars
        
        if not deadline_ok():
            continue
        
        try:
            res = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=bounds, 
                         method='highs', options={'presolve': True, 'time_limit': 30})
            
            if res.success:
                g = res.x
                f = np.zeros(K + 1)
                f[2:K+1] = g
                S = np.sum(g / ks)
                f[1] = -S
                
                if abs(f[1]) <= 10.0 + 1e-9:
                    # Verify feasibility for ALL x in [1, 10K] (not just critical ones)
                    if not deadline_ok():
                        continue
                    
                    # Check in chunks
                    max_M = -np.inf
                    check_chunk = 10000
                    for start in range(1, max_x + 1, check_chunk):
                        end = min(start + check_chunk, max_x + 1)
                        x_check = np.arange(start, end, dtype=np.float64)
                        M_vals = -np.sum(g[None, :] * (x_check[:, None] % ks[None, :]) / ks[None, :], axis=1)
                        max_M = max(max_M, np.max(M_vals))
                        if not deadline_ok():
                            break
                    
                    if max_M <= 1.0001 + 1e-9:
                        a = -np.sum(g * c)
                        if a > best_score:
                            best_score = a
                            domain = np.arange(1, K + 1)
                            values = np.clip(f[1:K+1], -10, 10)
                            best_domain = domain
                            best_values = values
                            print(f"K={K}, score={a:.6f}, time={time.time()-start_time:.1f}s")
                            
        except Exception as e:
            pass
    
    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

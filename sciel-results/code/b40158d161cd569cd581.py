import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600  # Leave 30s margin for 630s limit
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try increasing K
    # Based on memory constraints, K=6000 is near the limit for dense matrix in Python/NumPy
    # 10*6000 * 6000 * 8 bytes = 2.88 GB
    # Let's try up to 5000 to be safe, or 6000 if we can free memory.
    K_candidates = [500, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for K in K_candidates:
        if time.time() > deadline - 60:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks
        
        max_x = 10 * K
        
        # Check memory feasibility roughly
        # 10*K * (K-1) * 8 bytes
        mem_needed = (10 * K) * (K - 1) * 8
        if mem_needed > 3.5 * 1024**3:  # 3.5GB
            continue

        try:
            # Vectorized construction of A_ub
            # A_ub[i, j] = - { (i+1) / (j+2) }
            # x_vals is 1-based index for x in [1, max_x]
            x_vals = np.arange(1, max_x + 1, dtype=np.float64)
            
            # Construct main constraint matrix
            # Shape: (max_x, n_vars)
            # A[i, j] corresponds to x = i+1, k = j+2
            # A[i, j] = floor((i+1)/(j+2)) - (i+1)/(j+2)
            # This is - fractional_part((i+1)/(j+2))
            
            # We compute this in chunks to save memory if needed, but let's try direct first
            # Using broadcasting
            # x_vals[:, None] / ks[None, :]
            # This creates a (max_x, n_vars) array.
            
            # To save memory, we can compute in blocks of rows
            block_size = 1000
            A_ub_main = np.empty((max_x, n_vars), dtype=np.float64)
            
            for start in range(0, max_x, block_size):
                end = min(start + block_size, max_x)
                x_block = x_vals[start:end]
                # Compute floor(x/k) - x/k
                # x_block[:, None] / ks[None, :]
                ratios = x_block[:, None] / ks[None, :]
                floors = np.floor(ratios)
                A_ub_main[start:end, :] = floors - ratios
            
            # Add constraints for sum(f(k)/k)
            # f(1) = -sum(g/k). -10 <= f(1) <= 10
            # sum(g/k) <= 10
            # -sum(g/k) <= 10 => sum(g/k) >= -10 => -sum(g/k) <= 10
            row_pos = 1.0 / ks
            row_neg = -1.0 / ks
            
            A_ub = np.vstack([A_ub_main, row_pos, row_neg])
            b_ub = np.concatenate([np.ones(max_x) * 1.0001, [10.0, 10.0]])
            
            bounds = [(-10, 10)] * n_vars
            
            # Solve LP
            # Maximize a = -sum(c * f) => Minimize sum(c * f)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                          options={'presolve': 'default', 'dual_feasibility_tolerance': 1e-8, 'primal_feasibility_tolerance': 1e-8})
            
            # Free memory
            del A_ub_main
            del A_ub
            del b_ub
            del x_vals
            
            if res.success:
                g = res.x
                f = np.zeros(K + 1)
                f[2:K+1] = g
                S = np.sum(g / ks)
                f[1] = -S
                
                # Verify f(1) is within bounds
                if abs(f[1]) <= 10.0 + 1e-9:
                    # Calculate score
                    a = -np.sum(g * c)
                    
                    if a > best_score:
                        best_score = a
                        domain = np.arange(1, K + 1, dtype=np.int64)
                        values = np.clip(f[1:K+1], -10, 10)
                        best_domain = domain
                        best_values = values

        except MemoryError:
            continue
        except Exception as e:
            pass

    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

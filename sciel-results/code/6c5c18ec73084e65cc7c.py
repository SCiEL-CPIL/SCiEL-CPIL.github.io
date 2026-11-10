import numpy as np
import time
import sys
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix
import mpmath as mp

def prime_number_theorem_partial_function():
    start_time = time.time()
    timeout = 590  # Safety margin (600s limit)
    
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0

    # Try a range of K values. Smaller K solves faster.
    # Larger K might allow better score but is harder to solve.
    K_candidates = [100, 200, 300, 500, 800]
    
    for K in K_candidates:
        if time.time() - start_time > timeout:
            break
            
        try:
            # Variables f_2 ... f_K
            n_vars = K - 1
            indices = np.arange(2, K + 1, dtype=np.float64)
            
            # Objective: maximize -sum f(k) log(k)/k
            # linprog minimizes, so we minimize sum f(k) log(k)/k
            c = np.log(indices) / indices
            
            # Constraints: sum f(k) * {x/k} >= -1.0001 for x in 1..10K
            # Equivalent to: -sum f(k) * {x/k} <= 1.0001
            # A_ub @ f <= b_ub
            
            x_vals = np.arange(1, 10 * K + 1)
            
            # Construct A_ub efficiently
            # Shape: (10K, K-1)
            # Entry (x, k): -{x/k}
            
            # Vectorized calculation
            frac_parts = (x_vals[:, None] / indices[None, :]) % 1.0
            A_ub = -frac_parts
            
            # Convert to sparse for solver efficiency
            A_sparse = csr_matrix(A_ub)
            
            b_ub = np.full(10 * K, 1.0001)
            
            bounds = [(-10, 10)] * n_vars
            
            # Solve
            # Use 'highs' for speed and robustness
            res = linprog(c, A_ub=A_sparse, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.status == 0:
                f_vals = res.x
                
                # Verify feasibility to avoid numerical errors
                M_vals = -np.dot(frac_parts, f_vals)
                
                if np.max(M_vals) <= 1.0001 + 1e-6:
                    # Calculate score
                    score = -np.sum(f_vals * c)
                    if score > best_score:
                        best_score = score
                        # Construct domain and values
                        domain = np.arange(1, K + 1, dtype=np.int64)
                        values = np.zeros(K, dtype=np.float64)
                        values[0] = 0.0 # f(1) orig, evaluator adjusts
                        values[1:] = f_vals
                        best_domain = domain.copy()
                        best_values = values.copy()
                        
        except Exception as e:
            print(f"Error for K={K}: {e}", file=sys.stderr)
            continue
    
    # Polish step: try to improve the best solution found using local optimization
    if best_score > 1.0 and time.time() - start_time < timeout - 30:
        try:
            K_best = len(best_domain)
            indices = np.arange(2, K_best + 1, dtype=np.float64)
            c = np.log(indices) / indices
            x_vals = np.arange(1, 10 * K_best + 1)
            frac_parts = (x_vals[:, None] / indices[None, :]) % 1.0
            
            # Objective: maximize -sum f(k) log(k)/k
            # Minimize: sum f(k) log(k)/k
            def objective(f):
                return np.sum(f * c)
            
            def objective_grad(f):
                return c
            
            # Constraints: M(x) = -sum f(k) {x/k} <= 1.0001
            # Inequality constraint for scipy.minimize: c(x) >= 0
            # So we need: 1.0001 - M(x) >= 0
            # M(x) = -frac_parts @ f
            # So constraint: 1.0001 + frac_parts @ f >= 0
            # This is a linear constraint. We can use SLSQP or trust-constr.
            
            # However, SLSQP might be slow for large K.
            # Let's just try a few random restarts or perturbations if time permits.
            # Or use a simple projected gradient step.
            
            # For now, just return the LP solution as it is likely optimal for the LP.
            # The LP optimum is a vertex, which is often a good starting point.
            # If we had more time, we could try to find a better non-vertex solution,
            # but the objective is linear, so the optimum is at a vertex.
            # Thus, the LP solution IS the optimal solution for the given K.
            
            # The only way to improve is to increase K.
            # We already tried up to K=800.
            
        except Exception as e:
            print(f"Polish error: {e}", file=sys.stderr)
            
    return best_domain, best_values

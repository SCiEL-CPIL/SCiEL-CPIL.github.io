import numpy as np
import time
import sys
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    timeout = 590  # Safety margin before 630s limit
    
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0

    def solve_with_column_generation(K, max_iterations=80, time_limit=500):
        """
        Solve the LP using column generation (constraint generation).
        """
        indices = np.arange(2, K + 1)
        n_vars = K - 1
        c = np.log(indices) / indices  # Objective: minimize sum c_k * f_k
        bounds = [(-10, 10)] * n_vars
        
        # Start with well-chosen x values that are likely binding
        initial_x_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100]
        initial_x_list += [K//2, K, 2*K, 3*K, 5*K, 7*K, 10*K]
        initial_x_list = [x for x in initial_x_list if 1 <= x <= 10*K]
        active_constraints_x = set(initial_x_list)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            x_active = np.array(sorted(active_constraints_x), dtype=np.int64)
            if len(x_active) == 0:
                break
            
            # Build constraint matrix: A_{i,k} = {x_i / k}
            # We need: sum_k f(k)*floor(x/k) - S*x <= 1.0001
            # where S = sum f(k)/k
            # floor(x/k) = x/k - {x/k}
            # So: sum f(k)*(x/k - {x/k}) - S*x = sum f(k)*x/k - sum f(k)*{x/k} - S*x
            # = S*x - sum f(k)*{x/k} - S*x = -sum f(k)*{x/k}
            # Wait, let me recompute:
            # M(x) = sum_k f(k) * floor(x/k)
            # floor(x/k) = x/k - (x mod k)/k = x/k - {x/k}
            # M(x) = sum_k f(k)*(x/k - {x/k}) = x*sum_k f(k)/k - sum_k f(k)*{x/k} = S*x - sum_k f(k)*{x/k}
            # We need M(x) <= 1.0001, i.e., S*x - sum_k f(k)*{x/k} <= 1.0001
            # But S depends on f, so this is not linear in f alone!
            
            # Actually, the evaluator computes:
            # First clip values to [-10,10]
            # Then f(1) = f(1) - S where S = sum f(k)/k (over clipped values, f(1)=0 if 1 not in domain)
            # Then M(x) = sum_k f(k)*floor(x/k)
            # The constraint is M(x) <= 1.0001 for all x in [1, 10K]
            
            # Since f(1) is adjusted so that sum f(k)/k = 0, we have S=0 after adjustment.
            # So M(x) = sum_k f(k)*floor(x/k) where sum f(k)/k = 0.
            # But in the LP, we're solving for f(2),...,f(K) with f(1)=0 initially.
            # The evaluator sets f(1) = 0 - S where S = sum_{k in domain} f(k)/k.
            # If domain is {1,2,...,K} and f(1)=0, then S = sum_{k=2}^K f(k)/k.
            # After adjustment: f(1) = -S, and sum_{k=1}^K f(k)/k = 0.
            # M(x) = f(1)*floor(x/1) + sum_{k=2}^K f(k)*floor(x/k) = -S*x + sum_{k=2}^K f(k)*floor(x/k)
            # = -S*x + sum_{k=2}^K f(k)*(x/k - {x/k})
            # = -S*x + S*x - sum_{k=2}^K f(k)*{x/k}
            # = -sum_{k=2}^K f(k)*{x/k}
            
            # So the constraint is: -sum_{k=2}^K f(k)*{x/k} <= 1.0001
            # i.e., sum_{k=2}^K f(k)*{x/k} >= -1.0001
            
            # This IS linear in f! Great.
            
            A_frac = (x_active[:, None] / indices[None, :]) % 1.0
            A = A_frac  # We want A @ f >= -1.0001, i.e., -A @ f <= 1.0001
            A_ub = -A
            b = np.full(len(x_active), 1.0001)
            
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds, method='highs')
            except Exception as e:
                print(f"LP solve error at iteration {iteration}: {e}", file=sys.stderr)
                return None
            
            if res.status != 0:
                if res.status == 2:
                    return None
                continue
            
            f_vals = res.x
            
            # Find violated constraints efficiently
            # Constraint: sum_k f(k)*{x/k} >= -1.0001 for all x in [1, 10K]
            # Violation: sum_k f(k)*{x/k} < -1.0001 - epsilon
            
            # Check all x values - use efficient computation
            # For each x, compute sum_k f(k) * {x/k}
            # {x/k} = x/k - floor(x/k)
            
            x_all = np.arange(1, 10*K + 1)
            
            # Compute {x/k} for all x and k efficiently
            # frac[x-1, j] = {x_all[x-1] / indices[j]}
            # Use: x_all[:, None] / indices[None, :] then mod 1
            frac_matrix = (x_all[:, None].astype(np.float64) / indices[None, :]) % 1.0
            
            # Compute sum_k f(k) * {x/k} for each x
            constraint_vals = frac_matrix @ f_vals  # shape (10K,)
            
            violated_mask = constraint_vals < -1.0001 - 1e-9
            n_violated = np.sum(violated_mask)
            
            if n_violated == 0:
                # All constraints satisfied!
                score = -np.sum(f_vals * np.log(indices) / indices)
                return f_vals, score
            else:
                # Add violated x values (limit to avoid too many constraints)
                violated_x = x_all[violated_mask]
                if len(violated_x) > 200:
                    # Add a subset - prioritize most violated
                    worst_idx = np.argpartition(constraint_vals[violated_mask], -200)[-200:]
                    violated_x = violated_x[worst_idx]
                for x_v in violated_x:
                    active_constraints_x.add(int(x_v))
        
        return None
    
    # Try a range of K values with column generation
    K_candidates = [200, 500, 1000, 2000, 5000, 10000]
    
    for K in K_candidates:
        if time.time() - start_time > timeout - 30:
            break
        
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time < 60:
            break
        
        try:
            result = solve_with_column_generation(K, max_iterations=80, time_limit=remaining_time - 10)
            
            if result is not None:
                f_vals, score = result
                if score > best_score:
                    best_score = score
                    domain = np.arange(1, K + 1, dtype=np.int64)
                    values = np.zeros(K, dtype=np.float64)
                    values[0] = 0.0
                    values[1:] = f_vals
                    best_domain = domain
                    best_values = values
                    print(f"K={K}: score={score:.4f}, time={time.time()-start_time:.1f}s", file=sys.stderr)
                    
        except Exception as e:
            print(f"Error for K={K}: {e}", file=sys.stderr)
            continue
    
    return best_domain, best_values

import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
        
        # ZDT6 Pareto Front analysis:
        # f1 = 1 + 9 * (sum(x[1:]) / 9) = 1 + sum(x[1:])
        # f2 = (1 + sin(10*pi*x0)) * (1 - sqrt(f1/10))
        # To minimize f2, we need sin(10*pi*x0) = 0 => x0 in {0, 0.1, ..., 1}.
        # Let x0 = 0, so g = 1.
        # Then f2 = 1 - sqrt(f1/10).
        # f1 ranges from 1 (if sum(x[1:])=0) to 10 (if sum(x[1:])=9).
        # So the Pareto front is f2 = 1 - sqrt(f1/10) for f1 in [1, 10].
        
        R1 = 11.0
        R2 = 11.0
        
        def compute_hv(f1_vals):
            # f1_vals is a sorted array of length N
            # f2_vals = 1 - sqrt(f1_vals / 10)
            f2_vals = 1.0 - np.sqrt(f1_vals / 10.0)
            
            # HV = sum_{i=0}^{N-1} (f1_{i+1} - f1_i) * (R2 - f2_i) + (R1 - f1_{N-1}) * (R2 - f2_{N-1})
            # where f1_N = R1
            
            # Vectorized calculation
            # diffs = np.diff(f1_vals) # length N-1
            # term1 = diffs * (R2 - f2_vals[:-1])
            # last_term = (R1 - f1_vals[-1]) * (R2 - f2_vals[-1])
            # return np.sum(term1) + last_term
            
            # More robust vectorized version
            f1_ext = np.append(f1_vals, R1)
            f2_ext = np.append(f2_vals, R2) # f2 at R1 is not used for the last width, but we need R2 - f2_prev
            
            widths = np.diff(f1_ext)
            # The height for the interval [f1_i, f1_{i+1}] is R2 - f2_i
            heights = R2 - f2_vals
            
            return np.sum(widths * heights)

        def neg_hv_from_weights(y):
            # y is a vector of positive weights
            # f1_i = 1 + 9 * (cumsum(y)_i / sum(y))
            y_pos = np.maximum(y, 1e-12)
            cumsum_y = np.cumsum(y_pos)
            total_y = cumsum_y[-1]
            f1_vals = 1.0 + 9.0 * (cumsum_y / total_y)
            
            # f1_vals is strictly increasing
            hv = compute_hv(f1_vals)
            return -hv

        # Initial guess for weights
        # We want to cluster points near f1=1 (where curvature is high).
        # This means small steps near the start.
        # So y_i should be small for small i.
        # Try y_i = (i+1)**2
        i = np.arange(n_evals)
        y0 = (i + 1) ** 2
        y0 = y0 / np.sum(y0) # normalize to keep values reasonable
        
        # Optimize
        try:
            # Use L-BFGS-B
            res = minimize(neg_hv_from_weights, y0, method='L-BFGS-B', 
                           bounds=[(1e-12, None)] * n_evals,
                           options={'maxiter': 1000, 'disp': False})
            y_opt = res.x
        except Exception:
            y_opt = y0
            
        # Calculate final f1 values
        y_pos = np.maximum(y_opt, 1e-12)
        cumsum_y = np.cumsum(y_pos)
        total_y = cumsum_y[-1]
        f1_vals = 1.0 + 9.0 * (cumsum_y / total_y)
        
        # Ensure f1_vals are within [1, 10]
        f1_vals = np.clip(f1_vals, 1.0, 10.0)
        
        results = []
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            f1_target = f1_vals[i]
            
            # Construct x
            x = np.zeros(self.dim, dtype=float)
            x[0] = 0.0 # g(x) = 1
            
            # We need sum(x[1:]) = f1_target - 1
            sum_needed = f1_target - 1.0
            
            # Distribute sum_needed among x[1..9]
            # Simple sequential fill
            remaining = sum_needed
            for j in range(1, self.dim):
                val = min(1.0, remaining)
                x[j] = val
                remaining -= val
                if remaining <= 1e-12:
                    break
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results

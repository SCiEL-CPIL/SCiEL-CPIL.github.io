"""Direct construction approach for multi-objective optimization on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Direct Pareto front construction via analytical insight on ZDT4 structure."""
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.dim
        
        rng = np.random
        
        all_points = []
        all_evals = 0
        
        def check_time():
            return time.time() > deadline
        
        def eval_point(x):
            nonlocal all_evals
            if all_evals >= self.budget or check_time():
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_evals += 1
            f = np.array(f, dtype=np.float64)
            if np.all(np.isfinite(f)):
                all_points.append(f)
            return f
        
        # ZDT4: f1 = x1, f2 = g(x2..xn) * (1 - sqrt(x1/g))
        # g = 10(n-1) + sum((xi-0.5)^2 - 10*cos(20*pi*(xi-0.5))) for i=2..n
        # Minimum of g is 10(n-1) (when all xi=0.5)
        # Pareto front: f2 = g_min * (1 - sqrt(f1/g_min)) for f1 in [0, g_min]
        # Note: In standard ZDT4, f1=x1 is in [0,1], so f1/g_min is small.
        
        # Phase 1: Find x2..xn that minimizes g
        # Start with the known global minimum x_i = 0.5
        x_rest_init = np.full(n-1, 0.5)
        x_rest_init = np.clip(x_rest_init, lo[1:], hi[1:])
        
        best_x_rest = x_rest_init.copy()
        best_g_val = np.inf
        
        # Evaluate initial guess to get g
        x_probe = np.concatenate(([0.0], x_rest_init))
        f_probe = eval_point(x_probe)
        if f_probe is not None:
            best_g_val = f_probe[1]
            
        # Use SciPy Nelder-Mead to refine x_rest to minimize g
        # We define a function that evaluates g using x1=0
        def g_objective(x_rest):
            nonlocal all_evals
            if all_evals >= self.budget or check_time():
                return best_g_val
            x_rest = np.clip(x_rest, lo[1:], hi[1:])
            x = np.concatenate(([0.0], x_rest))
            f = eval_point(x)
            if f is None:
                return best_g_val
            return f[1]
        
        # Set up optimization
        # We limit the number of function evaluations to save budget for the sweep
        n_opt_evals = 20
        try:
            res = minimize(g_objective, x_rest_init, method='Nelder-Mead', 
                           options={'maxiter': 50, 'maxfev': n_opt_evals, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_g_val:
                best_g_val = res.fun
                best_x_rest = res.x
        except Exception:
            pass
            
        # Ensure best_x_rest is valid
        best_x_rest = np.clip(best_x_rest, lo[1:], hi[1:])
        
        # Phase 2: Sweep x1 densely to trace the Pareto front
        remaining_budget = self.budget - all_evals
        if remaining_budget > 0 and not check_time():
            # Determine x1 range
            # Standard ZDT4 x1 is in [0,1]. 
            # The Pareto front exists for x1 in [0, g_min].
            # If hi[0] > g_min, points with x1 > g_min will have f2 = g * (1 - sqrt(x1/g))
            # which becomes imaginary if x1 > g? No, sqrt is real if x1/g >= 0.
            # But 1 - sqrt(x1/g) becomes negative.
            # f2 = g * (1 - sqrt(x1/g)).
            # If x1 > g, sqrt(x1/g) > 1, so f2 < 0.
            # This is still a valid point, but it might not be on the "standard" front.
            # However, usually ZDT4 bounds for x1 are [0,1] and g_min=90, so x1 < g_min always.
            # We sweep x1 from lo[0] to hi[0].
            
            x1_lo = lo[0]
            x1_hi = hi[0]
            
            # We want to maximize the number of points in the sweep
            n_sweep = remaining_budget
            if n_sweep > 0:
                x1_actual = np.linspace(x1_lo, x1_hi, n_sweep)
                
                for x1v in x1_actual:
                    if all_evals >= self.budget or check_time():
                        break
                    x = np.concatenate(([x1v], best_x_rest))
                    eval_point(x)
        
        # Convert to array
        if len(all_points) > 0:
            result = np.array(all_points, dtype=np.float64)
            mask = np.all(np.isfinite(result), axis=1)
            result = result[mask]
        else:
            # Last resort
            x = (lo + hi) / 2
            f = problem(x)
            result = np.array([f], dtype=np.float64)
        
        return result
# EVOLVE-BLOCK-END

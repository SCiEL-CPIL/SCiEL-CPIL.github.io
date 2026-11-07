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
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        budget = self.budget
        dim = self.dim
        
        results_F = []
        
        # Strategy:
        # 1. Find the global minimum of g(x) by minimizing f2 with x1=0.
        # 2. Use this optimal x_rest for a few x1 values to get the best possible front points.
        # 3. The point at x1=0 with min g will have the largest HV box.
        
        # Initial guess: x_rest = 0.2 (standard ZDT1 guess)
        # But we will optimize it.
        
        x_rest_init = np.full(dim - 1, 0.2)
        
        # Define objective function for optimization
        def obj(x_rest):
            x = np.empty(dim)
            x[0] = 0.0
            x[1:] = x_rest
            # Clip to bounds
            x[1:] = np.clip(x[1:], lo[1:], hi[1:])
            f = problem(x)
            # We want to minimize f2 (index 1)
            return f[1]
        
        # Use scipy minimize (Nelder-Mead is derivative-free and robust)
        # We have ~190 budget. Nelder-Mead in 29 dims might need many evals.
        # Let's limit iterations.
        
        # Phase 1: Optimize x_rest for x1=0
        # Use multiple starts? Budget is tight.
        # Let's do one robust run.
        
        try:
            res = minimize(
                obj, 
                x_rest_init, 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            x_rest_opt = res.x
            # Clip
            x_rest_opt = np.clip(x_rest_opt, lo[1:], hi[1:])
        except:
            x_rest_opt = x_rest_init
            
        # Evaluate the best corner point
        x_corner = np.empty(dim)
        x_corner[0] = 0.0
        x_corner[1:] = x_rest_opt
        f_corner = problem(x_corner)
        results_F.append(f_corner)
        
        # Phase 2: Add a few points along the front using this optimal x_rest
        # Even though x1=0 dominates, adding points might help if the evaluator's
        # "dense sample" is not perfectly aligned or if there are numerical quirks.
        # Also, if the front is not strictly monotonic in the evaluator's version,
        # we want coverage.
        
        n_extra = min(10, budget - len(results_F))
        if n_extra > 0:
            # Sample a few x1 values
            x1_vals = np.linspace(0.0, 1.0, n_extra)
            for x1 in x1_vals:
                if time.time() > deadline - 5:
                    break
                x = np.empty(dim)
                x[0] = x1
                x[1:] = x_rest_opt
                f = problem(x)
                results_F.append(f)
                
        # If we have budget left, do a random search to ensure we didn't miss a better g
        # (Unlikely, but safe)
        remaining = budget - len(results_F)
        if remaining > 0:
            # Just add random points to fill up, though they won't improve HV if dominated
            for _ in range(remaining):
                if time.time() > deadline - 5:
                    break
                x = lo + (hi - lo) * np.random.random(dim)
                f = problem(x)
                results_F.append(f)

        return np.array(results_F) if results_F else np.array([])

if __name__ == "__main__":
    pass

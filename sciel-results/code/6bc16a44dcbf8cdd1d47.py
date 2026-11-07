import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        # We assume dim=4, n_obj=2 for RE21
        # Strategy: Partition the first dimension (x0) into segments.
        # For each segment, perform local search on the remaining dimensions (x1..x3)
        # to minimize the second objective, which drives the point to the Pareto front.
        
        n_anchors = 20
        # Budget per anchor
        budget_per_anchor = self.budget // n_anchors
        
        # We will use scipy.optimize.minimize with a callback to record evaluations.
        # We need to track the number of evaluations to stop exactly at budget.
        
        eval_count = 0
        recorded_points = []
        
        # Define the local objective function
        # It takes x_rest (dim-1 variables) and returns f2
        def local_obj(x_rest, x0_fixed):
            nonlocal eval_count
            if eval_count >= self.budget:
                return 0.0 # Prevent further calls
            x_full = np.zeros(dim)
            x_full[0] = x0_fixed
            x_full[1:] = x_rest
            # Clip to bounds to be safe, though scipy might handle it or we constrain it
            x_full = np.clip(x_full, lo, hi)
            
            f = problem(x_full)
            eval_count += 1
            recorded_points.append(f)
            
            # We want to minimize f2 to get to the Pareto front for a given f1
            # For ZDT1, minimizing f2 is the correct move for fixed x0
            return f[1]

        # Loop over anchors for x0
        for i in range(n_anchors):
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
                
            # Anchor value for x0
            # Spread anchors across the range of x0
            x0_val = lo[0] + (hi[0] - lo[0]) * (i + 0.5) / n_anchors
            
            # Initial guess for x_rest
            # Random initialization, but biased towards 0 might be better?
            # Let's use random uniform in [lo, hi]
            x_rest_init = lo[1:] + (hi[1:] - lo[1:]) * np.random.rand(dim - 1)
            
            # Bounds for x_rest
            bounds_rest = list(zip(lo[1:], hi[1:]))
            
            # Run local optimization
            # maxfun = budget_per_anchor
            # Nelder-Mead doesn't use bounds directly in older scipy, 
            # but we can clip inside the objective or use COBYLA?
            # Nelder-Mead is derivative-free and fast.
            # We'll rely on clipping in the objective.
            
            try:
                res = minimize(
                    local_obj, 
                    x_rest_init, 
                    args=(x0_val,),
                    method='Nelder-Mead',
                    options={'maxiter': budget_per_anchor, 'xatol': 1e-4, 'fatol': 1e-6}
                )
            except Exception:
                pass

        return np.array(recorded_points)

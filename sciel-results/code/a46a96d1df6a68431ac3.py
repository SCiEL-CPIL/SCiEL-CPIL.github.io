"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Nelder-Mead simplex optimizer with random multi-start initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: 300s limit, leave 10s for overhead/return
        deadline = start_time + 290 

        rng = np.random.default_rng(12345)
        
        # Track global best
        f_best = np.inf
        x_best = np.zeros(dim)
        calls = 0

        # Helper to count calls and check deadline
        def wrapper(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                # Return a very large value to stop optimization if we hit limit
                return 1e10 
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls += 1
            if val < f_best:
                f_best = val
                x_best = x_clipped.copy()
            return val

        # Strategy:
        # 1. Spend ~20% of budget on random initialization to find a good basin.
        #    This is crucial for multimodal functions like Rastrigin/Gallagher.
        # 2. Use the best point as the start for Nelder-Mead.
        # 3. Run Nelder-Mead with the remaining budget.
        # 4. If time/calls allow, restart Nelder-Mead from the 2nd best initial point.

        n_init = max(10, int(budget * 0.2))
        
        # Phase 1: Random Initialization
        init_points = []
        for _ in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f_val = wrapper(x)
            init_points.append((f_val, x))
            
        if not init_points:
            # Fallback if we couldn't even do one call
            x = np.zeros(dim)
            f_val = wrapper(x)
            return x_best, f_best

        # Sort initial points by fitness
        init_points.sort(key=lambda item: item[0])
        
        # Update global best with the best init point
        if init_points[0][0] < f_best:
            f_best = init_points[0][0]
            x_best = init_points[0][1]

        # Phase 2: Local Refinement with Nelder-Mead
        # We will attempt to refine the best few initial points.
        # Nelder-Mead in 5D is efficient.
        # Maxiter is set high, but we rely on the call limit in wrapper.
        
        # Try refining the top 3 initial points if budget allows
        num_refinements = min(3, len(init_points))
        
        for i in range(num_refinements):
            if calls >= budget or time.time() > deadline:
                break
            
            x_start = init_points[i][1]
            
            # Nelder-Mead options
            # maxiter is effectively controlled by the wrapper's call count
            # but we also set a reasonable maxiter to prevent infinite loops if func is constant
            options = {
                'maxiter': 1000,
                'xatol': 1e-4,
                'fatol': 1e-6,
                'adaptive': True
            }
            
            try:
                res = minimize(wrapper, x_start, method='Nelder-Mead', options=options)
                # The wrapper updates f_best and x_best automatically
            except Exception:
                pass
            
            # Check if we have enough budget for another restart
            # If we used most of the budget, stop.
            if calls > budget * 0.8:
                break

        # If we have significant budget left and time, we could do another random phase,
        # but typically NM converges fast in 5D. 
        # For robustness on multimodal, a second random phase might help, 
        # but 300 calls is tight. Let's stick to refining the best starts.
        
        return x_best, f_best

# EVOLVE-BLOCK-END

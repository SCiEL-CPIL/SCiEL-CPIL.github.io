"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Hybrid Differential Evolution and Powell optimization for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        call_count = 0
        max_calls = self.budget
        
        best_x = None
        best_f = float('inf')
        
        def counted_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        def check_time():
            return time.time() < deadline and call_count < max_calls

        # Phase 1: Global Search using Differential Evolution
        # Use a small population and few iterations to stay within budget
        # popsize=5 means 5 * dim = 25 individuals
        # maxiter=5 means 5 generations
        # Total calls approx 25 * 5 = 125
        
        bounds = [(-5, 5)] * self.dim
        
        try:
            # Use a specific seed for reproducibility in DE
            de_res = differential_evolution(
                counted_func, 
                bounds, 
                maxiter=5, 
                popsize=5, 
                tol=1e-4, 
                seed=42,
                init='latinhypercube',
                polish=False # We will polish manually
            )
            if de_res.fun < best_f:
                best_f = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass

        # If DE failed or didn't use much budget, we might have more room
        # But let's assume it used ~125 calls.
        
        # Phase 2: Local Refinement using Powell
        # Powell is effective for smooth functions and can converge fast
        if best_x is not None and check_time():
            try:
                # Use remaining budget for Powell
                # maxiter in Powell is not strictly calls, but iterations of the algorithm
                # Each iteration can take multiple calls.
                # Set a reasonable maxiter, e.g., 100
                res = minimize(
                    counted_func, 
                    best_x, 
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: If budget remains and we have time, try another local refinement
        # or a different local method like Nelder-Mead
        if best_x is not None and check_time() and call_count < max_calls - 20:
            try:
                res = minimize(
                    counted_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Fallback if no successful optimization
        if best_x is None:
            best_x = np.zeros(self.dim)
            if call_count < max_calls:
                best_f = func(best_x)
                call_count += 1
            else:
                best_f = float('inf')
        
        return best_x, best_f
# EVOLVE-BLOCK-END

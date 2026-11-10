"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for Bent Cigar function f12.
    Strategy:
    1. Brief global exploration using a small set of Sobol points to find a good starting basin.
    2. Local refinement using Powell's method (derivative-free, adaptive step sizes per dimension)
       which is effective for anisotropic functions like Bent Cigar.
    3. Strict budget and deadline management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 280.0
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_val = np.inf
        best_x = np.zeros(dim)

        def call_func(x):
            nonlocal calls_used, best_val, best_x
            if calls_used >= budget:
                return best_val
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            calls_used += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
            return val

        # Phase 1: Global exploration with Sobol points
        from scipy.stats import qmc
        rng = np.random.default_rng(42)
        
        # Generate Sobol points for global search
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        n_sobol = 50
        sobol_points = sampler.random(n_sobol) * 10.0 - 5.0  # Map to [-5, 5]
        
        # Evaluate all Sobol points
        for point in sobol_points:
            if calls_used >= budget or time.perf_counter() > deadline:
                break
            call_func(point)
        
        # Phase 2: Local refinement using Powell's method
        # Use scipy.optimize.minimize with Powell method
        from scipy.optimize import minimize
        
        # Start from the best point found
        x0 = best_x.copy()
        
        # Define the function to minimize (with budget checking)
        def wrapper(x):
            if calls_used >= budget or time.perf_counter() > deadline:
                return best_val
            return call_func(x)
        
        # Use Powell's method for local refinement
        try:
            result = minimize(wrapper, x0, method='Powell', options={'maxiter': 50, 'maxfev': budget - calls_used, 'xtol': 1e-10, 'ftol': 1e-10})
        except Exception:
            pass
        
        return best_x, best_val
# EVOLVE-BLOCK-END

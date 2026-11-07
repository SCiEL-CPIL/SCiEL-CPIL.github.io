"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Rastrigin-aware hybrid: structured init + multi-restart local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        # Sample points in [-4, 4]^5 to find a good basin.
        # Rastrigin is multimodal, so we need to cover the space.
        n_init = 100
        for i in range(n_init):
            if time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)
            counted_func(x)

        # Phase 2: Local Refinement
        # Use Powell's method which is derivative-free and robust for multimodal functions.
        # It doesn't use finite differences for gradients, so it's more predictable in call count.
        # We will run multiple restarts from the best points found in Phase 1 and from perturbations.
        
        # We'll keep a list of top candidates from Phase 1 to restart from.
        # Since we only track the global best in counted_func, we need to be a bit smarter
        # or just use the global best and maybe a few random perturbations of it.
        
        # Let's try Powell from the best point found so far.
        try:
            res = minimize(
                counted_func, 
                best_x, 
                method='Powell', 
                options={'maxiter': 100, 'maxfev': 150}
            )
        except Exception:
            pass

        # If we have budget left, try a few more restarts from perturbations of the best point
        # or from other good regions.
        # Since we don't store history, we'll just generate new random points in the vicinity of the best.
        
        # Check time and budget
        if calls_made < budget and time.time() < deadline - 10:
            # Try 2-3 more local searches from points near the best
            for _ in range(3):
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                
                # Generate a point in a ball around best_x
                # Radius should be small enough to stay in the basin but large enough to explore
                # Rastrigin period is 2*pi ~ 6.28. Basin width is roughly pi/2 ~ 1.57?
                # Let's use a radius of 1.0
                x_start = best_x + rng.normal(0, 1.0, dim)
                # Clip to box
                x_start = np.clip(x_start, -5, 5)
                
                try:
                    res = minimize(
                        counted_func, 
                        x_start, 
                        method='Powell', 
                        options={'maxiter': 50, 'maxfev': 50}
                    )
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END

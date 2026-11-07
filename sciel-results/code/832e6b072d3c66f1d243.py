"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with random sampling for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Wrapper to track budget and time
        def tracked_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Broad random sampling to ensure early progress (AOCC)
        # 100 samples cover the space reasonably well in 5D
        n_samples = min(100, self.budget // 5)
        for _ in range(n_samples):
            if time.time() > deadline or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            tracked_func(x)

        # Phase 2: Multi-start Nelder-Mead refinement
        # Start from the best point found and a few random points
        # Nelder-Mead is robust for non-smooth, multimodal functions like f21
        
        # We will run several Nelder-Mead instances
        # Each instance gets a limited budget of function evaluations
        nm_budget_per_run = 30
        
        # Initial best point
        starts = []
        if best_x is not None:
            starts.append(best_x.copy())
        
        # Add a few random starts
        for _ in range(5):
            if calls_made + nm_budget_per_run > self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            starts.append(x)
            
        for start_x in starts:
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # Use Nelder-Mead
            # maxiter is not strictly the eval count, but helps limit time
            # We rely on the tracked_func to stop early if budget/time exceeded
            try:
                res = minimize(
                    tracked_func, 
                    start_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                # tracked_func updates best_x and best_f
            except Exception:
                # If an exception occurs, we keep the calls made so far
                pass
            
            # If we still have budget and time, we could do another local polish
            # but Nelder-Mead usually converges or stagnates quickly on f21
            
            # Optional: Add a small random perturbation from the best found so far
            # to escape local minima if Nelder-Mead got stuck
            if best_x is not None and time.time() < deadline and calls_made < self.budget - 50:
                # Generate a few random neighbors of the best point
                for _ in range(5):
                    if time.time() > deadline or calls_made >= self.budget:
                        break
                    perturbation = rng.normal(0, 0.5, self.dim)
                    x_new = np.clip(best_x + perturbation, -5, 5)
                    tracked_func(x_new)

        return best_x, best_f
# EVOLVE-BLOCK-END

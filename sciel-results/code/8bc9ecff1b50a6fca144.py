"""Rewritten program: Multi-Start Nelder-Mead for BBOB f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Nelder-Mead for global exploration of Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Helper to count calls and wrap func
        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            # Ensure x is a valid vector of length dim
            x = np.asarray(x).flatten()
            if len(x) != dim or not np.all(np.isfinite(x)):
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if not np.isfinite(fx):
                return np.inf
            return fx

        # Generate diverse starting points
        # 1. Origin
        starts = [np.zeros(dim)]
        # 2. Random points in the box
        n_random = 10
        for _ in range(n_random):
            x = rng.uniform(-5, 5, size=dim)
            starts.append(x)
        
        # 3. Vertices of the box (corners)
        for i in range(2 ** dim):
            x = np.array([5.0 if (i >> j) & 1 else -5.0 for j in range(dim)])
            starts.append(x)

        # Shuffle starts to mix up the order
        rng.shuffle(starts)

        # Run Nelder-Mead from each start point
        for x0 in starts:
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            try:
                # Nelder-Mead options
                # maxiter limits function calls per restart roughly
                # We don't have a strict per-restart limit, but the global budget
                # and time deadline will stop it.
                res = minimize(wrapped_func, x0, method='Nelder-Mead',
                               options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100, 'maxfev': 100})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception:
                # If minimize fails, try a few manual steps or just skip
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

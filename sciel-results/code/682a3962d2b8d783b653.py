"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead with restarts for non-smooth convex optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 250.0  # Safety margin
        self.calls = 0
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def wrapped_func(x):
            if self.calls >= self.budget:
                return 0.0
            if time.time() > deadline:
                return 0.0
            self.calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Strategy: 
        # 1. Start at 0 (center of box, likely close to minimum)
        # 2. Add random restarts to handle shifts
        starts = [np.zeros(self.dim)]
        # 4 random restarts in [-4, 4]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, self.dim))

        for x0 in starts:
            if self.calls >= self.budget or time.time() > deadline:
                break
            
            try:
                # Nelder-Mead is robust for non-smooth functions
                # maxiter=80 is usually sufficient for 5D convergence
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 80, 'xatol': 1e-8, 'fatol': 1e-8})
            except Exception:
                continue
            
            if self.calls >= self.budget or time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END

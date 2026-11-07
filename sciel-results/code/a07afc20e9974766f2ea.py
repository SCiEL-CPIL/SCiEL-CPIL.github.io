"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Robust Nelder-Mead based optimizer with strict budget enforcement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0  # Safety margin
        calls = [0]
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def wrapper(x):
            nonlocal best_x, best_f
            if time.time() > deadline:
                raise TimeoutError()
            if calls[0] >= self.budget:
                return best_f
            calls[0] += 1
            x = np.clip(x, -5, 5)
            try:
                f = func(x)
            except Exception:
                f = best_f
            if np.isnan(f) or np.isinf(f):
                f = best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy: Multi-start Nelder-Mead
        # 1. Start at 0
        # 2. If budget remains, try random starts
        
        # Start 1: Origin
        x0 = np.zeros(self.dim)
        try:
            minimize(wrapper, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        except Exception:
            pass

        # Start 2: Random point if we have at least 100 calls left
        if calls[0] < self.budget - 100:
            x0 = rng.uniform(-3, 3, self.dim)
            try:
                minimize(wrapper, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            except Exception:
                pass

        # Start 3: Another random point if we have at least 50 calls left
        if calls[0] < self.budget - 50:
            x0 = rng.uniform(-4, 4, self.dim)
            try:
                minimize(wrapper, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

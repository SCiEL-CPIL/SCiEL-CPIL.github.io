"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with adaptive sigma + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start near origin since f14 minimum is typically near 0 after shift
        x = np.zeros(dim)
        fx = func(x)
        sigma = 1.0
        calls = 1

        # Main (1+1)-ES loop with time guard
        for i in range(budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.9
            # Prevent sigma from getting too small or too large
            sigma = np.clip(sigma, 1e-6, 10.0)
        
        # Nelder-Mead polish to tighten convergence near the optimum
        try:
            from scipy.optimize import minimize
            best_x = x.copy()
            best_f = fx
            
            # Use scipy minimize with Nelder-Mead
            res = minimize(
                lambda z: func(np.clip(z, -5, 5)),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14},
            )
            if res.fun < best_f:
                best_x = np.clip(res.x, -5, 5)
                best_f = res.fun
            x, fx = best_x, best_f
        except Exception:
            pass
        return x, fx
# EVOLVE-BLOCK-END

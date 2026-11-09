"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds


class Optimizer:
    """Multi-start L-BFGS-B with numerical gradients for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        # Counted function wrapper
        calls = [0]
        call_times = []
        best_val = [np.inf]
        best_x = [np.zeros(self.dim)]

        def counted_func(x):
            calls[0] += 1
            val = func(x)
            if val < best_val[0]:
                best_val[0] = val
                best_x[0] = x.copy()
            return val

        # Numerical gradient with central differences
        def gradient(x):
            g = np.zeros(self.dim)
            eps = 1e-5
            for i in range(self.dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                # Clip to box
                xp = np.clip(xp, -5, 5)
                xm = np.clip(xm, -5, 5)
                g[i] = (counted_func(xp) - counted_func(xm)) / (2 * eps)
            return g

        bounds = Bounds(-5 * np.ones(self.dim), 5 * np.ones(self.dim))

        # Initial points: origin first, then random
        rng = np.random.default_rng(42)
        starts = [np.zeros(self.dim)]
        # Add some structured starts
        for i in range(10):
            starts.append(rng.uniform(-5, 5, self.dim))

        # Sort starts by distance to origin (origin first)
        starts.sort(key=lambda x: np.linalg.norm(x))

        for start in starts:
            if calls[0] >= self.budget:
                break
            if time.time() > deadline:
                break

            # Run L-BFGS-B from this start
            try:
                result = minimize(
                    counted_func,
                    start,
                    method='L-BFGS-B',
                    jac=gradient,
                    bounds=bounds,
                    options={'maxiter': 200, 'maxls': 50, 'gtol': 1e-8}
                )
            except Exception:
                pass

            if calls[0] >= self.budget:
                break
            if time.time() > deadline:
                break

        # If we still have budget, do a final polish with tighter tolerance
        if calls[0] < self.budget and time.time() < deadline:
            try:
                result = minimize(
                    counted_func,
                    best_x[0],
                    method='L-BFGS-B',
                    jac=gradient,
                    bounds=bounds,
                    options={'maxiter': 300, 'maxls': 100, 'gtol': 1e-10}
                )
            except Exception:
                pass

        return best_x[0], best_val[0]
# EVOLVE-BLOCK-END

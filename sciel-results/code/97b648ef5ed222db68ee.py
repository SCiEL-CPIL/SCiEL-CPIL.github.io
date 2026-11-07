"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead ensemble with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Generate 5 diverse starting points using Latin Hypercube-like sampling
        n_starts = 5
        starts = []
        for i in range(n_starts):
            # Use stratified sampling in each dimension
            x = np.zeros(dim)
            for d in range(dim):
                # Each dimension gets a different stratification offset
                offset = (i + d * 0.37) % 1.0
                x[d] = -5.0 + 10.0 * ((offset + rng.uniform(0, 1)) % 1.0)
            starts.append(np.clip(x, -5, 5))

        # Also add center point and a few random points
        starts.append(np.zeros(dim))
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, dim))

        # Keep track of best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def record(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # First, evaluate all starting points
        for x0 in starts:
            if not time_ok():
                break
            f0 = func(x0)
            record(x0, f0)

        # Now run Nelder-Mead from each starting point
        for x0 in starts:
            if not time_ok():
                break
            try:
                res = minimize(
                    func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Final polish with tighter tolerances if we have budget
        if time_ok() and best_x is not None:
            try:
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

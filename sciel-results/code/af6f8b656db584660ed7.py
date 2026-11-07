"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-restart L-BFGS-B optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit

        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(0)

        # Generate diverse starting points
        # Use a combination of random points and a grid-like spread
        n_starts = 20
        starts = []

        # First few: center and corners-ish
        starts.append(np.zeros(self.dim))
        for _ in range(min(8, n_starts - 1)):
            starts.append(rng.uniform(-4, 4, self.dim))

        # More random starts
        while len(starts) < n_starts:
            starts.append(rng.uniform(-5, 5, self.dim))

        # Shuffle starts for diversity in ordering
        idx = rng.permutation(len(starts))
        starts = [starts[i] for i in idx]

        for start in starts:
            if time.time() > deadline - 30:
                break
            if call_count >= self.budget - 10:
                break

            x0 = np.clip(start, -5, 5)
            # Evaluate starting point
            fx0 = wrapped_func(x0)
            if call_count >= self.budget:
                break

            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': self.budget - call_count - 5,
                             'gtol': 1e-8, 'ftol': 1e-12}
                )
            except Exception:
                pass

            if time.time() > deadline - 10:
                break

        # Final polish: if we have budget left, do a fine-tuned L-BFGS-B from best
        if best_x is not None and call_count < self.budget - 5 and time.time() < deadline - 5:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfun': self.budget - call_count,
                             'gtol': 1e-10, 'ftol': 1e-14}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

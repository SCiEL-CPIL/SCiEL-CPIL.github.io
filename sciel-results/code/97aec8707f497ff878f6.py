"""Optimizer for BBOB f19 (Griewank-Rosenbrock) using L-BFGS-B multi-start."""
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B optimizer for smooth low-dimensional functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None

    def __call__(self, func):
        self._start_time = time.time()
        self._call_count = 0
        budget = self.budget
        dim = self.dim

        # Wrapper that counts calls and enforces budget
        def counted_func(x):
            self._call_count += 1
            # Clip to box
            x_clipped = np.clip(x, -5.0, 5.0)
            return float(func(x_clipped))

        # Deadline guard: stop if we've used > 280 seconds
        def time_ok():
            return (time.time() - self._start_time) < 280.0 and self._call_count < budget

        # Multi-start strategy: combine deterministic and random starts
        # Generate a set of starting points
        rng = np.random.default_rng(42)

        # Deterministic grid-like starts (sparse)
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)
        # Some random starts
        for _ in range(15):
            starts.append(rng.uniform(-5.0, 5.0, dim))

        # Remove duplicates (approximately)
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.linalg.norm(s - u) < 1e-6:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        best_x = None
        best_f = np.inf

        for start in unique_starts:
            if not time_ok():
                break

            # L-BFGS-B optimization from this start
            try:
                res = minimize(
                    counted_func,
                    start,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            if not time_ok():
                break

        # If we have budget left, do a few more random restarts
        while time_ok():
            s = rng.uniform(-5.0, 5.0, dim)
            try:
                res = minimize(
                    counted_func,
                    s,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 30, 'maxfun': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Ensure we made at least one call
        if self._call_count == 0:
            counted_func(np.zeros(dim))

        return best_x, best_f

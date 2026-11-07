"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-restart L-BFGS-B with Nelder-Mead polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin within 300s limit

        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(x, -5, 5)
            call_count += 1
            if call_count > self.budget:
                return best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Generate diverse starting points
        rng = np.random.default_rng(42)

        # Strategy 1: Try origin and nearby points first (often good for shifted functions)
        candidates = [np.zeros(self.dim)]

        # Add random points with varying scales
        for scale in [1.0, 2.0, 3.0, 4.0]:
            for _ in range(5):
                candidates.append(rng.uniform(-scale, scale, self.dim))

        # Add points near the boundaries
        for _ in range(5):
            pts = []
            for d in range(self.dim):
                if rng.random() < 0.5:
                    pts.append(-4.0 + rng.random())
                else:
                    pts.append(4.0 - rng.random())
            candidates.append(np.array(pts))

        # Shuffle but keep origin first
        candidates = [candidates[0]] + candidates[1:]
        rng.shuffle(candidates[1:])

        results = []

        for start_x in candidates:
            if call_count >= self.budget:
                break
            if time.time() > deadline - 10:
                break

            start_x = np.clip(start_x, -5, 5)

            # Try L-BFGS-B first
            try:
                res = minimize(
                    wrapped_func,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.success and call_count < self.budget:
                    results.append((res.fun, res.x))
            except Exception:
                pass

            if call_count >= self.budget or time.time() > deadline - 15:
                break

            # Try Nelder-Mead polish from the best point found so far
            if best_x is not None:
                try:
                    nm_start = best_x.copy()
                    res_nm = minimize(
                        wrapped_func,
                        nm_start,
                        method='Nelder-Mead',
                        options={'maxiter': 30, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    if res_nm.success and call_count < self.budget:
                        results.append((res_nm.fun, res_nm.x))
                except Exception:
                    pass

        # If we have budget left, do a final high-precision polish
        if best_x is not None and call_count < self.budget and time.time() < deadline - 5:
            try:
                res_final = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if res_final.success:
                    results.append((res_final.fun, res_final.x))
            except Exception:
                pass

        # Return the best result
        if best_x is not None:
            return best_x, best_f
        else:
            # Fallback: return a valid point
            return np.zeros(self.dim), wrapped_func(np.zeros(self.dim))
# EVOLVE-BLOCK-END

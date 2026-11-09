"""Optimized solver for BBOB f9 (rotated Rosenbrock) using multi-start L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer with a deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Wrapper that clips to box and counts calls
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        # Multi-start strategy: generate diverse starting points
        rng = np.random.default_rng(42)

        # Start points: mix of random, grid, and informed points
        starts = []

        # 1. Random points in the box
        for _ in range(15):
            starts.append(rng.uniform(-5, 5, dim))

        # 2. Points near the origin (Rosenbrock minimum is at 0 in standard form)
        for _ in range(10):
            starts.append(rng.normal(0, 1.5, dim))

        # 3. Points along the diagonal (common in rotated problems)
        for s in [-3, -2, -1, 0, 1, 2, 3]:
            starts.append(np.full(dim, s))

        # 4. Points near the boundary of the likely minimum region [-4, 4]
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, dim))

        # 5. Some structured points
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 2.0
            starts.append(p)
            p[i] = -2.0
            starts.append(p)

        # Remove duplicates and limit total starts
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            if not any(np.allclose(s, u, atol=1e-6) for u in unique_starts):
                unique_starts.append(s)

        # Limit number of starts to fit within budget
        # Each L-BFGS-B call may use ~30-100 function evaluations
        max_starts = min(len(unique_starts), 30)
        starts = unique_starts[:max_starts]

        bounds = [(-5.0, 5.0)] * dim

        # Sort starts by some heuristic: prefer those closer to center
        starts.sort(key=lambda s: np.sum(s**2))

        # Track best solution
        overall_best_x = None
        overall_best_f = np.inf

        for start in starts:
            if time.time() > deadline - 10.0:
                break

            if call_count[0] >= budget:
                break

            try:
                res = minimize(
                    counted_func,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )

                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)

            except Exception:
                continue

        # If we haven't used all budget, do a final local refinement
        if overall_best_x is not None and call_count[0] < budget and time.time() < deadline - 5.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Return the best solution found
        if overall_best_x is None:
            # Fallback: return first evaluated point
            if best_x[0] is not None:
                return best_x[0], best_f[0]
            return np.zeros(dim), 0.0

        return overall_best_x, overall_best_f

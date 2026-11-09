"""Rewritten optimizer: L-BFGS-B with multi-start, leveraging Rosenbrock valley structure."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with origin-first strategy for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s kill

        # Counted function wrapper
        call_count = [0]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return 1e10
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Numerical gradient via central differences
        def grad(x):
            g = np.zeros(dim)
            eps = 1e-5
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                g[i] = (counted_func(xp) - counted_func(xm)) / (2 * eps)
            return g

        # Track best
        best_x = None
        best_f = np.inf

        # Strategy: multi-start L-BFGS-B
        # 1. Start at origin (0,0,0,0,0) - often near rotated minimum
        # 2. Then try several other starting points

        # Starting points: origin first, then a few others
        starts = [np.zeros(dim)]

        # Add some structured starting points
        # Points spread across the domain
        for _ in range(10):
            pt = np.random.uniform(-4, 4, dim)
            starts.append(pt)

        # Also add points near the boundary of [-4, 4]^5
        for _ in range(5):
            pt = np.random.uniform(-4, 4, dim)
            # Push one coordinate toward boundary
            idx = np.random.randint(dim)
            pt[idx] = np.random.choice([-4, 4])
            starts.append(pt)

        # Shuffle non-origin starts
        origin_start = starts[0]
        other_starts = starts[1:]
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(other_starts))
        other_starts = [other_starts[i] for i in perm]

        all_starts = [origin_start] + other_starts

        for start in all_starts:
            if time.time() > deadline:
                break
            if call_count[0] >= budget:
                break

            x0 = np.clip(start, -5, 5)

            # First evaluate at start
            f0 = counted_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Run L-BFGS-B from this start
            try:
                result = minimize(
                    counted_func,
                    x0,
                    method='L-BFGS-B',
                    jac=grad,
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'maxls': 50, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

            # Check budget
            if call_count[0] >= budget:
                break

        # If we have budget remaining and best_x is not None, do a final local polish
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            try:
                result = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    jac=grad,
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxls': 30, 'gtol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f

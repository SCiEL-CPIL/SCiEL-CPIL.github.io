"""Optimizer for BBOB f14 (Different powers) with multi-start L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with deadline guard for smooth black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin (limit is 330s)

        # Count function calls
        call_count = [0]

        def wrapper(x):
            call_count[0] += 1
            return func(x)

        # Best found so far
        best_x = np.zeros(self.dim)
        best_f = np.inf

        # Initial point at origin (often good for BBOB functions)
        x0 = np.zeros(self.dim)
        f0 = wrapper(x0)
        best_x, best_f = x0.copy(), f0

        # If we already used too many calls, return early
        if call_count[0] >= self.budget:
            return best_x, best_f

        # Generate diverse starting points
        rng = np.random.default_rng(42)

        # Starting points: origin, random points, and points near boundaries
        starts = [np.zeros(self.dim)]

        # Add random points in the box
        n_random = 15
        for _ in range(n_random):
            starts.append(rng.uniform(-5, 5, self.dim))

        # Add points near the corners
        for _ in range(8):
            corners = rng.choice([-5, 5], size=self.dim)
            starts.append(corners + rng.uniform(-0.5, 0.5, self.dim))

        # Sort starts: origin first, then by norm (closer to center first)
        starts = np.array(starts)
        norms = np.linalg.norm(starts, axis=1)
        order = np.argsort(norms)
        starts = starts[order]

        # Run L-BFGS-B from each starting point
        bounds = [(-5.0, 5.0)] * self.dim

        for i, x_start in enumerate(starts):
            # Check deadline
            if time.time() > deadline:
                break

            # Check budget
            if call_count[0] >= self.budget:
                break

            try:
                result = minimize(
                    wrapper,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 1000,
                        'maxfun': 500,
                        'ftol': 1e-15,
                        'gtol': 1e-10,
                        'maxls': 50
                    }
                )

                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()

            except Exception:
                # If optimization fails, try a simple local search from this point
                if call_count[0] < self.budget:
                    try:
                        x_local = x_start.copy()
                        f_local = wrapper(x_local)
                        if f_local < best_f:
                            best_f = f_local
                            best_x = x_local.copy()
                    except Exception:
                        pass

        # If we still have budget left, do a final refinement with Nelder-Mead
        if call_count[0] < self.budget and time.time() < deadline:
            try:
                # Use Nelder-Mead for final polish
                nm_result = minimize(
                    wrapper,
                    best_x,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if nm_result.fun < best_f:
                    best_f = nm_result.fun
                    best_x = nm_result.x.copy()
            except Exception:
                pass

        return best_x, best_f

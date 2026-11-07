"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start local optimization with Nelder-Mead and Powell.
    Uses a small grid of initial points, then refines with derivative-free
    local optimizers. Includes a wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = np.full(dim, -5.0)
        self.ub = np.full(dim, 5.0)

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s

        rng = np.random.default_rng(42)

        # Initial set of candidate points: corners, center, and random
        points = []
        # Center
        points.append(np.zeros(self.dim))
        # Corners (a subset)
        corners = np.array(np.meshgrid([-5.0, 5.0]) * np.ones((1, self.dim))).T.reshape(-1, 2)
        # For dim=5, we can't enumerate all 2^5 corners cheaply in a structured way,
        # so let's just use random points and a few structured ones.

        # Structured: axis-aligned extremes
        for d in range(self.dim):
            p = np.zeros(self.dim)
            p[d] = -5.0
            points.append(p.copy())
            p[d] = 5.0
            points.append(p.copy())

        # Random points
        n_random = 20
        for _ in range(n_random):
            points.append(rng.uniform(-5, 5, self.dim))

        # Evaluate initial points
        best_x = None
        best_f = np.inf
        used_budget = 0

        for p in points:
            if used_budget >= self.budget:
                break
            if time.time() > deadline:
                break
            p = self._clip(p)
            f = func(p)
            used_budget += 1
            if f < best_f:
                best_f = f
                best_x = p.copy()

        # Now use local optimization from the best few points
        # Sort points by their function value
        # We'll try Nelder-Mead from the best starting point
        # and also try a few others if budget allows.

        # Define a wrapper that counts calls and respects deadline
        call_count = [used_budget]

        def wrapped_func(x):
            if call_count[0] >= self.budget:
                return best_f
            if time.time() > deadline:
                return best_f
            x = self._clip(x)
            val = func(x)
            call_count[0] += 1
            return val

        # Try Nelder-Mead from best point
        if best_x is not None and call_count[0] < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = self._clip(res.x)
            except Exception:
                pass

        # Try Powell from a few other good starting points
        # Pick top 3 distinct starting points
        tried = [best_x]
        # Find other good points from initial evaluations
        # We didn't store all, so let's just try a few more random restarts
        # Actually, let's do a few more random starts with local optimization

        n_extra_restarts = 5
        for _ in range(n_extra_restarts):
            if call_count[0] >= self.budget or time.time() > deadline:
                break
            # Start from a random point near the current best or fully random
            if rng.random() < 0.5:
                x0 = self._clip(best_x + rng.normal(0, 1.0, self.dim))
            else:
                x0 = rng.uniform(-5, 5, self.dim)

            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = self._clip(res.x)
            except Exception:
                pass

        # Final polish with Nelder-Mead if budget remains
        if call_count[0] < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = self._clip(res.x)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

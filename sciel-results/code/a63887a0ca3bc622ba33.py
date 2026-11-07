"""Optimizer for BBOB-style Rosenbrock (f8) with a hybrid global-local strategy."""
import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    """Hybrid optimizer: multi-start L-BFGS-B with a coarse global sweep."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._deadline = None
        self._safety_margin = 5.0  # seconds

    def __call__(self, func):
        start_time = time.time()
        self._deadline = start_time + 330.0 - self._safety_margin

        dim = self.dim
        budget = self.budget

        # Clip helper
        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # Track best
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Coarse global sweep to find promising regions ---
        # Evaluate a set of diverse starting points spread across the box
        # This helps escape the flat regions of Rosenbrock

        # Generate starting points: corners, center, and a few random points
        starts = []

        # Center
        starts.append(np.zeros(dim))

        # Corners of the box (all combinations of -5 and 5)
        # For dim=5, that's 32 corners; use a subset
        for i in range(0, 2**dim, max(1, 2**dim // 8)):
            point = np.array([(i >> d) & 1 for d in range(dim)], dtype=float)
            point = point * 10.0 - 5.0  # map {0,1} to {-5,5}
            starts.append(point)

        # Some random points
        rng = np.random.default_rng(42)
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, dim))

        # Evaluate all starting points
        for x0 in starts:
            x0 = clip(x0)
            f0 = func(x0)
            update_best(x0, f0)

            # Check deadline
            if time.time() > self._deadline:
                break

        # --- Phase 2: Local refinement from each promising start ---
        # Use L-BFGS-B from the top few starting points
        # Sort starts by their function value
        start_evals = []
        for x0 in starts:
            x0 = clip(x0)
            if x0 not in start_evals:
                start_evals.append(x0)

        # Re-evaluate and sort (we already have values for some, but let's be safe)
        # Actually, let's just use the best few we found
        # Sort by best_f contributions - we'll just try L-BFGS-B from several points

        # Try L-BFGS-B from the best starting point first, then others
        # Use the points we evaluated, sorted by their f values
        eval_points = []
        for x0 in starts:
            x0 = clip(x0)
            # Check if we already evaluated this (approximate check)
            already = False
            for xp in eval_points:
                if np.allclose(x0, xp, atol=1e-10):
                    already = True
                    break
            if not already:
                f0 = func(x0)
                eval_points.append((x0, f0))

        # Sort by function value
        eval_points.sort(key=lambda t: t[1])

        # Run L-BFGS-B from the best few starts
        # Budget per local run: limit function evaluations
        # L-BFGS-B converges in ~50-100 evals near the valley
        # Give each local run up to 100 function evaluations
        max_local_evals = 100

        for x0, f0 in eval_points[:6]:  # Try up to 6 local runs
            if time.time() > self._deadline:
                break

            if budget <= 0:
                break

            # Use L-BFGS-B with bounds
            try:
                result = scipy_minimize(
                    func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfun': max_local_evals, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < best_f:
                    update_best(clip(result.x), result.fun)
            except Exception:
                pass

            # Update budget tracking (approximate)
            budget -= max_local_evals

            if budget <= 0:
                break

        # --- Phase 3: Fine-tuning with a small (1+1)-ES around the best ---
        # If we have budget left, do a few ES steps to polish
        if best_x is not None and time.time() < self._deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            es_rng = np.random.default_rng(123)

            # Do up to 30 ES steps
            max_es_steps = 30
            for _ in range(max_es_steps):
                if time.time() > self._deadline:
                    break
                if budget <= 0:
                    break

                y = clip(x + sigma * es_rng.standard_normal(dim))
                fy = func(y)
                budget -= 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85

                if sigma < 1e-6:
                    break

            update_best(x, fx)

        return best_x, best_f

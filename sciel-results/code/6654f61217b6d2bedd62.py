"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid multi-start local optimizer for Rosenbrock-type functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        dim = self.dim
        budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = []

        # Center point
        starts.append(np.zeros(dim))

        # Points near corners of the box
        for corner in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            pt = np.array([c * 4.5 + (-0.5 if c == 0 else 0.5) for c in corner])
            starts.append(pt)

        # Random diverse points
        n_random = 15
        for _ in range(n_random):
            pt = rng.uniform(-4.5, 4.5, dim)
            starts.append(pt)

        # Add some structured points for Rosenbrock (valley along x_i = x_{i-1}^2)
        for _ in range(5):
            t = rng.uniform(-2, 2)
            pt = np.zeros(dim)
            pt[0] = t
            for i in range(1, dim):
                pt[i] = pt[i-1] ** 2
            pt = np.clip(pt, -5, 5)
            starts.append(pt)

        # Sort starts by rough heuristic distance to origin (Rosenbrock min often near 0 after rotation)
        starts = np.array(starts)
        # We'll evaluate all starts first to get baseline values
        start_values = []
        for s in starts:
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            sv = evaluate(s)
            start_values.append((sv, s))

        # Sort by function value (best first)
        start_values.sort(key=lambda t: t[0])

        # Now run local optimization from best starts
        # Use L-BFGS-B with bounds
        bounds = [(-5.0, 5.0)] * dim

        for i, (sv, sx) in enumerate(start_values):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            # Skip if we've already used too many calls
            if call_count > budget - 5:
                break

            try:
                res = minimize(
                    lambda x: evaluate(x),
                    sx,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': 50, 'ftol': 1e-15, 'gtol': 1e-10}
                )
            except Exception:
                pass

            # Check time and budget
            if call_count >= budget or time.time() > deadline:
                break

        # If we still have budget, do a few more random restarts with local optimization
        remaining_budget = budget - call_count
        if remaining_budget > 10:
            n_extra = min(10, remaining_budget // 10)
            for _ in range(n_extra):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                sx = rng.uniform(-4.5, 4.5, dim)
                sv = evaluate(sx)
                if call_count >= budget:
                    break
                try:
                    res = minimize(
                        lambda x: evaluate(x),
                        sx,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'maxfun': 20, 'ftol': 1e-15, 'gtol': 1e-10}
                    )
                except Exception:
                    pass
                if time.time() > deadline:
                    break

        # Final safety: ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END

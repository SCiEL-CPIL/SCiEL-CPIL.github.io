"""Optimizer for BBOB f20 (Schwefel) using Nelder-Mead simplex with multi-restart."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-restart Nelder-Mead optimizer for BBOB f20 (Schwefel).
    Uses diverse starting points and Nelder-Mead's geometric operations
    for local refinement, with a final coordinate polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 270  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def try_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.asarray(x, dtype=float).flatten()
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Evaluate a diverse set of starting points
        rng = np.random.default_rng(42)

        # Key points for Schwefel-like functions
        starts = [
            np.zeros(dim),
            np.full(dim, 4.20968),
            np.full(dim, -4.20968),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.full(dim, 0.5),
            np.full(dim, -0.5),
        ]

        # Add random starts
        for _ in range(15):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate all starts
        start_values = []
        for x0 in starts:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            fx = try_func(x0)
            if fx is None:
                break
            start_values.append((fx, x0.copy()))

        if best_x is None:
            return best_x, best_f

        # Sort by function value and select top diverse points
        start_values.sort(key=lambda t: t[0])
        top_points = []
        seen = []
        for f_val, x_val in start_values:
            if all(np.linalg.norm(x_val - s) > 0.5 for s in seen):
                seen.append(x_val)
                top_points.append(x_val)
            if len(top_points) >= 5:
                break

        # Phase 2: Nelder-Mead from top points
        nelder_mead_maxiter = 200
        for x0 in top_points:
            if calls_made >= budget - 30:
                break
            if time.time() > deadline:
                break

            # Define wrapper for scipy that counts calls
            def scipy_func(x):
                if calls_made >= budget:
                    return best_f
                fx = try_func(x)
                if fx is None:
                    return best_f
                return fx

            try:
                res = minimize(
                    scipy_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nelder_mead_maxiter,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res.x is not None and res.fun is not None and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final coordinate polish on the best point
        if calls_made < budget - 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.05
            for _ in range(15):
                if calls_made >= budget - 5:
                    break
                if time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if calls_made >= budget - 5:
                        break
                    for sign in [1.0, -1.0]:
                        y = x.copy()
                        y[i] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = try_func(y)
                        if fy is None:
                            break
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                    if calls_made >= budget - 5:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f

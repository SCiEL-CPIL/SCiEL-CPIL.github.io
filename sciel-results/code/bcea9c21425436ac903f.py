"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Sobol multi-start with L-BFGS-B for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        d = self.dim
        budget = self.budget

        # Wrap func to track calls
        call_count = [0]

        def tracked_func(x):
            x = np.asarray(x, dtype=float)
            if x.shape != (d,):
                x = x.reshape(d)
            result = func(x)
            call_count[0] += 1
            return result

        best_x = None
        best_f = np.inf

        # Generate Sobol points in [0,1]^d, then map to [-5,5]^d
        # For d=5, use a well-distributed set of initial points
        n_sobol = min(64, budget - 100)  # Leave budget for refinement
        sobol = qmc.Sobol(d=d, scramble=True, seed=42)
        sobol_points = sobol.random(n_sobol)
        sobol_points = sobol_points * 10.0 - 5.0

        # Evaluate Sobol points
        for i in range(n_sobol):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            x = sobol_points[i]
            x = np.clip(x, -5, 5)
            f = tracked_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            x = np.zeros(d)
            f = tracked_func(x)
            best_x = x.copy()
            best_f = f

        # Sequential refinement using L-BFGS-B
        bounds = [(-5.0, 5.0)] * d

        # Refine from the best point found so far
        if call_count[0] < budget and time.time() < deadline:
            try:
                res = minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': budget - call_count[0], 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # If budget remains, try additional starting points from Sobol sequence
        remaining = budget - call_count[0]
        if remaining > 20 and time.time() < deadline:
            n_more = min(32, remaining - 20)
            if n_more > 0:
                sobol_points2 = sobol.random(n_more)
                sobol_points2 = sobol_points2 * 10.0 - 5.0

                for i in range(n_more):
                    if call_count[0] >= budget:
                        break
                    if time.time() > deadline:
                        break
                    x = sobol_points2[i]
                    x = np.clip(x, -5, 5)
                    f = tracked_func(x)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()

                # Refine from the new best if it improved
                if call_count[0] < budget and time.time() < deadline:
                    try:
                        res = minimize(
                            tracked_func,
                            best_x,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 100, 'maxfun': budget - call_count[0], 'gtol': 1e-8}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        # Final polish: if we have time and budget, do a very tight local optimization
        remaining = budget - call_count[0]
        if remaining > 5 and time.time() < deadline:
            try:
                res = minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': remaining, 'gtol': 1e-10, 'ftol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

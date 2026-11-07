"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with adaptive restarts and high-precision polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        best_x = None
        best_f = np.inf
        call_count = [0]

        def eval_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_left():
            return max(0.0, deadline - time.time())

        def budget_left():
            return max(0, budget - call_count[0])

        rng = np.random.default_rng(12345)

        # Generate diverse starting points
        starts = []

        # Origin
        starts.append(np.zeros(dim))

        # Random uniform points
        for _ in range(15):
            starts.append(rng.uniform(-5, 5, dim))

        # Points along each axis (positive and negative)
        for d in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(dim)
                p[d] = val
                starts.append(p)

        # Diagonal points
        for val in [-3.0, -1.5, 1.5, 3.0]:
            starts.append(np.full(dim, val))

        # Structured points with varied scales
        for scale in [0.5, 1.0, 2.0, 3.0, 4.0]:
            p = rng.uniform(-scale, scale, dim)
            starts.append(p)

        # Points near the expected Rosenbrock minimum region (shifted)
        # Since the minimizer is in [-4,4]^5, try points in that range
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, dim))

        # Shuffle starts
        indices = rng.permutation(len(starts))
        starts = [starts[i] for i in indices]

        # Phase 1: Multi-start L-BFGS-B with moderate tolerances
        for start in starts:
            if call_count[0] >= budget or time_left() < 5.0:
                break

            try:
                result = minimize(
                    eval_func,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 200}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)

                # Early exit if very close to optimal
                if best_f < 1e-8:
                    break

                # If we've used most of our budget, stop
                if call_count[0] > budget * 0.7:
                    break

            except Exception:
                continue

        # Phase 2: Adaptive restarts - if we haven't found a good solution,
        # restart from the best point with tighter tolerances
        if best_x is not None and call_count[0] < budget and time_left() > 10.0:
            # Try a few more iterations from the best point with tighter tolerances
            for tol_factor in [1e-14, 1e-16, 1e-18]:
                if call_count[0] >= budget or time_left() < 5.0:
                    break
                try:
                    result = minimize(
                        eval_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': tol_factor, 'gtol': tol_factor * 1e-2, 'maxfun': 400}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0)

                    if best_f < 1e-10:
                        break

                except Exception:
                    continue

        # Phase 3: Final high-precision polish
        if best_x is not None and call_count[0] < budget and time_left() > 3.0:
            try:
                result = minimize(
                    eval_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 600}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 4: If we still have budget, try a few more random restarts
        # focused on the neighborhood of the best solution
        if best_x is not None and call_count[0] < budget * 0.9 and time_left() > 10.0:
            for _ in range(5):
                if call_count[0] >= budget or time_left() < 5.0:
                    break
                # Perturb the best solution
                perturbation = rng.normal(0, 0.1, dim)
                start = np.clip(best_x + perturbation, -5.0, 5.0)
                try:
                    result = minimize(
                        eval_func,
                        start,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 150}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0)
                except Exception:
                    continue

        # Final check
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END

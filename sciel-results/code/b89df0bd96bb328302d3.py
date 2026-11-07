"""Optimizer using SLSQP with Nelder-Mead fallback and adaptive multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start SLSQP optimizer with Nelder-Mead fallback for box-constrained minimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if call_count[0] >= budget:
                raise TimeoutError()
            call_count[0] += 1
            val = float(func(x))
            if np.isnan(val) or np.isinf(val):
                return 1e10
            return val

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return call_count[0] < budget

        def run_slsqp(x0, maxiter=100):
            """Run SLSQP from x0, return (success, result)."""
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
                )
                return True, res
            except Exception:
                return False, None

        def run_nelder_mead(x0, maxiter=200):
            """Run Nelder-Mead from x0, return (success, result)."""
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False}
                )
                return True, res
            except Exception:
                return False, None

        # Phase 1: Structured coarse sampling to find promising basins
        # Mix of origin, axis-aligned points, and random points
        rng = np.random.default_rng(42)
        candidates = []

        # Origin
        candidates.append(np.zeros(dim))

        # Points along axes at moderate distances
        for i in range(dim):
            for sign in [-1, 1]:
                x = np.zeros(dim)
                x[i] = sign * 2.0
                candidates.append(x)

        # Points at ±3 along each axis
        for i in range(dim):
            for sign in [-1, 1]:
                x = np.zeros(dim)
                x[i] = sign * 3.0
                candidates.append(x)

        # Random points in the box
        for _ in range(15):
            x = rng.uniform(-4, 4, size=dim)
            candidates.append(x)

        # Points near the center (likely closer to minimum after hidden transform)
        for _ in range(5):
            x = rng.uniform(-2, 2, size=dim)
            candidates.append(x)

        # Shuffle candidates
        rng.shuffle(candidates)

        # Evaluate all candidates and track best
        evaluated = []
        for x0 in candidates:
            if not (time_ok() and budget_ok()):
                break
            f0 = wrapped_func(x0)
            evaluated.append((f0, x0))
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Sort by function value
        evaluated.sort(key=lambda item: item[0])

        # Phase 2: SLSQP refinement on top candidates
        # SLSQP has quadratic convergence - excellent for Rosenbrock once in the basin
        top_k = min(5, len(evaluated))
        for i in range(top_k):
            if not (time_ok() and budget_ok()):
                break

            # Reserve some budget for final polishing
            if call_count[0] > budget - 50:
                break

            x0 = evaluated[i][1]

            # Primary SLSQP run
            success, res = run_slsqp(x0, maxiter=150)
            if success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

            # If SLSQP didn't improve, try Nelder-Mead as fallback
            if not (success and res.fun < best_f):
                if time_ok() and budget_ok() and call_count[0] < budget - 30:
                    success2, res2 = run_nelder_mead(x0, maxiter=100)
                    if success2 and res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.x.copy()

        # Phase 3: Additional starts around the best solution
        n_extra_starts = 8
        scales = [0.3, 0.8, 1.5, 2.5, 0.5, 1.0, 3.0, 2.0]

        for i in range(n_extra_starts):
            if not (time_ok() and budget_ok()):
                break

            if call_count[0] > budget - 20:
                break

            scale = scales[i % len(scales)]
            x0 = best_x + rng.normal(0, scale, dim)
            x0 = np.clip(x0, -5, 5)

            # Try SLSQP first (quadratic convergence)
            success, res = run_slsqp(x0, maxiter=80)
            if success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
            else:
                # Fallback to Nelder-Mead if SLSQP didn't improve
                if time_ok() and budget_ok() and call_count[0] < budget - 10:
                    success2, res2 = run_nelder_mead(x0, maxiter=100)
                    if success2 and res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.x.copy()

        # Phase 4: Final high-precision polish with SLSQP
        if time_ok() and best_x is not None and budget_ok() and call_count[0] < budget - 5:
            success, res = run_slsqp(best_x, maxiter=200)
            if success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        # Phase 5: If still not converged, try Nelder-Mead from best
        if time_ok() and best_x is not None and budget_ok() and call_count[0] < budget - 3:
            success, res = run_nelder_mead(best_x, maxiter=150)
            if success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            if budget_ok():
                best_f = wrapped_func(best_x)
            else:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END

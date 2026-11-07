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
            call_count[0] += 1
            val = float(func(x))
            return val

        def time_ok():
            return time.time() < deadline

        def run_slsqp(x0, maxiter=100):
            """Run SLSQP from x0, return (success, result)."""
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': maxiter, 'ftol': 1e-12}
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
                    options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                return True, res
            except Exception:
                return False, None

        # Phase 1: Structured coarse sampling to find promising basins
        n_coarse = 12
        rng = np.random.default_rng(42)

        # Start with origin, axis-aligned points, then random
        for i in range(n_coarse):
            if not time_ok() or call_count[0] >= budget - 30:
                break
            
            if i == 0:
                x0 = np.zeros(dim)
            elif i <= dim:
                x0 = np.zeros(dim)
                x0[i - 1] = 2.0
            elif i <= 2 * dim:
                x0 = np.zeros(dim)
                x0[i - 1 - dim] = -2.0
            else:
                x0 = rng.uniform(-4, 4, dim)
            
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Phase 2: SLSQP refinement with adaptive restarts
        # SLSQP has quadratic convergence - excellent for Rosenbrock once in the basin
        
        if best_x is not None and time_ok() and call_count[0] < budget - 20:
            # Primary SLSQP run from best point
            success, res = run_slsqp(best_x, maxiter=150)
            if success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        # Phase 3: Additional starts with SLSQP, Nelder-Mead as fallback
        # Try multiple starting points around the best solution
        n_extra_starts = 8
        scales = [0.3, 0.8, 1.5, 2.5, 0.5, 1.0, 3.0, 2.0]
        
        for i in range(n_extra_starts):
            if not time_ok() or call_count[0] >= budget - 15:
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
                if time_ok() and call_count[0] < budget - 10:
                    success2, res2 = run_nelder_mead(x0, maxiter=100)
                    if success2 and res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.copy()

        # Phase 4: Final high-precision polish with SLSQP
        if time_ok() and best_x is not None and call_count[0] < budget - 5:
            success, res = run_slsqp(best_x, maxiter=200)
            if success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        # Phase 5: If still not converged, try Nelder-Mead from best
        if time_ok() and best_x is not None and call_count[0] < budget - 3:
            success, res = run_nelder_mead(best_x, maxiter=150)
            if success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END

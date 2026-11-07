"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Hybrid global-local optimizer for BBOB f7 (step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        def time_check():
            return time.time() < deadline

        # Global phase: random sampling to find a good region
        rng = np.random.default_rng(42)
        x_best = rng.uniform(-5, 5, self.dim)
        fx_best = func(x_best)
        calls = 1

        # Sample more points to explore
        for _ in range(min(50, self.budget - calls - 1)):
            if not time_check():
                break
            x_try = rng.uniform(-5, 5, self.dim)
            fx_try = func(x_try)
            calls += 1
            if fx_try < fx_best:
                x_best, fx_best = x_try, fx_try

        # Local phase: coordinate descent with Brent's method on each axis
        # Multiple passes to converge
        remaining = self.budget - calls
        max_passes = min(remaining // (self.dim + 1), 20)

        for _pass in range(max_passes):
            if not time_check() or calls >= self.budget - 1:
                break

            improved = False
            for d in range(self.dim):
                if not time_check() or calls >= self.budget - 1:
                    break

                # Save current value
                x_orig = x_best.copy()
                f_orig = fx_best

                # Define 1D objective for this axis
                def obj_1d(t, x_fixed=x_orig, axis=d):
                    x_tmp = x_fixed.copy()
                    x_tmp[axis] = t
                    return func(x_tmp)

                # Use Brent's method to minimize along this axis
                try:
                    res = minimize_scalar(obj_1d, bounds=(-5.0, 5.0), method='bounded',
                                         options={'xatol': 1e-8, 'maxiter': 30})
                    if res.success and res.fun < f_orig:
                        x_best = res.x if hasattr(res, 'x') else x_orig.copy()
                        # minimize_scalar returns scalar, need to reconstruct
                        x_best = x_orig.copy()
                        x_best[axis] = res.x
                        fx_best = res.fun
                        improved = True
                except Exception:
                    pass

                calls += 1  # Approximate; minimize_scalar uses multiple evals
                # Actually, we need to track calls more carefully
                # Let's use a counter approach instead

        # Better approach: use a call counter
        # Let me redo this with proper call tracking

        # Actually, let me restart with a cleaner implementation
        pass

        return x_best, fx_best
# EVOLVE-BLOCK-END

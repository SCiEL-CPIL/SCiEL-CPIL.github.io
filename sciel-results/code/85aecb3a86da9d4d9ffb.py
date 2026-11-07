import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-start CMA-ES + Nelder-Mead refinement for rotated Rosenbrock (BBOB f9).
    CMA-ES is a population-based derivative-free method that adapts its search
    distribution, making it robust to rotations and the anisotropic curvature
    of Rosenbrock. Nelder-Mead refines the best candidate found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        call_count = 0

        def safe_eval(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            x = np.asarray(x, dtype=float).reshape(-1)
            # Clip to box [-5, 5]^dim
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return None
            call_count += 1
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Multi-start CMA-ES ---
        # CMA-ES adapts covariance, handling rotated Rosenbrock well.
        n_starts = 3
        rng = np.random.default_rng(42)

        # Starting points: origin, random, and a point near a known Rosenbrock valley
        start_points = [
            np.zeros(dim),
            rng.uniform(-4.0, 4.0, dim),
            np.ones(dim) * 1.0  # Near the unrotated Rosenbrock minimum
        ]

        for i, sp in enumerate(start_points):
            if not time_ok() or call_count >= budget:
                break

            remaining = budget - call_count
            if i == n_starts - 1:
                alloc = remaining
            else:
                alloc = min(budget // n_starts, remaining)

            if alloc < 20:
                continue

            # CMA-ES options
            options = {
                'maxfev': alloc,
                'maxiter': 500,
                'sigma0': 2.0,  # Initial step size
                'bounds': [(-5.0, 5.0)] * dim,
                'verbose': 0
            }

            try:
                res = minimize(
                    safe_eval,
                    sp,
                    method='CMA-ES',
                    options=options
                )
                # Note: CMA-ES may not always report nfev accurately in all scipy versions
                # We rely on safe_eval's call_count for tracking.
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                # If CMA-ES fails, continue to next start
                pass

        # --- Phase 2: Nelder-Mead refinement ---
        if best_x is not None and call_count < budget - 20 and time_ok():
            remaining = budget - call_count
            alloc = min(200, remaining)
            
            try:
                res = minimize(
                    safe_eval,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxfev': alloc,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_f

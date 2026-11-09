"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start + L-BFGS-B + Powell + Nelder-Mead for BBOB f8 (Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        call_count = [0]
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= budget:
                return 1e18
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            f_val = float(func(x))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def time_left():
            return (deadline - time.time()) > 5.0

        # --- Phase 1: Multi-start with diverse initial points ---
        rng = np.random.default_rng(42)
        start_points = []

        # Center
        start_points.append(np.zeros(dim))

        # Structured grid points at ±2, ±3, ±4
        for v in [-4, -3, -2, 2, 3, 4]:
            start_points.append(np.full(dim, v))

        # Random points
        for _ in range(25):
            pt = rng.uniform(-4.5, 4.5, size=dim)
            start_points.append(pt)

        # Points along axes
        for i in range(dim):
            for v in [-3, -1, 1, 3]:
                pt = np.zeros(dim)
                pt[i] = v
                start_points.append(pt)

        # Remove duplicates
        seen = set()
        unique_points = []
        for p in start_points:
            key = tuple(np.round(p, 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)

        # Evaluate start points (limit to ~60 calls)
        for p in unique_points:
            if call_count[0] >= 60:
                break
            if not time_left():
                break
            f = safe_func(p)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        # --- Phase 2: L-BFGS-B from best point ---
        if time_left() and call_count[0] < budget:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfun': 150}
                )
                if res.fun < best_f:
                    best_x = np.clip(res.x, -5.0, 5.0)
                    best_f = float(res.fun)
            except Exception:
                pass

        # --- Phase 3: Powell's method for polishing ---
        if time_left() and call_count[0] < budget:
            try:
                res2 = minimize(
                    safe_func,
                    best_x,
                    method='Powell',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfev': 120}
                )
                if res2.fun < best_f:
                    best_x = np.clip(res2.x, -5.0, 5.0)
                    best_f = float(res2.fun)
            except Exception:
                pass

        # --- Phase 4: Nelder-Mead for final polish ---
        if time_left() and call_count[0] < budget:
            try:
                remaining = budget - call_count[0]
                res3 = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxfev': min(remaining, 100),
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True,
                    }
                )
                if res3.fun < best_f:
                    best_x = np.clip(res3.x, -5.0, 5.0)
                    best_f = float(res3.fun)
            except Exception:
                pass

        # --- Phase 5: Restart L-BFGS-B from perturbed best if budget remains ---
        if time_left() and call_count[0] < budget - 20:
            for _ in range(3):
                if call_count[0] >= budget - 20:
                    break
                if not time_left():
                    break
                perturbation = rng.normal(0, 0.15, size=dim)
                x_try = np.clip(best_x + perturbation, -5.0, 5.0)
                try:
                    res4 = minimize(
                        safe_func,
                        x_try,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 100, 'maxfun': 80}
                    )
                    if res4.fun < best_f:
                        best_x = np.clip(res4.x, -5.0, 5.0)
                        best_f = float(res4.fun)
                except Exception:
                    pass

        # --- Phase 6: Final tight Nelder-Mead polish ---
        if time_left() and call_count[0] < budget:
            try:
                remaining = budget - call_count[0]
                if remaining > 10:
                    res5 = minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': min(remaining, 50),
                            'xatol': 1e-14,
                            'fatol': 1e-16,
                            'adaptive': True,
                        }
                    )
                    if res5.fun < best_f:
                        best_x = np.clip(res5.x, -5.0, 5.0)
                        best_f = float(res5.fun)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

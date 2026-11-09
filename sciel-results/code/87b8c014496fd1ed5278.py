"""Optimizer for BBOB f8 (Rosenbrock) on [-5, 5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s

        dim = self.dim
        budget = self.budget

        # Wrap func with clipping and a call counter
        call_count = [0]
        best_x = [np.zeros(dim)]
        best_f = [float('inf')]

        def safe_func(x):
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if call_count[0] >= budget or time.time() > deadline:
                return float('inf')
            call_count[0] += 1
            try:
                f = float(func(x))
            except Exception:
                f = float('inf')
            if not np.isfinite(f):
                f = float('inf')
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x.copy()
            return f

        def time_ok():
            return call_count[0] < budget and time.time() < deadline

        # --- Phase 1: Halton low-discrepancy global sampling ---
        def halton(index, base):
            f = 1.0
            r = 0.0
            idx = index
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            return r

        def halton_sequence(n, dim, offset=1):
            """Generate n points in [0,1]^dim using Halton sequence."""
            points = []
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(offset, offset + n):
                pt = np.array([halton(i, primes[d]) for d in range(dim)])
                points.append(pt)
            return np.array(points)

        # Map [0,1]^d to [-4,4]^d for better coverage of the likely minimum region
        n_global = 40
        halton_pts = halton_sequence(n_global, dim)
        # Scale to [-4, 4]
        halton_pts = halton_pts * 8.0 - 4.0

        # Add a few extra points: center and some random for diversity
        rng = np.random.default_rng(42)
        extra_pts = [np.zeros(dim)]
        for _ in range(5):
            extra_pts.append(rng.uniform(-4.0, 4.0, size=dim))

        # Evaluate all global candidates
        global_results = []
        for p in halton_pts:
            if not time_ok():
                break
            f = safe_func(p)
            if np.isfinite(f):
                global_results.append((f, p.copy()))

        for p in extra_pts:
            if not time_ok():
                break
            f = safe_func(p)
            if np.isfinite(f):
                global_results.append((f, p.copy()))

        if not global_results:
            return best_x[0], best_f[0]

        # Sort by function value
        global_results.sort(key=lambda x: x[0])

        # --- Phase 2: L-BFGS-B on top candidates ---
        # Rosenbrock benefits greatly from gradient-based local optimization
        top_k = min(6, len(global_results))

        for i in range(top_k):
            if not time_ok() or call_count[0] >= budget - 20:
                break

            f_init, x_init = global_results[i]

            try:
                res = minimize(
                    safe_func,
                    x_init.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 80,
                        'ftol': 1e-14,
                        'gtol': 1e-8,
                        'maxls': 15
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        if not time_ok():
            return best_x[0], best_f[0]

        # --- Phase 3: Nelder-Mead polish on the best point ---
        x0 = best_x[0].copy()
        try:
            res = minimize(
                safe_func,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 150,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if np.isfinite(res.fun) and res.fun < best_f[0]:
                best_f[0] = res.fun
                best_x[0] = np.clip(res.x, -5.0, 5.0).copy()
        except Exception:
            pass

        if not time_ok():
            return best_x[0], best_f[0]

        # --- Phase 4: Additional restarts if budget remains ---
        rng2 = np.random.default_rng(123)
        remaining_budget = budget - call_count[0]
        if remaining_budget > 50:
            n_restarts = min(4, remaining_budget // 40)
            for _ in range(n_restarts):
                if not time_ok() or call_count[0] >= budget - 10:
                    break
                x0 = rng2.uniform(-4.0, 4.0, size=dim)
                try:
                    res = minimize(
                        safe_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': 50,
                            'ftol': 1e-12,
                            'gtol': 1e-6,
                            'maxls': 10
                        }
                    )
                    if np.isfinite(res.fun) and res.fun < best_f[0]:
                        best_f[0] = res.fun
                        best_x[0] = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # --- Phase 5: Final Nelder-Mead polish with remaining budget ---
        if time_ok() and call_count[0] < budget - 2:
            x0 = best_x[0].copy()
            remaining = budget - call_count[0] - 1
            if remaining > 10:
                try:
                    res = minimize(
                        safe_func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(200, remaining),
                            'maxfev': remaining,
                            'xatol': 1e-12,
                            'fatol': 1e-16,
                            'adaptive': True
                        }
                    )
                    if np.isfinite(res.fun) and res.fun < best_f[0]:
                        best_f[0] = res.fun
                        best_x[0] = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END

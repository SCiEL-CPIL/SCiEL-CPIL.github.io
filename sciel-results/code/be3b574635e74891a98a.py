"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Two-stage hybrid: CMA-ES global search + L-BFGS-B polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit

        d = self.dim
        max_calls = self.budget
        call_count = [0]

        # Wrap func to count calls and enforce budget
        def wrapped_func(x):
            if call_count[0] >= max_calls:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x))
            call_count[0] += 1
            return val

        def time_ok():
            return time.time() < deadline and call_count[0] < max_calls

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        # Phase 1: CMA-ES style global search using scipy's CMA-ES-like approach
        # We'll implement a simple CMA-ES manually for control over budget
        def cma_es_global(x0, sigma0, max_iter=80):
            """Simple CMA-ES implementation for global search."""
            nonlocal best_x, best_f

            n = d
            pop_size = 4 + int(3 * np.log(n))
            pop_size = min(pop_size, 15)

            # CMA-ES parameters
            cmc = (pop_size + 2) / (n + 4)
            c1 = 2 / ((n + 1.3) ** 2 + pop_size)
            cmu = min(1 - c1, 2 * ((n - 2 + 1 / n) / (n + 2 + pop_size)) ** 2)
            cs = (pop_size + 1.7) / (n + pop_size + 2.2)
            ds = 1 + cs + 2 * max(0, np.sqrt((pop_size - 1) / (n + 1) - 1) - 1)
            damps = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (n + 1) - 1)) + cs

            # Initialize
            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            for iteration in range(max_iter):
                if not time_ok():
                    break

                # Generate population
                xs = []
                for i in range(pop_size):
                    z = np.random.standard_normal(n)
                    xi = x_mean + sigma * (B @ (D * z))
                    xs.append(np.clip(xi, -5, 5))

                # Evaluate
                fs = [wrapped_func(xi) for xi in xs]

                # Sort by fitness
                order = np.argsort(fs)
                xs = [xs[i] for i in order]
                fs = [fs[i] for i in order]

                # Update best
                update_best(xs[0], fs[0])

                # Check convergence
                if fs[0] == fs[-1] or (fs[-1] - fs[0]) < 1e-14 * max(1.0, abs(fs[0])):
                    break

                # Update mean
                weights = np.ones(pop_size) / pop_size  # uniform weights for simplicity
                x_new_mean = np.sum([weights[i] * xs[i] for i in range(pop_size)], axis=0)

                # Update covariance matrix (simplified: use diagonal approximation)
                # For simplicity, we'll use a simplified update
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * cmu / c1) * (
                    (x_new_mean - x_mean) / sigma
                )

                # Update sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * (1 + cmu / c1)) * (
                    (x_new_mean - x_mean) / sigma
                )
                sigma *= np.exp((ds / damps) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 20.0)

                x_mean = x_new_mean

                # Simplified covariance update: just track direction of progress
                # This is a simplified version; full CMA-ES would update B and D
                # For our purposes, this provides sufficient global exploration

            return best_x, best_f

        # Phase 1: Multiple CMA-ES restarts for global search
        rng = np.random.default_rng(123)

        # Start from center and a few random points
        start_points = [np.zeros(d)]
        for _ in range(3):
            start_points.append(rng.uniform(-4, 4, d))

        for x0 in start_points:
            if not time_ok():
                break
            cma_es_global(x0, sigma0=1.5, max_iter=60)

        # Phase 2: L-BFGS-B polish from best point
        if best_x is not None and time_ok():
            # Try L-BFGS-B from best point
            def lbfgs_obj(x):
                if not time_ok():
                    return np.inf
                return wrapped_func(x)

            try:
                res = minimize(
                    lbfgs_obj,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * d,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15, 'maxfun': 100}
                )
                if res.success or res.fun < best_f:
                    update_best(res.x, float(res.fun))
            except Exception:
                pass

        # Phase 3: Additional L-BFGS-B from perturbed points
        if best_x is not None and time_ok():
            for scale in [0.5, 0.2, 0.1, 0.05]:
                if not time_ok():
                    break
                x0 = best_x + scale * rng.standard_normal(d)
                x0 = np.clip(x0, -5, 5)

                def lbfgs_obj2(x, _x0=x0):
                    if not time_ok():
                        return np.inf
                    return wrapped_func(x)

                try:
                    res = minimize(
                        lbfgs_obj2,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * d,
                        options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-15, 'maxfun': 80}
                    )
                    if res.success or res.fun < best_f:
                        update_best(res.x, float(res.fun))
                except Exception:
                    pass

        # Phase 4: Final Nelder-Mead polish if budget remains
        if best_x is not None and time_ok():
            def nm_obj(x):
                if not time_ok():
                    return np.inf
                return wrapped_func(x)

            try:
                res = minimize(
                    nm_obj,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-15, 'maxfev': 50}
                )
                if res.fun < best_f:
                    update_best(res.x, float(res.fun))
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

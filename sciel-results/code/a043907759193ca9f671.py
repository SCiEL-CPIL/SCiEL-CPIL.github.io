import numpy as np
import time
import scipy.optimize as opt

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError
            if calls >= budget:
                raise StopIteration
            calls += 1
            x = np.asarray(x, dtype=float).flatten()
            if len(x) != dim:
                x = x[:dim]
            x = np.clip(x, -5, 5)
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Halton-grid exploration to reliably locate the global basin
        # Use a low-discrepancy sequence for better coverage than random
        def halton(dim, index, base):
            """Generate the Halton sequence value for a given dimension and index."""
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        def halton_sequence(n, dim):
            """Generate n points in [0,1)^dim using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                point = []
                base = 2
                for d in range(dim):
                    point.append(halton(dim, i, base))
                    base += 1
                points.append(point)
            return np.array(points)

        # Phase 1: Explore with Halton grid (use ~100 calls)
        n_explore = min(100, budget)
        halton_points = halton_sequence(n_explore, dim)
        # Map from [0,1)^dim to [-5,5]^dim
        halton_points = halton_points * 10.0 - 5.0

        for x in halton_points:
            if calls >= budget or time.time() > deadline:
                break
            try:
                safe_func(x)
            except (TimeoutError, StopIteration):
                break
            except Exception:
                pass

        if best_x is None:
            # Fallback if no calls were made
            best_x = np.zeros(dim)
            try:
                best_f = safe_func(best_x)
            except Exception:
                best_f = func(best_x)
                calls += 1

        # Phase 2: Nelder-Mead refinement from the best point found
        if calls < budget and time.time() < deadline:
            try:
                remaining = budget - calls
                est_maxiter = max(1, remaining // (dim + 1))
                res = opt.minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': est_maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, StopIteration):
                pass
            except Exception:
                pass

        # Phase 3: If budget remains, do a second Nelder-Mead pass with tighter tolerances
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                remaining = budget - calls
                if remaining >= 10:
                    est_maxiter = max(1, remaining // (dim + 1))
                    res = opt.minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': est_maxiter, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
            except (TimeoutError, StopIteration):
                pass
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = safe_func(best_x)
            except Exception:
                best_f = func(best_x)

        return best_x, best_f

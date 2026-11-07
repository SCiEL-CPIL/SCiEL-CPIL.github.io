"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B + Nelder-Mead refinement with aggressive deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 275  # 25s safety margin before 300s limit

        call_count = [0]
        max_calls = budget
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_func(x):
            if call_count[0] >= max_calls:
                return best_f if best_f != float('inf') else 1e10
            if time.time() > deadline:
                return best_f if best_f != float('inf') else 1e10
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            try:
                val = float(func(x))
            except Exception:
                val = best_f if best_f != float('inf') else 1e10
            if not np.isfinite(val):
                val = best_f if best_f != float('inf') else 1e10
            if val < best_f:
                best_f[0] = val
                best_x[:] = x
            return val

        # Use mutable containers for best tracking
        best_f_arr = [float('inf')]
        best_x_arr = np.zeros(dim)

        def evaluate(x):
            if call_count[0] >= max_calls:
                return best_f_arr[0]
            if time.time() > deadline:
                return best_f_arr[0]
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            try:
                val = float(func(x))
            except Exception:
                val = best_f_arr[0]
            if not np.isfinite(val):
                val = best_f_arr[0]
            if val < best_f_arr[0]:
                best_f_arr[0] = val
                best_x_arr[:] = x
            return val

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = []

        # Center
        starts.append(np.zeros(dim))

        # Random points in likely region [-4, 4]^5
        for _ in range(9):
            starts.append(rng.uniform(-4.0, 4.0, dim))

        # Evaluate all starts to rank basins
        start_results = []
        for x in starts:
            f_val = evaluate(x)
            start_results.append((f_val, x.copy()))

        if not start_results:
            return best_x_arr, best_f_arr[0]

        start_results.sort(key=lambda t: t[0])

        # Refine top 4 basins
        top_basins = start_results[:4]

        for i, (f_init, x_init) in enumerate(top_basins):
            if call_count[0] >= max_calls or time.time() > deadline:
                break

            # Phase 1: L-BFGS-B for fast gradient-based convergence
            try:
                max_iter = 150 if i == 0 else 80
                res = minimize(
                    evaluate,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': max_iter,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 30
                    }
                )
                if res.fun < best_f_arr[0]:
                    best_f_arr[0] = res.fun
                    best_x_arr[:] = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

            # Phase 2: Nelder-Mead to escape flat regions / shallow local minima
            if call_count[0] < max_calls and time.time() < deadline:
                try:
                    res_nm = minimize(
                        evaluate,
                        best_x_arr.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 500,
                            'xatol': 1e-11,
                            'fatol': 1e-11,
                            'adaptive': True
                        }
                    )
                    if res_nm.fun < best_f_arr[0]:
                        best_f_arr[0] = res_nm.fun
                        best_x_arr[:] = np.clip(res_nm.x, -5.0, 5.0)
                except Exception:
                    pass

            # Phase 3: High-precision L-BFGS-B polish
            if call_count[0] < max_calls - 20 and time.time() < deadline:
                try:
                    res_polish = minimize(
                        evaluate,
                        best_x_arr.copy(),
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': 300,
                            'ftol': 1e-15,
                            'gtol': 1e-14,
                            'maxls': 50
                        }
                    )
                    if res_polish.fun < best_f_arr[0]:
                        best_f_arr[0] = res_polish.fun
                        best_x_arr[:] = np.clip(res_polish.x, -5.0, 5.0)
                except Exception:
                    pass

            # Phase 4: Final Nelder-Mead polish
            if call_count[0] < max_calls - 10 and time.time() < deadline:
                try:
                    res_final = minimize(
                        evaluate,
                        best_x_arr.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 400,
                            'xatol': 1e-13,
                            'fatol': 1e-13,
                            'adaptive': True
                        }
                    )
                    if res_final.fun < best_f_arr[0]:
                        best_f_arr[0] = res_final.fun
                        best_x_arr[:] = np.clip(res_final.x, -5.0, 5.0)
                except Exception:
                    pass

        return best_x_arr, best_f_arr[0]
# EVOLVE-BLOCK-END

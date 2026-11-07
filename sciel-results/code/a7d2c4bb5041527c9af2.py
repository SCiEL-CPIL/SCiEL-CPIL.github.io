import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Basin Hopping with adaptive step size, combined with L-BFGS-B and
    Nelder-Mead local exploitation.  Uses a deadline guard so the program
    always returns before the 330 s wall-clock kill.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280          # 20 s safety margin before the 330 s kill
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf

        # --- evaluation wrapper with budget & time guard -------------------
        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("time exceeded")
            return float(func(x))

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        rng = np.random.default_rng(42)

        # --- 1. Diverse initial starts -------------------------------------
        starts = [np.zeros(dim)]
        for _ in range(12):
            starts.append(rng.uniform(-5, 5, dim))
        # structured axis points
        for i in range(dim):
            for val in (-4, -2, 0, 2, 4):
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)

        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            if not any(np.allclose(s, u, atol=0.1) for u in unique_starts):
                unique_starts.append(s)

        for s in unique_starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            try:
                f = wrapped(s)
                update_best(s, f)
            except Exception:
                pass

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)

        current_x = best_x.copy()
        current_f = best_f
        step_size = 1.5

        # --- 2. Basin Hopping loop -----------------------------------------
        while calls[0] < budget and time.time() < deadline:
            direction = rng.standard_normal(dim)
            perturbed_x = np.clip(current_x + step_size * direction, -5, 5)

            new_x, new_f = perturbed_x, np.inf

            # L-BFGS-B first
            try:
                res = minimize(
                    wrapped,
                    perturbed_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxls': 15, 'gtol': 1e-4}
                )
                new_x, new_f = res.x, res.fun
            except Exception:
                pass

            # Nelder-Mead if L-BFGS-B didn't help
            if (new_f >= current_f or new_f == np.inf) and calls[0] < budget:
                try:
                    res_nm = minimize(
                        wrapped,
                        perturbed_x,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-6,
                                 'fatol': 1e-10, 'adaptive': True}
                    )
                    if res_nm.fun < new_f:
                        new_x, new_f = res_nm.x, res_nm.fun
                except Exception:
                    pass

            if new_f < current_f:
                current_x = new_x.copy()
                current_f = new_f
                update_best(current_x, current_f)
                step_size *= 0.8
            else:
                step_size *= 1.3
                if step_size > 8.0:
                    step_size = 8.0

        # --- 3. Final high-precision polish --------------------------------
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            try:
                res_final = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-9,
                             'fatol': 1e-13, 'adaptive': True}
                )
                update_best(res_final.x, res_final.fun)
            except Exception:
                pass

        return best_x, best_f

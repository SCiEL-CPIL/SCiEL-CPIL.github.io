import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0

        rng = np.random.default_rng(12345)
        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 2:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        # Phase 1: Initial sampling
        n_structured = 12
        structured = [
            np.zeros(self.dim),
            np.ones(self.dim) * 0.5,
            np.ones(self.dim) * -0.5,
            np.ones(self.dim),
            np.ones(self.dim) * -1.0,
            np.ones(self.dim) * 2.0,
            np.ones(self.dim) * -2.0,
            np.ones(self.dim) * 0.1,
            np.ones(self.dim) * 3.0,
            np.ones(self.dim) * -3.0,
            np.ones(self.dim) * 1.5,
            np.ones(self.dim) * -1.5,
        ]
        n_random = 30
        random_pts = rng.uniform(-4, 4, (n_random, self.dim))

        all_pts = np.vstack([structured, random_pts])
        for p in all_pts:
            if call_count >= self.budget or time.time() > deadline - 10:
                break
            try:
                safe_func(p)
            except ValueError:
                break

        if best_x is None:
            return np.zeros(self.dim), 0.0

        # Phase 2: L-BFGS-B from best point
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 10:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 300, 'maxfun': remaining, 'maxls': 50, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        # Phase 3: Multi-start refinement with perturbed points
        n_perturb = 5
        for i in range(n_perturb):
            remaining = self.budget - call_count
            if remaining < 20 or time.time() > deadline - 10:
                break
            try:
                sigma = 0.2 + 0.1 * i
                perturb = rng.normal(0, sigma, self.dim)
                start_p = np.clip(best_x + perturb, -5, 5)
                res_p = minimize(
                    safe_func,
                    start_p,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 150, 'maxfun': remaining, 'maxls': 30, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                if res_p.fun < best_f:
                    best_f = float(res_p.fun)
                    best_x = np.array(res_p.x, dtype=float)
            except Exception:
                pass

        # Phase 4: Differential Evolution for global exploration if budget allows
        remaining = self.budget - call_count
        if remaining > 100 and time.time() < deadline - 30:
            try:
                def de_func(x):
                    nonlocal call_count
                    if call_count >= self.budget or time.time() > deadline - 5:
                        return 1e10
                    call_count += 1
                    val = float(func(x))
                    if val < best_f:
                        best_f = val
                        best_x = np.array(x, dtype=float)
                    return val

                de_res = differential_evolution(
                    de_func,
                    bounds=[(-5, 5)] * self.dim,
                    maxiter=50,
                    popsize=15,
                    tol=1e-10,
                    mutation=(0.3, 1.0),
                    recombination=0.8,
                    seed=42,
                    polish=False,
                    init='latinhypercube',
                    updating='immediate',
                    workers=1,
                )
                if de_res.fun < best_f:
                    best_f = float(de_res.fun)
                    best_x = np.array(de_res.x, dtype=float)
            except Exception:
                pass

        # Phase 5: Final high-precision polish with L-BFGS-B
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 5:
            try:
                res_final = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': remaining, 'maxls': 50, 'gtol': 1e-12, 'ftol': 1e-16}
                )
                if res_final.fun < best_f:
                    best_f = float(res_final.fun)
                    best_x = np.array(res_final.x, dtype=float)
            except Exception:
                pass

        # Phase 6: Nelder-Mead final polish
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 3:
            try:
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True, 'maxfev': remaining}
                )
                if res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.array(res_nm.x, dtype=float)
            except Exception:
                pass

        return best_x, best_f

import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def f(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            xc = np.clip(x, -5.0, 5.0)
            val = float(func(xc))
            if val < best_f:
                best_f = val
                best_x = xc.copy()
            return val

        # ---- (1+1)-ES refinement with adaptive step size ----
        def es(x, fx, iters, sigma_init=0.3):
            rng = np.random.default_rng(42)
            sigma = sigma_init
            for _ in range(iters):
                if time.time() > deadline or calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                else:
                    sigma *= 0.85
            return x, fx

        # ---- Multi-start strategy ----
        rng0 = np.random.default_rng(7)

        # Quick random scan to find a good starting region
        candidates = []
        for i in range(8):
            if time.time() > deadline or calls >= budget:
                break
            x_try = rng0.uniform(-4.0, 4.0, dim)
            fx_try = f(x_try)
            candidates.append((fx_try, x_try))

        # Sort by function value
        candidates.sort(key=lambda t: t[0])

        # Run Nelder-Mead from the best 2-3 starting points using scipy
        best_nm_x = None
        best_nm_f = np.inf

        for i in range(min(3, len(candidates))):
            if time.time() > deadline or calls >= budget:
                break
            x_start = candidates[i][1].copy()
            
            # Construct initial simplex with larger delta for faster convergence
            delta = 3.0
            simplex = [x_start.copy()]
            for j in range(dim):
                xi = x_start.copy()
                xi[j] += delta
                simplex.append(xi)

            try:
                res = minimize(f, x_start, method='Nelder-Mead',
                               options={'initial_simplex': simplex, 'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_nm_f:
                    best_nm_f = res.fun
                    best_nm_x = res.x
            except Exception:
                pass

        # Polish with ES
        if best_nm_x is not None:
            x_final, fx_final = es(best_nm_x, best_nm_f, 100)

            # If budget remains, do another round of ES with smaller steps
            if time.time() < deadline and calls < budget:
                x_final, fx_final = es(x_final, fx_final, 80, sigma_init=0.1)

        return best_x, best_f

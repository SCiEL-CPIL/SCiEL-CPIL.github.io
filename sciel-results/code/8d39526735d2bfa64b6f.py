import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration using Sobol sequence
        n_init = 50
        if n_init < budget:
            sobol = qmc.Sobol(d=n, scramble=True, seed=42)
            sobol.random(1)
            pts = sobol.random(n_init) * 10.0 - 5.0
            for i in range(n_init):
                try_func(pts[i])
                if calls >= budget or time.time() > deadline:
                    break

        # Phase 2: Powell Optimization
        # Powell is a derivative-free local optimizer that is robust to rotations.
        if calls < budget and time.time() < deadline:
            def obj(x):
                return try_func(x)
            
            try:
                res = minimize(obj, best_x, method='Powell',
                              bounds=[(-5.0, 5.0)] * n,
                              options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 3: Nelder-Mead Polish
        if calls < budget - 10 and time.time() < deadline:
            def obj2(x):
                return try_func(x)
            
            try:
                res = minimize(obj2, best_x, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 4: L-BFGS-B Polish (if we have budget left)
        if calls < budget - 10 and time.time() < deadline:
            def obj3(x):
                return try_func(x)
            
            try:
                res = minimize(obj3, best_x, method='L-BFGS-B',
                              bounds=[(-5.0, 5.0)] * n,
                              options={'maxiter': 50, 'gtol': 1e-12, 'ftol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f

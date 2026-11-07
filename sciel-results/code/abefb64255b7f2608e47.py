import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def f(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return 1e10
            if time.time() > deadline:
                return 1e10
            calls += 1
            xc = np.clip(x, -5.0, 5.0)
            val = float(func(xc))
            if val < best_f:
                best_f = val
                best_x = xc.copy()
            return val

        # ---- (1+1)-ES refinement with adaptive step size ----
        def es(x, fx, iters, rng):
            sigma = 0.5
            for _ in range(iters):
                if time.time() > deadline or calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
            return x, fx

        # ---- Multi-start Nelder-Mead strategy ----
        # Key insight: Use a larger initial simplex (delta=3.0) to cover the domain
        # quickly, which is crucial for AOCC scoring (early progress).
        # Run Nelder-Mead from multiple starting points to handle the hidden rotation
        # and shift, then polish the best result with ES.
        
        rng0 = np.random.default_rng(42)

        # First, evaluate a few random points to find promising starting regions
        # This helps with AOCC by getting some early progress
        candidates = []
        # Evaluate center first
        x_center = np.zeros(dim)
        f_center = f(x_center)
        candidates.append((f_center, x_center))

        # Evaluate a few random points
        for i in range(5):
            if time.time() > deadline or calls >= budget:
                break
            x_try = rng0.uniform(-4.0, 4.0, dim)
            fx_try = f(x_try)
            candidates.append((fx_try, x_try))

        # Sort by function value
        candidates.sort(key=lambda t: t[0])

        # Run Nelder-Mead from the best 2-3 starting points
        # Use a larger delta (3.0) for the initial simplex to cover more ground
        best_nm_x = None
        best_nm_f = np.inf

        for i in range(min(3, len(candidates))):
            if time.time() > deadline or calls >= budget:
                break
            
            x_start = candidates[i][1].copy()
            
            # Construct initial simplex with delta=3.0
            delta = 3.0
            simplex = [x_start.copy()]
            for j in range(dim):
                xi = x_start.copy()
                xi[j] += delta
                simplex.append(xi)
            
            # Use scipy's Nelder-Mead with adaptive settings
            options = {
                'initial_simplex': simplex,
                'maxiter': 50,
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
            
            try:
                res = minimize(f, x_start, method='Nelder-Mead', options=options)
                if res.fun < best_nm_f:
                    best_nm_f = res.fun
                    best_nm_x = res.x.copy()
            except Exception:
                pass

        # If Nelder-Mead found a good point, polish with ES
        if best_nm_x is not None and time.time() < deadline and calls < budget:
            # Use a fresh RNG for ES to avoid correlation with the search
            rng_es = np.random.default_rng(123)
            
            # First ES pass with moderate steps
            x_final, fx_final = es(best_nm_x, best_nm_f, 150, rng_es)
            
            # If budget remains, do another ES pass with smaller steps for refinement
            if time.time() < deadline and calls < budget:
                rng_es2 = np.random.default_rng(456)
                x_final, fx_final = es(x_final, fx_final, 100, rng_es2)

        return best_x, best_f

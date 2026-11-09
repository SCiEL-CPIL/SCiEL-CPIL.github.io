import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() - start_time < time_limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if not time_ok():
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                v = float(func(x))
            except Exception:
                return best_f
            if not np.isfinite(v):
                v = best_f
                return v
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        # 1. Initialization: Latin Hypercube Sampling
        # We want good coverage of [-5, 5]^dim
        # 15 points is a good start for 5D
        n_init = 15
        try:
            sampler = qmc.LatinHypercube(d=dim, seed=42)
            points = sampler.random(n=n_init)
            # Scale from [0, 1] to [-5, 5]
            points = points * 10.0 - 5.0
        except Exception:
            points = rng.uniform(-5, 5, (n_init, dim))

        # Add origin and a few axis-aligned points to ensure we check the center
        extra_points = [np.zeros(dim)]
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 4.0
            extra_points.append(p)
            p[i] = -4.0
            extra_points.append(p)
        
        all_init_points = np.vstack([points, extra_points])
        
        # Evaluate initial points
        init_values = []
        for p in all_init_points:
            if not time_ok() or calls >= budget - 50:
                break
            v = wrapped_func(p)
            init_values.append(v)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        # Sort by value to get top candidates
        # We need to handle the case where some evaluations failed (returned best_f)
        # But wrapped_func returns best_f on failure, so they will be at the end if best_f is inf initially?
        # No, if best_f is inf, and we fail, we return inf.
        # If we have valid values, they are finite.
        
        # Create a list of (value, point)
        candidates = list(zip(init_values, all_init_points))
        # Sort by value
        candidates.sort(key=lambda x: x[0])
        
        # Get top 3 unique points
        top_starts = []
        seen = []
        for val, p in candidates:
            if not np.isfinite(val):
                continue
            # Check if point is already in top_starts (within tolerance)
            is_unique = True
            for existing in seen:
                if np.linalg.norm(p - existing) < 0.1:
                    is_unique = False
                    break
            if is_unique:
                top_starts.append(p.copy())
                seen.append(p)
            if len(top_starts) >= 3:
                break

        if not top_starts:
            top_starts = [best_x.copy()]

        # 2. Local Optimization using Powell
        # Allocate budget among the top starts
        # We have budget - calls left.
        # Let's give 50% to the best, 30% to the 2nd, 20% to the 3rd.
        
        remaining_budget = budget - calls
        if remaining_budget < 10:
            return best_x, best_f

        # Define budget splits
        splits = [0.5, 0.3, 0.2]
        # If we have fewer than 3 starts, redistribute
        if len(top_starts) == 1:
            splits = [1.0]
        elif len(top_starts) == 2:
            splits = [0.7, 0.3]

        for i, start_point in enumerate(top_starts):
            if not time_ok() or calls >= budget - 5:
                break
            
            # Estimate max iterations for Powell
            # Powell uses roughly 2*dim evaluations per iteration in the worst case, but often more.
            # Let's just let it run until budget or time is exhausted, but cap maxiter.
            # A safe maxiter is 100-200.
            max_iter = 100
            
            try:
                res = minimize(
                    wrapped_func, 
                    start_point, 
                    method='Powell', 
                    options={
                        'maxiter': max_iter,
                        'ftol': 1e-12,
                        'xtol': 1e-10,
                        'disp': False
                    }
                )
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # 3. Final Polish with Nelder-Mead
        # Use remaining budget for a high-precision local search
        if calls < budget - 5 and time_ok() and best_x is not None:
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 500,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f

import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf
        candidates = []  # List of (f_val, x) for multi-start

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 2:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            candidates.append((val, np.array(x, dtype=float)))
            return val

        # Phase 1: Initial sampling to find the basin
        # Include origin, structured points, and random samples
        n_structured = 8
        structured = [
            np.zeros(self.dim),
            np.ones(self.dim) * 0.5,
            np.ones(self.dim) * -0.5,
            np.ones(self.dim),
            np.ones(self.dim) * -1.0,
            np.ones(self.dim) * 2.0,
            np.ones(self.dim) * -2.0,
            np.ones(self.dim) * 0.1,
        ]
        n_random = 40
        random_pts = rng.uniform(-4, 4, (n_random, self.dim))

        all_pts = np.vstack([structured, random_pts])
        for p in all_pts:
            if call_count >= self.budget or time.time() > deadline - 5:
                break
            try:
                safe_func(p)
            except ValueError:
                break

        if best_x is None:
            # Fallback: return a default point
            return np.zeros(self.dim), 0.0

        # Sort candidates by function value to get top seeds
        candidates.sort(key=lambda c: c[0])
        
        # Phase 2: Multi-start L-BFGS-B from top seeds
        # Rosenbrock has a narrow valley; starting from multiple good points
        # increases the chance of finding the global basin.
        remaining = self.budget - call_count
        n_seeds = min(5, len(candidates))  # Use top 5 seeds
        
        for i in range(n_seeds):
            if time.time() > deadline - 10 or call_count >= self.budget - 10:
                break
            
            start_x = candidates[i][1].copy()
            
            # Allocate remaining budget among remaining seeds
            calls_per_seed = max(10, (self.budget - call_count) // (n_seeds - i))
            
            try:
                res = minimize(
                    safe_func,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 200,
                        'maxfun': calls_per_seed,
                        'maxls': 50,
                        'gtol': 1e-8,
                        'ftol': 1e-12
                    }
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        # Phase 3: Final high-precision polish with L-BFGS-B
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 5:
            try:
                res2 = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxfun': remaining,
                        'maxls': 50,
                        'gtol': 1e-10,
                        'ftol': 1e-14
                    }
                )
                if res2.fun < best_f:
                    best_f = float(res2.fun)
                    best_x = np.array(res2.x, dtype=float)
            except Exception:
                pass

        # Phase 4: Nelder-Mead for potential escape from local minima
        remaining = self.budget - call_count
        if remaining > 30 and time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.array(res_nm.x, dtype=float)
            except Exception:
                pass

        return best_x, best_f

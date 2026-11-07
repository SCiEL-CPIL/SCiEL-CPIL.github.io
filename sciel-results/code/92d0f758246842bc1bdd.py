"""Optimized BBOB f21 (Gallagher 101 peaks) solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-phase optimizer: coarse grid + CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            return func(x)

        def time_ok():
            return time.time() < deadline

        # Phase 1: Deterministic coarse grid to guarantee early progress
        # Evaluate key structural points first for AOCC benefit
        # Vertices of [-4,4]^5, center, and axis midpoints
        key_points = []
        
        # Center
        key_points.append(np.zeros(n))
        
        # Axis points: +/- 4 on one axis, 0 elsewhere
        for i in range(n):
            for sign in [1, -1]:
                p = np.zeros(n)
                p[i] = sign * 4.0
                key_points.append(p)
        
        # Vertices of [-4, 4]^5
        for mask in range(2 ** n):
            v = np.array([4.0 if (mask >> i) & 1 else -4.0 for i in range(n)])
            key_points.append(v)
        
        # Some intermediate points
        for sign in [1, -1]:
            for i in range(n):
                p = np.zeros(n)
                p[i] = sign * 2.0
                key_points.append(p)
        
        # Deduplicate
        seen = set()
        unique_points = []
        for p in key_points:
            key = tuple(np.round(p, 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)
        
        results = []
        for pt in unique_points:
            if calls >= budget or not time_ok():
                break
            x = np.array(pt, dtype=np.float64)
            f = safe_call(x)
            if np.isfinite(f):
                results.append((f, x.copy()))
        
        if not results:
            # Fallback: just evaluate origin
            x0 = np.zeros(n)
            f0 = safe_call(x0)
            return x0, f0
        
        # Sort by function value
        results.sort(key=lambda item: item[0])
        
        # Get top candidates
        top_k = min(5, len(results))
        candidates = [results[i][1] for i in range(top_k)]
        
        best_x = candidates[0].copy()
        best_f = results[0][0]
        
        # Phase 2: Local refinement using Nelder-Mead and possibly CMA-ES
        # Use remaining budget for local search
        for i in range(top_k):
            if calls >= budget or not time_ok():
                break
            
            x0 = candidates[i].copy()
            remaining = budget - calls
            if remaining < 20:
                break
            
            # Try Nelder-Mead first (faster, good for smooth local minima)
            try:
                res = minimize(
                    safe_call, x0, method='Nelder-Mead',
                    options={'maxfev': min(remaining, 150), 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            if calls >= budget or not time_ok():
                break
            
            # If we still have budget, try a second refinement from the best found so far
            if i == 0:
                remaining = budget - calls
                if remaining > 50 and time_ok():
                    try:
                        res2 = minimize(
                            safe_call, best_x, method='Nelder-Mead',
                            options={'maxfev': min(remaining, 100), 'xatol': 1e-12, 'fatol': 1e-14}
                        )
                        if res2.fun < best_f and np.isfinite(res2.fun):
                            best_f = res2.fun
                            best_x = res2.x.copy()
                    except Exception:
                        pass
        
        # Phase 3: If budget remains, do a finer local search around the best point
        if calls < budget and time_ok():
            remaining = budget - calls
            if remaining > 10:
                # Try a few random perturbations around the best point
                rng = np.random.RandomState(42)
                for _ in range(min(remaining // 5, 20)):
                    if calls >= budget or not time_ok():
                        break
                    perturbation = rng.randn(n) * 0.1
                    x_try = best_x + perturbation
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = safe_call(x_try)
                    if np.isfinite(f_try) and f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END

"""Optimized solver for BBOB f4 (Büche-Rastrigin)"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        call_count = 0
        best_x = None
        best_f = float('inf')

        def count_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        rng = np.random.default_rng(12345)

        # Phase 1: Coarse global exploration with stratified sampling
        # Büche-Rastrigin has many local minima; we need good initial coverage
        # Use a grid + random mixture for the first ~50 evaluations
        
        # Stratified sampling: divide [-4,4]^5 into cells, sample one per cell
        # 5D with 2 per axis = 32 cells, but that's too many for 50 budget
        # Use 2 per axis for first 3 dims, 1 for last 2 = 8*8*2*2 = 256, too many
        # Instead: random points with a few structured ones
        
        n_explore = 30
        points = []
        
        # Center
        points.append(np.zeros(dim))
        
        # Axis-aligned points at ±3
        for axis in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[axis] = sign * 3.0
                points.append(pt)
        
        # Random points in [-4, 4]^5
        while len(points) < n_explore:
            points.append(rng.uniform(-4.0, 4.0, dim))
        
        for p in points:
            if call_count >= budget or time.time() > deadline:
                break
            try:
                count_call(p)
            except Exception:
                break

        # Phase 2: Multi-start CMA-ES with restarts
        # CMA-ES is well-suited for BBOB functions
        # Use multiple restarts with decreasing population sizes
        
        from scipy.optimize import minimize
        
        # Try Nelder-Mead first as it's fast for 5D
        if best_x is not None and call_count < budget - 20 and time.time() < deadline:
            try:
                res = minimize(count_call, best_x, method='Nelder-Mead',
                               options={'maxiter': 150, 'xatol': 1e-6, 'fatol': 1e-8, 
                                        'adaptive': True, 'maxfev': 150})
            except Exception:
                pass

        # Phase 3: CMA-ES style local search with multiple restarts
        # Implement a simple CMA-ES or use scipy's Powell as fallback
        
        if call_count < budget - 30 and time.time() < deadline:
            # Try Powell's method from the best point
            try:
                res = minimize(count_call, best_x, method='Powell',
                               options={'maxiter': 100, 'xtol': 1e-6, 'ftol': 1e-8})
            except Exception:
                pass

        # Phase 4: More random restarts with local refinement
        n_restarts = 5
        for r in range(n_restarts):
            if call_count >= budget - 20 or time.time() > deadline:
                break
            
            # Start near the current best with some perturbation
            if best_x is not None:
                x0 = best_x + rng.normal(0, 1.0, dim)
                x0 = np.clip(x0, -4.5, 4.5)
            else:
                x0 = rng.uniform(-4.0, 4.0, dim)
            
            try:
                res = minimize(count_call, x0, method='Nelder-Mead',
                               options={'maxiter': 80, 'xatol': 1e-5, 'fatol': 1e-7,
                                        'adaptive': True, 'maxfev': 80})
            except Exception:
                pass

        # Phase 5: Fine-tune with coordinate descent
        if best_x is not None and call_count < budget - 5 and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            step_size = 0.5
            for _ in range(30):
                if call_count >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * step_size
                        try:
                            f_try = count_call(x_try)
                        except Exception:
                            break
                        if f_try < f:
                            x[i] = x_try[i]
                            f = f_try
                            improved = True
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-5:
                        break
            best_x = x
            best_f = f

        return best_x, best_f

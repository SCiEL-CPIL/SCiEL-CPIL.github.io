import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin out of 330s limit
        call_count = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            if time.time() > deadline:
                return best_f
            call_count += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float).copy()
            return fx

        def remaining_calls():
            return max(0, self.budget - call_count)

        def time_ok():
            return time.time() < deadline

        # Phase 1: Global search with Differential Evolution
        # Use ~55% of budget for DE to find a good starting point
        de_budget = int(self.budget * 0.55)
        
        try:
            bounds = [(-5, 5)] * self.dim
            popsize = 8
            maxiter = max(1, de_budget // popsize - 1)
            
            res = differential_evolution(
                counted_func,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-8,
                seed=42,
                polish=False
            )
            if res.x is not None and res.fun < best_f:
                best_f = res.fun
                best_x = np.array(res.x, dtype=float).copy()
        except Exception:
            pass

        # Fallback if DE didn't work
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = counted_func(best_x)

        # Phase 2: Multi-start Nelder-Mead refinement
        # Generate diverse starting points for local refinement
        rng = np.random.default_rng(123)
        
        # Collect candidate starting points
        candidates = [best_x.copy()]
        
        # Add some random points near the best
        for _ in range(3):
            if not time_ok() or remaining_calls() < 10:
                break
            perturbation = 0.5 * rng.standard_normal(self.dim)
            candidate = np.clip(best_x + perturbation, -5, 5)
            candidates.append(candidate)
        
        # Add some points in the central region
        for _ in range(2):
            if not time_ok() or remaining_calls() < 10:
                break
            candidate = 3.0 * rng.standard_normal(self.dim)
            candidate = np.clip(candidate, -5, 5)
            candidates.append(candidate)

        # Apply Nelder-Mead to each candidate
        for cand in candidates:
            if not time_ok() or remaining_calls() < 5:
                break
            
            try:
                res = minimize(
                    counted_func,
                    cand,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining_calls() // 2, 200),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x, dtype=float).copy()
            except Exception:
                pass

        # Phase 3: Final (1+1)-ES polish with adaptive sigma
        if time_ok() and remaining_calls() > 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Start small for fine polishing
            rng2 = np.random.default_rng(99)
            
            for _ in range(remaining_calls()):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng2.standard_normal(self.dim), -5, 5)
                fy = counted_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05  # Slightly increase on success
                else:
                    sigma *= 0.9   # Decrease on failure
                
                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 1e-6, 2.0)
            
            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f

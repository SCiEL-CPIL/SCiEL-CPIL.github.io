import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        bounds = [(-5.0, 5.0)] * self.dim

        # Phase 1: Global Search - evaluate strategic points to find basin
        # Origin + random points in [-4,4]^5 where minimizer is known to lie
        # Use a slightly larger set of points to increase chance of hitting the basin
        starts = [np.zeros(self.dim)]
        for _ in range(14):
            starts.append(rng.uniform(-4.0, 4.0, self.dim))
        
        # Evaluate starts
        for x0 in starts:
            if call_count >= self.budget - 300: # Leave 300 for local opt
                break
            tracked_func(x0)
        
        # Sort starts by function value to prioritize best ones for local optimization
        # We need to re-evaluate or store values. We didn't store values in the list.
        # Let's re-evaluate the top few? No, we can just run local opt on the current best.
        # To do multi-start, we need to know the values of the other starts.
        # Let's modify Phase 1 to store (x, f) pairs.
        
        # Reset for a cleaner multi-start approach
        call_count = 0
        best_fx = np.inf
        best_x = np.zeros(self.dim)
        
        # Re-define tracked_func to be safe (it's a closure, so it's fine)
        
        candidates = []
        for x0 in starts:
            if call_count >= self.budget - 300:
                break
            fx = tracked_func(x0)
            candidates.append((x0, fx))
            
        # Sort candidates by fx
        candidates.sort(key=lambda item: item[1])
        
        # Phase 2: Local Optimization from top candidates
        # Use Nelder-Mead for robustness in low dimensions
        # It is derivative-free and handles the Rosenbrock valley well
        
        # We have roughly 300 calls left.
        # Try the best candidate first with a large budget.
        if time_ok() and call_count < self.budget - 10:
            try:
                x_start = candidates[0][0].copy()
                maxfun_remaining = min(self.budget - call_count - 5, 200)
                if maxfun_remaining < 10:
                    maxfun_remaining = 10
                    
                res = minimize(
                    tracked_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'maxfev': maxfun_remaining,
                        'adaptive': True,
                        'xatol': 1e-12,
                        'fatol': 1e-14
                    }
                )
            except Exception:
                pass
        
        # If budget remains, try the second best candidate
        if time_ok() and call_count < self.budget - 10 and len(candidates) > 1:
            try:
                x_start = candidates[1][0].copy()
                remaining = self.budget - call_count
                if remaining > 50:
                    res = minimize(
                        tracked_func,
                        x_start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 200,
                            'maxfev': min(remaining, 100),
                            'adaptive': True,
                            'xatol': 1e-12,
                            'fatol': 1e-14
                        }
                    )
            except Exception:
                pass

        # If budget remains, try the third best candidate
        if time_ok() and call_count < self.budget - 10 and len(candidates) > 2:
            try:
                x_start = candidates[2][0].copy()
                remaining = self.budget - call_count
                if remaining > 30:
                    res = minimize(
                        tracked_func,
                        x_start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': min(remaining, 50),
                            'adaptive': True,
                            'xatol': 1e-12,
                            'fatol': 1e-14
                        }
                    )
            except Exception:
                pass

        return best_x, best_fx

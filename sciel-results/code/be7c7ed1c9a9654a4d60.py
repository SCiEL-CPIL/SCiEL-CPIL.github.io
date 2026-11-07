import numpy as np
import scipy.optimize as opt
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin before 330s limit
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        used_calls = 0
        max_calls = self.budget
        
        def safe_func(x):
            nonlocal used_calls
            if used_calls >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            used_calls += 1
            return func(np.clip(x, -5, 5))
        
        # 1. Global Exploration Phase with more diverse starting points
        n_initial = 15
        initial_points = rng.uniform(-5, 5, size=(n_initial, self.dim))
        
        for p in initial_points:
            if used_calls >= max_calls:
                break
            if time.time() > deadline:
                break
            try:
                fp = safe_func(p)
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
            except RuntimeError:
                break
        
        if best_x is None:
            return None, np.inf
        
        # 2. Multi-start local optimization with Nelder-Mead
        # Use remaining budget for local optimization
        remaining = max_calls - used_calls
        if remaining > 10 and time.time() < deadline:
            try:
                result = opt.minimize(
                    safe_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining, 200),
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True,
                        'maxfev': remaining
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except (RuntimeError, Exception):
                pass  # Keep the best found so far
        
        # 3. If we have budget left, try a few more random restarts with local optimization
        if used_calls < max_calls and time.time() < deadline:
            n_restarts = min(5, (max_calls - used_calls) // 50)
            for _ in range(n_restarts):
                if used_calls >= max_calls or time.time() > deadline:
                    break
                # Sample near the current best or randomly
                if rng.random() < 0.5 and best_x is not None:
                    x_start = np.clip(best_x + rng.normal(0, 0.5, self.dim), -5, 5)
                else:
                    x_start = rng.uniform(-5, 5, self.dim)
                
                try:
                    fp = safe_func(x_start)
                    if fp < best_f:
                        best_f = fp
                        best_x = x_start.copy()
                    
                    if used_calls < max_calls and time.time() < deadline:
                        local_remaining = max_calls - used_calls
                        if local_remaining > 10:
                            result = opt.minimize(
                                safe_func, 
                                x_start, 
                                method='Nelder-Mead',
                                options={
                                    'maxiter': min(local_remaining, 100),
                                    'xatol': 1e-8,
                                    'fatol': 1e-8,
                                    'adaptive': True,
                                    'maxfev': local_remaining
                                }
                            )
                            if result.fun < best_f:
                                best_f = result.fun
                                best_x = result.x.copy()
                except (RuntimeError, Exception):
                    break
        
        return best_x, best_f

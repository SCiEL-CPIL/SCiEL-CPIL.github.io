import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin out of 300s
        
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        budget = self.budget
        bound = self.bound

        def time_ok():
            return time.time() < deadline and self.calls < budget

        def wrapped_func(x):
            if not time_ok():
                return self.best_f
            x = np.clip(np.asarray(x, dtype=float), -bound, bound)
            self.calls += 1
            try:
                val = float(func(x))
            except Exception:
                val = 1e10
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Phase 1: Global Search
        # Use a mix of random and structured points to find the valley
        # 5D space is small, 30-40 points is usually enough
        n_global = 40
        rng = np.random.default_rng(42)
        
        candidates = []
        
        # Add some structured points to increase coverage
        # Center
        candidates.append(np.zeros(dim))
        # Axes
        for i in range(dim):
            for sign in [-1, 1]:
                x = np.zeros(dim)
                x[i] = sign * 3.0
                candidates.append(x)
        
        # Fill rest with random
        while len(candidates) < n_global:
            x = rng.uniform(-bound, bound, dim)
            candidates.append(x)
            
        # Evaluate candidates
        for x in candidates:
            if not time_ok():
                break
            val = wrapped_func(x)
        
        if self.best_x is None:
            # Fallback if no calls made (should not happen)
            x0 = np.zeros(dim)
            wrapped_func(x0)
            
        # Sort candidates by function value for local search initialization
        # We need to re-evaluate or track values. Let's just use the best found.
        # Actually, let's keep a list of (value, x) for better multi-start.
        # Refactor Phase 1 to collect results properly.
        
        # Re-do Phase 1 collection logic for better multi-start
        # The above loop already updated best_x/best_f. 
        # Let's just use best_x for the primary local search.
        # If we want multi-start, we should have collected values.
        # Let's stick to a simpler approach: 
        # 1. Global search finds best_x.
        # 2. Local search from best_x.
        # 3. If budget remains, try a second local search from a random point or second best.
        
        # Phase 2: Local Optimization with L-BFGS-B
        # L-BFGS-B is robust and uses finite differences. Very effective for smooth functions.
        bounds = [(-bound, bound)] * dim
        
        # Run L-BFGS-B from the best global point
        if time_ok():
            try:
                # maxiter is not the same as maxfun. L-BFGS-B uses maxfun if provided in options?
                # In scipy, 'maxfun' is not a standard option for L-BFGS-B in all versions, 
                # but 'maxiter' is. However, each iteration can use multiple function evaluations.
                # It is safer to use a callback to check budget.
                
                def lbfgs_callback(xk):
                    if not time_ok():
                        return True
                    return False

                res = scipy.optimize.minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    callback=lbfgs_callback,
                    options={
                        'maxiter': 500, # High enough to not restrict if budget allows
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20 
                    }
                )
            except Exception:
                pass

        # Phase 3: Polish with Nelder-Mead if budget remains
        # Nelder-Mead is derivative-free and can refine further or escape shallow local minima.
        if time_ok() and self.calls < budget - 10:
            try:
                res_nm = scipy.optimize.minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': max(10, budget - self.calls),
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
            except Exception:
                pass

        # Phase 4: Final Polish with tight L-BFGS-B if budget remains
        if time_ok() and self.calls < budget - 5:
            try:
                res_final = scipy.optimize.minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-15,
                        'gtol': 1e-10
                    }
                )
            except Exception:
                pass

        return self.best_x, self.best_f

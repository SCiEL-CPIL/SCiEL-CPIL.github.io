import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = None
        self.time_limit = 290.0  # Safety margin from 300s limit

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            self.calls += 1
            if self.calls > self.budget:
                return self.best_f
            if time.time() - self.start_time > self.time_limit:
                return self.best_f
            
            x = np.asarray(x, dtype=float)
            try:
                fx = func(x)
            except Exception:
                return self.best_f
                
            if np.isnan(fx) or np.isinf(fx):
                return self.best_f
                
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Random Sampling to identify basins
        num_samples = 10
        candidates = []
        for _ in range(num_samples):
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break
            x0 = rng.uniform(-5, 5, self.dim)
            fx = wrapped_func(x0)
            candidates.append((fx, x0))
        
        if not candidates:
            return self.best_x, self.best_f

        # Sort by fitness to pick best starting points for local search
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement with L-BFGS-B
        # Use top 3 candidates for restarts
        num_restarts = 3
        top_candidates = candidates[:num_restarts]
        
        remaining_budget = self.budget - self.calls
        if remaining_budget <= 0:
            return self.best_x, self.best_f
            
        # Allocate remaining budget to each restart
        budget_per_restart = remaining_budget // num_restarts
        if budget_per_restart < 10:
            budget_per_restart = 10

        for fx, x0 in top_candidates:
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break
            
            # Run L-BFGS-B
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * self.dim,
                               options={'maxiter': 100, 'maxfun': budget_per_restart,
                                        'ftol': 1e-8, 'gtol': 1e-5})
                # Update best (wrapper already did, but ensure consistency)
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

        return self.best_x, self.best_f

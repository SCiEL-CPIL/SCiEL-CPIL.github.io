import numpy as np
from scipy.optimize import minimize
import time
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_fx = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        def time_ok():
            return time.time() < deadline

        def tracked_func(x):
            if not time_ok() or self.call_count >= self.budget:
                raise ValueError("Budget or time exceeded")
            
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            
            self.call_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            
            if fx < self.best_fx:
                self.best_fx = fx
                self.best_x = x.copy()
            return fx

        dim = self.dim
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search
        # Use Sobol-like or random sampling to find a good starting point.
        # Since dim is small (5), random is fine, but Sobol is better.
        # We'll use a simple random sampling with a mix of center and random.
        
        n_global = 30
        candidates = []
        
        # Center
        candidates.append(np.zeros(dim))
        
        # Random points
        for _ in range(n_global - 1):
            candidates.append(rng.uniform(-5.0, 5.0, size=dim))
            
        # Evaluate all candidates
        for x0 in candidates:
            if not time_ok() or self.call_count >= self.budget:
                break
            try:
                fx = tracked_func(x0)
            except Exception:
                pass

        if self.best_x is None:
            self.best_x = np.zeros(dim)
            self.best_fx = 0.0

        # Phase 2: Local Optimization with L-BFGS-B
        # L-BFGS-B is very efficient for smooth functions like Rosenbrock.
        if time_ok() and self.call_count < self.budget - 10:
            try:
                # Use remaining budget for maxfun
                maxfun = self.budget - self.call_count - 5
                res = minimize(
                    tracked_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 100,
                        'maxfun': maxfun,
                        'ftol': 1e-15,
                        'gtol': 1e-12
                    }
                )
            except Exception:
                pass

        # Phase 3: Polish with Nelder-Mead if budget remains
        # Nelder-Mead can sometimes squeeze out more precision if L-BFGS-B stops early
        # due to numerical issues, though L-BFGS-B is usually better.
        # Let's try a small (1+1)-ES instead, it's more robust for final polish.
        if time_ok() and self.call_count < self.budget - 5 and self.best_x is not None:
            sigma = 0.1
            x = self.best_x.copy()
            fx = self.best_fx
            for _ in range(20):
                if not time_ok() or self.call_count >= self.budget - 2:
                    break
                # (1+1)-ES step
                step = sigma * rng.normal(size=dim)
                x_new = np.clip(x + step, -5.0, 5.0)
                try:
                    fx_new = tracked_func(x_new)
                except Exception:
                    break
                
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    # Update best
                    if fx < self.best_fx:
                        self.best_fx = fx
                        self.best_x = x.copy()
                    # Adapt sigma
                    sigma = max(sigma * 1.5, 1e-6)
                else:
                    sigma = max(sigma * 0.7, 1e-6)

        if self.best_x is None:
            self.best_x = np.zeros(dim)
            self.best_fx = 0.0
            
        return self.best_x, self.best_fx

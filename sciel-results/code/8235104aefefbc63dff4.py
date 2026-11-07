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
        
        # Phase 1: Start from Origin
        # The minimizer is in [-4, 4]^5. Origin is a good prior.
        # We skip global random sampling to preserve budget for convergence.
        
        x_start = np.zeros(dim)
        
        # Evaluate start point
        try:
            fx_start = tracked_func(x_start)
        except Exception:
            # If origin fails (shouldn't happen), try a random point
            x_start = rng.uniform(-2.0, 2.0, size=dim)
            try:
                fx_start = tracked_func(x_start)
            except Exception:
                return np.zeros(dim), np.inf

        # Phase 2: Local Optimization with L-BFGS-B
        # L-BFGS-B is efficient for smooth functions.
        # We give it most of the budget.
        if time_ok() and self.call_count < self.budget - 10:
            try:
                maxfun = self.budget - self.call_count - 10
                res = minimize(
                    tracked_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 200,
                        'maxfun': maxfun,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'maxls': 50
                    }
                )
            except Exception:
                pass

        # Phase 3: Polish with Nelder-Mead
        # Nelder-Mead is derivative-free and can refine the solution
        # if L-BFGS-B stopped due to gradient tolerance but not function tolerance.
        if time_ok() and self.call_count < self.budget - 5 and self.best_x is not None:
            try:
                max_nelder_calls = self.budget - self.call_count - 2
                if max_nelder_calls > 10:
                    # Initialize simplex around best_x
                    # Standard scipy Nelder-Mead handles initialization, 
                    # but we can pass x0.
                    res_nm = minimize(
                        tracked_func,
                        self.best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': max_nelder_calls,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
            except Exception:
                pass

        if self.best_x is None:
            self.best_x = np.zeros(dim)
            self.best_fx = 0.0
            
        return self.best_x, self.best_fx

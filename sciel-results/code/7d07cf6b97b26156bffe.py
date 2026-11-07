import numpy as np
import time
import scipy.optimize as so

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        # Deadline guard: 300s limit, keep 20s margin for safety
        self.deadline = self.start_time + 280.0

    def _check_time(self):
        return time.time() < self.deadline

    def __call__(self, func):
        # Wrap func to track calls and handle safety
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            
            x = np.asarray(x, dtype=np.float64)
            x_clipped = np.clip(x, -5, 5)
            
            try:
                f = float(func(x_clipped))
            except Exception:
                return 1e100
                
            self.calls += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x_clipped.copy()
                
            return f

        x0 = np.zeros(self.dim)
        bounds = [(-5, 5)] * self.dim

        # Phase 1: Differential Evolution for global exploration
        # DE is robust for multimodal landscapes like Lunacek.
        # We use a subset of the budget here.
        max_feasible_de = max(50, int(self.budget * 0.6))
        
        try:
            # tol=1e-8, maxiter=None means it runs until maxfev is reached
            result = so.differential_evolution(
                safe_func,
                bounds,
                maxiter=1000,
                maxfev=max_feasible_de,
                tol=1e-12,
                seed=42,
                polish=False, # We will do our own polish to control budget
                init='sobol', # Better initial spread than 'latinhypercube' or 'random'
                popsize=15
            )
            
            if result.x is not None:
                if self.best_x is None or result.fun < self.best_f:
                    self.best_x = result.x.copy()
                    self.best_f = result.fun
        except Exception:
            pass

        # Phase 2: Local Polish with Nelder-Mead
        # Use the remaining budget to refine the best solution found by DE.
        remaining_budget = self.budget - self.calls
        if remaining_budget > 10 and self._check_time():
            try:
                result_nm = so.minimize(
                    safe_func,
                    self.best_x,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 1000, 'maxfev': remaining_budget}
                )
                
                if result_nm.x is not None:
                    if self.best_x is None or result_nm.fun < self.best_f:
                        self.best_x = result_nm.x.copy()
                        self.best_f = result_nm.fun
            except Exception:
                pass

        # Phase 3: Final Random Walk Polish (safety net)
        # If we still have budget left (e.g., DE converged early but we didn't use all fevs)
        # and Nelder-Mead didn't run or used little, do a simple random walk.
        # This ensures we don't waste potential if the previous steps converged too fast.
        
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = safe_func(self.best_x)
            
        x = self.best_x.copy()
        f = self.best_f
        sigma = 0.1

        while self.calls < self.budget and self._check_time():
            z = np.random.standard_normal(self.dim)
            x_new = x + sigma * z
            x_new = np.clip(x_new, -5, 5)
            
            f_new = safe_func(x_new)
            
            if f_new < f:
                x = x_new
                f = f_new
                sigma *= 1.1
            else:
                sigma *= 0.9
                if sigma < 1e-5:
                    sigma = 0.05

        if f < self.best_f:
            self.best_x = x
            self.best_f = f

        return self.best_x, self.best_f

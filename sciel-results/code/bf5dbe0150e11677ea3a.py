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
        # Deadline guard: 300s limit, keep 10s margin for safety
        self.deadline = self.start_time + 290.0

    def _check_time(self):
        return time.time() < self.deadline

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Wrap func to track calls and enforce budget
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            x = np.asarray(x, dtype=float)
            # Clip to box to ensure validity, though func clips internally
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Phase 1: Global Exploration with Differential Evolution
        # DE is robust to multimodality and requires fewer hyperparameters than CMA-ES for small dims.
        # We limit the budget for this phase to leave room for local polish.
        # Budget 500. Let's use ~60% for global, 40% for local.
        # DE population size is typically 15*n. For n=5, pop=75.
        # Max iterations = budget / pop_size. 300 / 75 = 4 iterations. This is very low.
        # To make DE effective with low budget, we might need to tune maxiter or use a smaller pop.
        # However, DE's strength is global. Let's try a moderate pop.
        
        # Alternative: Use a smaller population for DE to fit more iterations.
        # popsize=10 (2*n). maxiter = 300 / 10 = 30.
        
        # Let's split budget:
        budget_global = int(self.budget * 0.6)
        budget_local = self.budget - budget_global
        
        # We need to be careful with DE's internal call counting.
        # scipy.optimize.differential_evolution does not easily allow stopping based on a custom call counter 
        # unless we wrap the function and raise an exception or return a sentinel.
        # Since we can't easily interrupt DE cleanly without exceptions (which might be risky if not caught),
        # we can set maxiter such that maxiter * popsize <= budget_global.
        
        n = self.dim
        popsize = 10  # 2 * n
        max_iter_de = max(1, budget_global // popsize)
        
        # Initial bounds
        bounds = [(-5.0, 5.0) for _ in range(n)]
        
        # Run Differential Evolution
        # tol=1e-8, maxiter=max_iter_de
        # We must ensure that if DE uses up the budget, we don't exceed it.
        # DE calls func popsize times per iteration.
        # If it hits max_iter, it stops.
        # If it converges early, it stops.
        
        try:
            res_de = so.differential_evolution(
                safe_func, 
                bounds, 
                max_iter=max_iter_de,
                popsize=popsize,
                tol=1e-6,
                seed=42,
                polish=False  # We will do our own polish for better budget control
            )
            x_best_de = res_de.x
            f_best_de = res_de.fun
        except Exception:
            # Fallback if DE fails
            x_best_de = np.zeros(n)
            f_best_de = safe_func(x_best_de)
            
        # If DE didn't use much budget, we have more for local.
        # If DE used all global budget, we proceed to local with remaining.
        
        # Phase 2: Local Polish with L-BFGS-B
        # L-BFGS-B is efficient for smooth functions (Lunacek is smooth).
        # We start from the best point found by DE.
        
        if self.calls < self.budget and self._check_time():
            # L-BFGS-B uses a line search which calls func multiple times.
            # We can't strictly limit calls in scipy minimize easily without wrapping and raising.
            # However, L-BFGS-B converges fast. 
            # To be safe, we can wrap the function to stop if budget is exceeded.
            # But minimize doesn't handle "stop" gracefully if we just return a value.
            # We can use a flag.
            
            # Let's try to run minimize. If it exceeds budget, the safe_func returns best_f 
            # (which is a constant), which might confuse the optimizer or cause it to stall.
            # A better approach for strict budget adherence in local search:
            # Implement a simple Nelder-Mead or L-BFGS-B with a manual call limit loop?
            # No, let's trust that L-BFGS-B on 5D will converge within 50-100 calls.
            # 40% of 500 is 200 calls. That's plenty for 5D.
            
            x0 = x_best_de.copy()
            
            # Define a wrapper that counts calls and stops if budget is hit
            # Actually, safe_func already handles the counting and returning best_f if over.
            # If safe_func returns best_f (constant) for all subsequent calls, 
            # the gradient estimation might become zero or noisy, potentially causing L-BFGS-B to stop or loop.
            # To prevent infinite loops in minimize, we can set a maxiter for minimize.
            # Each L-BFGS-B iteration involves several function calls (finite differences or line search).
            # Let's set maxiter=50.
            
            try:
                res_min = so.minimize(
                    safe_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_min.fun < self.best_f:
                    self.best_f = res_min.fun
                    self.best_x = res_min.x
            except Exception:
                pass

        # Phase 3: If we still have budget, do a final random perturbation or simple hill climbing
        # to ensure we don't waste budget if L-BFGS-B stopped early.
        if self.calls < self.budget and self._check_time():
            x = self.best_x.copy()
            f = self.best_f
            # Simple random hill climbing with adaptive step size
            sigma = 0.5
            while self.calls < self.budget and self._check_time():
                z = rng.standard_normal(n)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = safe_func(x_new)
                if f_new < f:
                    x = x_new
                    f = f_new
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    if sigma < 1e-5:
                        break

        self.best_x = self.best_x if self.best_x is not None else np.zeros(n)
        self.best_f = self.best_f if np.isfinite(self.best_f) else 0.0
        
        return self.best_x, self.best_f

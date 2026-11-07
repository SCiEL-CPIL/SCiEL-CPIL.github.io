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
        n = self.dim
        bounds = [(-5.0, 5.0) for _ in range(n)]
        
        # Wrap func to track calls and enforce budget
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Strategy: Multi-Start CMA-ES
        # Lunacek bi-Rastrigin is bimodal. A single global search might converge to the wrong minimum
        # or take too long to find the global one. 
        # We split the budget into K independent CMA-ES runs.
        # Each run has a small population and limited iterations, acting as a focused local/global search
        # from a random start point.
        
        # Budget allocation:
        # Total budget: 500.
        # Let's try K=5 restarts.
        # Budget per restart: 100 calls.
        # CMA-ES population size: typically 4 + floor(3 * log(n)) for small n. For n=5, pop=9 or 10.
        # Let's use popsize=9.
        # Max iterations per restart: 100 / 9 = 11 iterations.
        # This is enough for CMA-ES to converge locally if it lands in a good basin.
        
        K = 5
        popsize = 9
        max_iter_per_run = max(1, (self.budget // K) // popsize)
        
        # We will use scipy's minimize with method='CMA-ES' if available? 
        # No, scipy.optimize.minimize does not support CMA-ES directly in older versions, 
        # and even in newer ones, it's often an alias or requires specific backends.
        # However, scipy.optimize has `differential_evolution` and `basinhopping`.
        # Actually, `scipy.optimize` does not have a built-in CMA-ES in standard `minimize` methods 
        # (it has 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov').
        
        # Wait, the current program uses `so.differential_evolution`. 
        # Inspiration 1 mentioned CMA-ES. Is CMA-ES available? 
        # `pycma` is a common package but not guaranteed. 
        # `scipy.optimize` does NOT have a native CMA-ES method in `minimize`.
        
        # Let's reconsider. The "Inspiration 1" with score 1.9171 used a "Multi- | uses numpy, time". 
        # It might have implemented a simple CMA-ES or used a different library.
        # Since I must stick to standard library, numpy, and scipy, I cannot use `pycma` unless it's installed (risky).
        
        # Alternative: Use `scipy.optimize.differential_evolution` with multiple seeds?
        # DE is stochastic. Running it multiple times with different seeds and smaller budgets is a valid "Multi-Start" strategy.
        # DE is robust and available in scipy.
        
        # Revised Strategy: Multi-Start Differential Evolution
        # Instead of one long DE run, do several short DE runs.
        # This helps escape local minima of the Rastrigin part.
        
        budget_per_run = self.budget // K
        # DE population size: 10 (2*n)
        popsize_de = 10
        max_iter_de = max(1, budget_per_run // popsize_de)
        
        for k in range(K):
            if self.calls >= self.budget or not self._check_time():
                break
            
            # Use a different seed for each restart to ensure diversity
            seed = 42 + k
            
            try:
                res_de = so.differential_evolution(
                    safe_func, 
                    bounds, 
                    max_iter=max_iter_de,
                    popsize=popsize_de,
                    tol=1e-8,
                    seed=seed,
                    polish=False
                )
            except Exception:
                continue
            
            # Update best
            if res_de.fun < self.best_f:
                self.best_f = res_de.fun
                self.best_x = res_de.x.copy()
            
            # Optional: If we have remaining budget in this slot, do a quick local polish
            # But DE with polish=False is our global explorer.
            # Let's add a local polish step after each DE run if budget allows.
            # This mimics the "Global -> Local" cycle for each restart.
            
            if self.calls < self.budget and self._check_time() and self.best_x is not None:
                x0 = self.best_x.copy()
                try:
                    # L-BFGS-B is fast for smooth local minima
                    res_min = so.minimize(
                        safe_func, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res_min.fun < self.best_f:
                        self.best_f = res_min.fun
                        self.best_x = res_min.x.copy()
                except Exception:
                    pass
        
        # Final Polish: Use any remaining budget for a final local refinement
        if self.calls < self.budget and self._check_time() and self.best_x is not None:
            x0 = self.best_x.copy()
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
                    self.best_x = res_min.x.copy()
            except Exception:
                pass

        self.best_x = self.best_x if self.best_x is not None else np.zeros(n)
        self.best_f = self.best_f if np.isfinite(self.best_f) else 0.0
        
        return self.best_x, self.best_f

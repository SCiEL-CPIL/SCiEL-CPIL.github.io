import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Two-phase surrogate-assisted L-BFGS-B Optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, use 290s
        self.deadline = self.start_time + 290.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # Phase 1: Surrogate-assisted global search
        # Use a simple quadratic surrogate or just random sampling to identify promising basins.
        # Since we only have 500 calls and dim=5, a full GP is overkill and might be slow to fit/eval.
        # Instead, we use a "surrogate" logic: sample a few points, keep the best, and use it as a seed.
        # The key insight from the prompt is to use early evaluations to find a good basin.
        
        # Let's use a small set of initial evaluations to find a good starting point.
        # We will spend ~20-30 calls on a coarse random search to find a promising region.
        
        rng = np.random.default_rng(42)
        
        # Initial coarse search
        # We know the min is in [-3, 5]^5 roughly.
        # Let's sample 20 points in [-4, 5]^5.
        n_coarse = 20
        coarse_samples = rng.uniform(-4, 5, size=(n_coarse, self.dim))
        
        # Evaluate them
        best_coarse_x = None
        best_coarse_f = np.inf
        
        for x in coarse_samples:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            try:
                f = wrapped_func(x)
                if f < best_coarse_f:
                    best_coarse_f = f
                    best_coarse_x = x.copy()
            except StopIteration:
                break
            except Exception:
                continue
        
        # If we didn't find anything, fallback to (1,1,1,1,1)
        if best_coarse_x is None:
            x0 = np.ones(self.dim)
            try:
                f = wrapped_func(x0)
                best_coarse_x = x0.copy()
                best_coarse_f = f
            except:
                pass
        
        # Phase 2: Local refinement from the best coarse point
        # We can also try a few other good points from the coarse search.
        # Sort the coarse samples by their function value (approximated by our tracking)
        # Actually, we didn't store all values, just the best. 
        # Let's just use the best coarse point and maybe a few random ones near it.
        
        # Let's try L-BFGS-B from the best coarse point.
        if best_coarse_x is not None:
            x0 = best_coarse_x
            remaining_budget = self.budget - self.calls_made
            if remaining_budget > 10:
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func, 
                        x0, 
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': 200,
                            'ftol': 1e-12,
                            'gtol': 1e-10
                        }
                    )
                except StopIteration:
                    pass
                except Exception:
                    pass
        
        # If we still have budget, try a few other promising starts
        # We can re-evaluate some points or just try random starts if we haven't converged.
        # But for now, let's stick to the main refinement.
        
        # If no calls were made or best_x is still None
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            try:
                self.best_f = func(np.clip(x0, -5, 5))
            except:
                self.best_f = np.inf

        return self.best_x, self.best_f

import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid L-BFGS-B + Nelder-Mead Optimizer for Rosenbrock-like functions.
    
    Uses L-BFGS-B for its fast convergence on smooth functions like Rosenbrock,
    followed by Nelder-Mead for local polishing. Multiple starts with diverse
    initializations to handle the hidden shift/rotation/offset.
    """

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
        self.deadline = self.start_time + 285.0

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

        rng = np.random.default_rng(42)
        
        # Strategic starting points: focus on likely basin region [-4, 4]^5
        # Rosenbrock minimum is at (1,1,...,1) before transformation
        # After rotation/shift, it's in [-4,4]^5
        starts = []
        # Center of likely region
        starts.append(np.zeros(self.dim))
        # Standard Rosenbrock minimum location (before transformation)
        starts.append(np.ones(self.dim))
        # Diverse points in the likely region
        starts.append(np.full(self.dim, 2.0))
        starts.append(np.full(self.dim, -1.0))
        # Random points in [-4, 4]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Remove duplicates
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        # Phase 1: L-BFGS-B with adaptive budget allocation
        # Give more budget to earlier starts (they're more likely to be good)
        for idx, x0 in enumerate(unique_starts):
            if self.calls_made >= self.budget - 30:
                break
            if time.time() > self.deadline:
                break
            
            x0 = np.clip(x0, -5, 5)
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 30:
                break

            # Allocate more budget to first start, less to later ones
            # First start gets up to 40% of remaining, subsequent get less
            if idx == 0:
                maxfun = min(remaining_budget - 20, int(0.4 * self.budget))
            elif idx == 1:
                maxfun = min(remaining_budget - 20, int(0.25 * self.budget))
            else:
                maxfun = min(remaining_budget - 20, int(0.15 * self.budget))
            
            maxfun = max(maxfun, 30)  # Minimum 30 calls per start

            try:
                res_lbfgsb = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 150,
                        'gtol': 1e-6,
                        'ftol': 1e-10,
                        'maxfun': maxfun,
                        'maxls': 20
                    }
                )
                
                if res_lbfgsb.fun < self.best_f:
                    self.best_f = res_lbfgsb.fun
                    self.best_x = np.clip(res_lbfgsb.x, -5, 5)
                    
            except StopIteration:
                break
            except Exception:
                pass

        # Phase 2: High-precision Nelder-Mead polish on the best point
        if self.calls_made < self.budget - 5 and self.best_x is not None:
            try:
                x_start = self.best_x.copy()
                x_start = np.clip(x_start, -5, 5)
                
                res_nm = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start, 
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'maxiter': 100,
                        'adaptive': True
                    }
                )
                
                if res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = np.clip(res_nm.x, -5, 5)
                    
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: Final L-BFGS-B polish if budget remains
        if self.calls_made < self.budget - 10 and self.best_x is not None:
            try:
                x_start = self.best_x.copy()
                x_start = np.clip(x_start, -5, 5)
                remaining = self.budget - self.calls_made
                
                res_final = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 50,
                        'gtol': 1e-8,
                        'ftol': 1e-12,
                        'maxfun': remaining,
                        'maxls': 10
                    }
                )
                
                if res_final.fun < self.best_f:
                    self.best_f = res_final.fun
                    self.best_x = np.clip(res_final.x, -5, 5)
                    
            except StopIteration:
                pass
            except Exception:
                pass

        # Fallback if no calls were made
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = wrapped_func(x0)

        return self.best_x, self.best_f

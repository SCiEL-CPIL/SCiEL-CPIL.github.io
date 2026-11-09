import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Two-phase optimizer: fast coarse L-BFGS-B exploration + Powell valley refinement.
    
    Phase 1 runs many short L-BFGS-B optimizations from diverse starts to quickly
    locate the Rosenbrock valley. Phase 2 uses Powell's method, which is specifically
    effective for Rosenbrock's narrow valley structure, for precise convergence.
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
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = float(func(x))
            if not np.isfinite(val):
                val = 1e10
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # Priority-ordered starts: standard Rosenbrock min first, then perturbations
        starts = [np.ones(self.dim)]
        
        # Small perturbations around (1,...,1) - most likely region for shifted min
        for scale in [0.2, 0.5, 1.0, 2.0]:
            starts.append(np.ones(self.dim) + rng.normal(0, scale, self.dim))
        
        # Additional structured points covering [-4, 4]^5
        for val in [-3.0, -2.0, -1.0, 0.0, 2.0, 3.0]:
            starts.append(np.full(self.dim, val))
        
        # Random in likely region
        for _ in range(6):
            starts.append(rng.uniform(-4, 4, self.dim))
        
        # Few in full box
        for _ in range(2):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        # Remove duplicates
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        # Phase 1: Fast coarse search - many short L-BFGS-B runs
        # Short maxiter so we quickly get to the valley, not stuck in one long run
        for x0 in unique_starts:
            if self.calls_made >= self.budget - 100:
                break
            if time.time() > self.deadline:
                break
            
            x0 = np.clip(x0, -5, 5)
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 30:
                break

            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 25,
                        'gtol': 1e-4,
                        'ftol': 1e-10,
                        'maxfun': min(remaining_budget, 60)
                    }
                )
                
                if np.isfinite(res.fun) and res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
                    
            except StopIteration:
                break
            except Exception:
                pass

        # Phase 2: Powell refinement - excellent for Rosenbrock valley structure
        if self.calls_made < self.budget - 20 and self.best_x is not None:
            try:
                x_start = np.clip(self.best_x.copy(), -5, 5)
                
                res_powell = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start, 
                    method='Powell',
                    options={
                        'maxiter': 200,
                        'xtol': 1e-8,
                        'ftol': 1e-10,
                        'maxfev': self.budget - self.calls_made - 10
                    }
                )
                
                if np.isfinite(res_powell.fun) and res_powell.fun < self.best_f:
                    self.best_f = res_powell.fun
                    self.best_x = np.clip(res_powell.x, -5, 5)
                    
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: Final Nelder-Mead polish if budget allows
        if self.calls_made < self.budget - 5 and self.best_x is not None and self.best_f < 1e4:
            try:
                x_start = np.clip(self.best_x.copy(), -5, 5)
                
                res_nm = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start, 
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'maxiter': 30,
                        'adaptive': True
                    }
                )
                
                if np.isfinite(res_nm.fun) and res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = np.clip(res_nm.x, -5, 5)
                    
            except StopIteration:
                pass
            except Exception:
                pass
        
        # Fallback if no calls were made
        if self.best_x is None:
            x0 = np.ones(self.dim)
            try:
                self.best_f = wrapped_func(x0)
                self.best_x = x0.copy()
            except Exception:
                self.best_x = x0
                self.best_f = 1e10

        return self.best_x, self.best_f

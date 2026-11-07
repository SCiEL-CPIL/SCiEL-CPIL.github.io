import numpy as np
import scipy.optimize as so
import time

class Optimizer:
    """Optimizer using structured initialization and L-BFGS-B / Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        # Use a safer deadline. Evaluator kills at 330s. 250s is safe.
        self.deadline = 250.0

    def _check_time(self):
        if time.time() > self.deadline:
            raise TimeoutError("Time limit exceeded")

    def _eval(self, func, x):
        self._check_time()
        # Clip to box
        x = np.clip(x, -5.0, 5.0)
        self.calls += 1
        if self.calls > self.budget:
            raise ValueError("Budget exceeded")
        
        try:
            fx = func(x)
            # Handle non-finite values
            if not np.isfinite(fx):
                fx = float('inf')
        except Exception:
            fx = float('inf')
        
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None

        # Phase 1: Initial Exploration
        candidates = []
        rng = np.random.default_rng(42)
        
        # Structured points
        candidates.append(np.zeros(self.dim))
        candidates.append(np.ones(self.dim))
        candidates.append(-np.ones(self.dim))
        
        # Random uniform in full box
        for _ in range(30):
            candidates.append(rng.uniform(-5.0, 5.0, self.dim))
            
        # Random uniform in smaller box (denser near origin)
        for _ in range(20):
            candidates.append(rng.uniform(-2.0, 2.0, self.dim))

        # Evaluate all candidates
        for x in candidates:
            try:
                self._eval(func, x)
            except (TimeoutError, ValueError):
                break
            except Exception:
                # If evaluation fails unexpectedly, stop exploration
                break

        if self.best_x is None:
            # Fallback: if no valid evaluation happened, try zero vector manually
            try:
                self.best_x = np.zeros(self.dim)
                self.best_f = self._eval(func, self.best_x)
            except Exception:
                self.best_x = np.zeros(self.dim)
                self.best_f = float('inf')

        # Phase 2: Local Optimization with L-BFGS-B
        remaining_calls = self.budget - self.calls
        if remaining_calls > 0 and self.best_x is not None:
            def scipy_func(x):
                return self._eval(func, x)
            
            bounds = [(-5.0, 5.0)] * self.dim
            
            try:
                # L-BFGS-B is efficient for smooth bounded problems
                # maxfun limits the number of function evaluations
                res = so.minimize(scipy_func, self.best_x, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 100, 'maxfun': remaining_calls})
                if res.success and np.isfinite(res.fun):
                    self.best_x = res.x
                    self.best_f = res.fun
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Phase 3: Polish with Nelder-Mead
        remaining_calls = self.budget - self.calls
        if remaining_calls > 20 and self.best_x is not None:
            def scipy_func_nm(x):
                return self._eval(func, x)
            
            try:
                # Nelder-Mead for final polish
                res_nm = so.minimize(scipy_func_nm, self.best_x, method='Nelder-Mead',
                                     options={'maxiter': 50, 'maxfev': remaining_calls, 'xatol': 1e-8, 'fatol': 1e-8})
                if res_nm.success and np.isfinite(res_nm.fun):
                    self.best_x = res_nm.x
                    self.best_f = res_nm.fun
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Ensure best_x is a valid numpy array
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)

        return self.best_x, self.best_f

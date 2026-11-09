import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280  # Safety margin before 300s limit

    def _halton_seq(self, base, n):
        """Generate the first n terms of the Halton sequence for a given base."""
        f = np.zeros(n)
        for i in range(1, n + 1):
            f[i - 1] = 0
            denom = base
            num = i
            while num > 0:
                denom *= base
                num, rem = divmod(num, base)
                f[i - 1] += rem / denom
        return f

    def _halton_points(self, dim, n):
        """Generate n Halton points in [0, 1]^dim."""
        seqs = []
        for i in range(1, dim + 1):
            seqs.append(self._halton_seq(i, n))
        return np.array(seqs).T

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        # Wrap func to count calls and check deadline
        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            val = func(x)
            if np.isnan(val) or np.isinf(val):
                val = 1e10
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Generate diverse starting points using Halton sequence for better coverage
        # 10 Halton points + Origin = 11 points
        n_starts = 10
        halton_unit = self._halton_points(dim, n_starts)
        # Map [0, 1] to [-5, 5]
        starts = halton_unit * 10.0 - 5.0
        
        # Prepend origin
        starts = np.vstack([np.zeros(dim), starts])

        # Phase 1: Evaluate all Halton starts to identify promising regions
        start_values = []
        for s in starts:
            if self.calls >= budget - 50: # Reserve budget for local optimization
                break
            try:
                val = wrapped_func(s)
                start_values.append((val, s))
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        if not start_values:
            # Fallback if even initial evaluation failed
            if self.best_x is None:
                self.best_x = np.zeros(dim)
                try:
                    self.best_f = wrapped_func(self.best_x)
                except:
                    self.best_f = 1e10
            return self.best_x, self.best_f

        # Sort by value to prioritize local optimization on the best candidates
        start_values.sort(key=lambda x: x[0])
        
        # Phase 2: Nelder-Mead polishing from the best starting points
        # Nelder-Mead is derivative-free and adapts its simplex to the local
        # curvature, effectively acting as a quasi-Newton method without
        # explicit gradient estimation, which is robust for rotated Rosenbrock.
        
        # Try NM on the top 4 starts
        for val, s in start_values[:4]:
            if self.calls >= budget - 20 or time.time() > self.deadline:
                break
            
            try:
                res_nm = minimize(
                    wrapped_func, 
                    s, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                # Update best is handled by wrapped_func
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # Phase 3: Final Polish from the absolute best found so far
        if self.calls < budget - 10 and time.time() < self.deadline and self.best_x is not None:
            try:
                res_nm = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # If no valid result was found, evaluate origin
        if self.best_x is None:
             self.best_x = np.zeros(dim)
             try:
                 self.best_f = wrapped_func(self.best_x)
             except:
                 self.best_f = 1e10

        return self.best_x, self.best_f

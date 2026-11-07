import numpy as np
import scipy
import time

class BudgetExhausted(Exception):
    pass

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = 280.0  # Safety margin before 330s kill

        def wrapped_func(x):
            if time.time() - self.start_time > deadline:
                raise TimeoutError()
            if self.call_count >= self.budget:
                raise BudgetExhausted()
            
            self.call_count += 1
            x = np.asarray(x, dtype=float)
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Stratified initialization
        n_starts = 10
        try:
            sampler = scipy.stats.qmc.LatinHypercube(d=self.dim, seed=42)
            points = sampler.random(n_starts)
            # Scale to [-4, 4]
            points = points * 8.0 - 4.0
        except Exception:
            points = np.random.uniform(-4, 4, size=(n_starts, self.dim))

        # Local search phase
        # Use Powell for initial refinement as it handles coordinate directions well
        for p in points:
            if self.call_count >= self.budget:
                break
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    p, 
                    method='Powell',
                    options={'maxiter': 10, 'ftol': 1e-6}
                )
            except (TimeoutError, BudgetExhausted):
                break
            except Exception:
                continue

        # Final polish on the best point found
        if self.best_x is not None and self.call_count < self.budget - 10:
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
                )
            except (TimeoutError, BudgetExhausted):
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

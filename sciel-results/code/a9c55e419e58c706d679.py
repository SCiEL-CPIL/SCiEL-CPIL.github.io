import numpy as np
from scipy.optimize import basinhopping

class BudgetExceeded(Exception):
    pass

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_f = np.inf
        self.best_x = np.zeros(dim)

    def __call__(self, func):
        self.calls = 0
        self.best_f = np.inf
        self.best_x = np.zeros(self.dim)

        def counted_func(x):
            self.calls += 1
            if self.calls > self.budget:
                raise BudgetExceeded()
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Use a deterministic seed for reproducibility of the search path
        # though the function itself is seeded by the evaluator.
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, self.dim)
        
        try:
            basinhopping(
                counted_func,
                x0,
                stepsize=1.0,
                minimizer_kwargs={
                    'method': 'Nelder-Mead',
                    'maxiter': 50,
                    'xatol': 1e-6,
                    'fatol': 1e-6
                },
                niter=1000,
                kick_mode='random',
                seed=42
            )
        except BudgetExceeded:
            pass
        except Exception:
            pass

        return self.best_x, self.best_f

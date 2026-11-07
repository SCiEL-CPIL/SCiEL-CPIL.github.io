import numpy as np
import scipy
import time

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
        deadline = 280.0  # safety margin before 330s kill

        def wrapped_func(x):
            if time.time() - self.start_time > deadline:
                raise TimeoutError()
            if self.call_count >= self.budget:
                raise BudgetExhausted()
            self.call_count += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = np.array(x, dtype=float).copy()
            return fx

        bounds = [(-4.0, 4.0)] * self.dim

        # Phase 1: DE for global exploration (~100 calls)
        # popsize=8, maxiter=3 -> 8 + 3*8 = 32... too few
        # Let's do popsize=10, maxiter=5 -> 10 + 5*10 = 60 calls
        # Actually DE with popsize p and maxiter m uses p*(1+m) calls
        # popsize=8, maxiter=8 -> 8*9 = 72 calls. Good.
        try:
            res = scipy.optimize.differential_evolution(
                wrapped_func,
                bounds,
                popsize=8,
                maxiter=8,
                polish=False,
                seed=42,
                tol=0.05,
                mutation=(0.4, 1.0),
                recombination=0.7
            )
        except (TimeoutError, BudgetExhausted):
            pass

        # If we hit budget or timeout during DE, we're done
        if self.call_count >= self.budget or time.time() - self.start_time > deadline:
            return self.best_x, self.best_f

        # Phase 2: (1+1)-ES for mid-range convergence
        # Use remaining budget minus ~150 for NM polish
        es_budget = max(50, self.budget - self.call_count - 150)
        rng = np.random.default_rng(123)
        
        # Start ES from best_x found by DE
        if self.best_x is not None:
            x = self.best_x.copy()
        else:
            x = rng.uniform(-4, 4, self.dim)
            fx = wrapped_func(x)
            self.best_x = x.copy()
            self.best_f = fx
        
        sigma = 0.5
        for _ in range(es_budget):
            if time.time() - self.start_time > deadline:
                break
            if self.call_count >= self.budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -4, 4)
            fy = wrapped_func(y)
            if fy < self.best_f:
                x, self.best_f = y, fy
                self.best_x = x.copy()
                sigma *= 1.3
            else:
                sigma *= 0.85

        if self.call_count >= self.budget or time.time() - self.start_time > deadline:
            return self.best_x, self.best_f

        # Phase 3: Nelder-Mead polish for final precision
        # NM will make many calls; we need to wrap it carefully
        # Use scipy.optimize.minimize with Nelder-Mead
        try:
            nm_result = scipy.optimize.minimize(
                wrapped_func,
                self.best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': self.budget - self.call_count,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
        except (TimeoutError, BudgetExhausted):
            pass

        return self.best_x, self.best_f

class BudgetExhausted(Exception):
    pass

class TimeoutError(Exception):
    pass

import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = None
        self.func = None

    def __call__(self, func):
        self.func = func
        self.start_time = time.time()
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        
        # 1. Initial diverse sampling
        n_initial = 20
        # Use LHS for diversity
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        points = sampler.random(n=n_initial)
        points = points * 10 - 5 # Scale to [-5, 5]
        
        # Evaluate initial points
        for p in points:
            self._evaluate(p)
        
        # 2. Local refinement of best points
        # Sort by function value, take top 3
        # We need to keep track of evaluated points to sort them
        # ... (Need to store history)
        
        # 3. Main loop: Nelder-Mead on best, with restarts
        # ...
        
        return self.best_x, self.best_f

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return
        if time.time() - self.start_time > 290: # Safety margin
            return
        x = np.clip(x, -5, 5)
        f = self.func(x)
        self.calls += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

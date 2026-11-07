"""Rewritten program using a hybrid (1+1)-ES and L-BFGS-B approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid optimizer: (1+1)-ES for initialization, L-BFGS-B for convergence."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep a 15s safety margin
        deadline = start_time + 285
        
        calls = 0
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            x = np.clip(x, -5, 5)
            val = float(func(x))
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: (1+1)-ES to find a good starting point in the valley
        # Use 20% of the budget for exploration
        es_budget = max(10, self.budget // 5)
        
        x = np.random.uniform(-5, 5, self.dim)
        fx = wrapper(x)
        sigma = 2.0
        
        for _ in range(es_budget - 1):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
            
            y = np.clip(x + sigma * np.random.standard_normal(self.dim), -5, 5)
            fy = wrapper(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.25
            else:
                sigma *= 0.9
                if sigma < 1e-5:
                    break

        # Phase 2: L-BFGS-B to converge to the minimum
        if calls < self.budget:
            remaining = self.budget - calls
            # L-BFGS-B with finite differences uses dim + 1 evaluations per step
            # Estimate max iterations
            max_iter = max(10, remaining // (self.dim + 1))
            
            try:
                # Use L-BFGS-B
                # bounds are provided to help the optimizer stay in the box
                bounds = [(-5, 5)] * self.dim
                
                res = minimize(
                    wrapper, 
                    x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': max_iter, 
                        'maxfev': remaining, 
                        'ftol': 1e-12, 
                        'gtol': 1e-8,
                        'disp': False
                    }
                )
                # wrapper updates best_x and best_f automatically
            except Exception:
                # If L-BFGS-B fails, we keep the best found in Phase 1
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

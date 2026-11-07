import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        def check_deadline():
            return time.time() > deadline or calls >= self.budget
        
        # Phase 1: Multi-start CMA-ES for global exploration
        # Start from multiple diverse points to handle the shifted/rotated landscape
        
        # Initial candidate points: origin, nominal Rosenbrock minimum, and random points
        candidate_starts = [np.zeros(self.dim), np.ones(self.dim)]
        # Add a few random points
        rng = np.random.default_rng(42)
        for _ in range(3):
            candidate_starts.append(rng.uniform(-4, 4, size=self.dim))
        
        # Run CMA-ES from each start, using a portion of the budget
        es_budget_per_start = max(20, int(self.budget * 0.4 / len(candidate_starts)))
        
        for start_point in candidate_starts:
            if check_deadline():
                break
            
            x0 = np.clip(start_point, -5, 5)
            f0 = func(x0)
            calls += 1
            update_best(x0, f0)
            
            if check_deadline():
                break
            
            # Run CMA-ES from this start
            def wrapper(x_eval):
                nonlocal calls
                if check_deadline():
                    raise Exception("Budget or time exceeded")
                calls += 1
                return func(x_eval)
            
            try:
                res = minimize(wrapper, x0, method='Powell',
                               options={'maxfev': es_budget_per_start, 'maxiter': 50})
            except Exception:
                pass
        
        # Phase 2: Fine-tuning from the best point found
        if calls < self.budget and best_x is not None and not check_deadline():
            remaining = self.budget - calls
            
            def wrapper2(x_eval):
                nonlocal calls
                if check_deadline():
                    raise Exception("Budget or time exceeded")
                calls += 1
                return func(x_eval)
            
            try:
                res = minimize(wrapper2, best_x, method='Powell',
                               options={'maxfev': remaining, 'maxiter': 200})
            except Exception:
                pass
        
        # Final safety: ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls += 1
        
        return best_x, best_f

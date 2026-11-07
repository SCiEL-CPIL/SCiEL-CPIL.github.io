import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """
    Optimizer using Random Sampling followed by Powell's Method.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        calls = 0
        best_x = None
        best_f = float('inf')
        
        # Deadline guard variables
        start_time = time.time()
        deadline = 320.0  # Safety margin from 330s limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            # Check deadline
            if time.time() > deadline:
                # Return a large value to signal stop, but we can't stop scipy immediately
                # We rely on maxfev to stop.
                pass
            
            if calls >= self.budget:
                return float('inf')
            
            calls += 1
            try:
                val = func(x)
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        # Phase 1: Random Exploration
        # Use a fraction of the budget for exploration
        # 20 random points is a good balance for dim=5
        n_explore = min(20, self.budget // 10)
        
        for _ in range(n_explore):
            if calls >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            wrapped_func(x)

        # Phase 2: Local Exploitation using L-BFGS-B
        # L-BFGS-B is effective for smooth functions like Rosenbrock
        if best_x is not None and calls < self.budget:
            remaining = self.budget - calls
            # Reserve some budget for potential polish
            polish_budget = 50
            main_budget = max(10, remaining - polish_budget)
            
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxls': 50,
                        'maxfev': main_budget
                    }
                )
            except Exception:
                pass

            # Phase 3: Polish Pass with Nelder-Mead
            # If we have budget left, try to refine the solution
            if calls < self.budget:
                remaining = self.budget - calls
                if remaining > 10:
                    try:
                        res = minimize(
                            wrapped_func, 
                            best_x, 
                            method='Nelder-Mead', 
                            options={
                                'maxiter': 100,
                                'xatol': 1e-10,
                                'fatol': 1e-10,
                                'maxfev': remaining
                            }
                        )
                    except Exception:
                        pass

        return best_x, best_f

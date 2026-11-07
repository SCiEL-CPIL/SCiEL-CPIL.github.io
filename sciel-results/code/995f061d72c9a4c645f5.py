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
                return float('inf')
            
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
        # Increase exploration to 50 points to better locate the basin of attraction
        # for the Rosenbrock function, which has a narrow valley.
        n_explore = min(50, self.budget // 10)
        
        for _ in range(n_explore):
            if calls >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            wrapped_func(x)

        # Phase 2: Local Exploitation using Powell's Method
        # Powell is effective for Rosenbrock as it adapts search directions
        if best_x is not None and calls < self.budget:
            remaining = self.budget - calls
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Powell', 
                    options={
                        'maxfev': remaining,
                        'xtol': 1e-10,
                        'ftol': 1e-10
                    }
                )
            except Exception:
                pass
        
        return best_x, best_f

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290
        rng = np.random.default_rng(12345)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget
        n = self.dim
        
        def can_continue():
            return calls_made < max_calls and time.time() < deadline
        
        def simple_es(x0, sigma, budget_remaining):
            nonlocal calls_made, best_x, best_f
            
            x = x0.copy()
            
            # Evaluate initial
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            for _ in range(budget_remaining - 1):
                if not can_continue():
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.15
                else:
                    sigma *= 0.92
                
                sigma = np.clip(sigma, 0.003, 5.0)
            
            return x, fx
        
        # Multi-start strategy
        n_starts = 4
        per_start = max(10, self.budget // n_starts)
        
        # Start 1: Center
        x0 = np.zeros(self.dim)
        simple_es(x0, 1.5, per_start)
        
        if can_continue():
            # Start 2: Random in [-3, 3]
            x0 = rng.uniform(-3, 3, self.dim)
            simple_es(x0, 1.0, per_start)
        
        if can_continue():
            # Start 3: Random in [-5, 5]
            x0 = rng.uniform(-5, 5, self.dim)
            simple_es(x0, 2.0, per_start)
        
        if can_continue():
            # Start 4: Use remaining budget with smaller step
            remaining = max_calls - calls_made
            if remaining > 10:
                x0 = rng.uniform(-4, 4, self.dim)
                simple_es(x0, 0.5, remaining)
        
        return best_x, best_f
# EVOLVE-BLOCK-END

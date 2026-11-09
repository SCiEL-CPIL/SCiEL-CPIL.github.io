"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES-like optimizer with adaptive step size for bent cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 320.0  # 300s limit, 20s margin
        
        # Wrapper to count calls and enforce limits
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(x)

        best_x = np.zeros(self.dim)
        best_fx = float('inf')
        
        # Initial point at origin
        x = np.zeros(self.dim)
        fx = eval_func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Adaptive step size
        sigma = 2.0
        rng = np.random.default_rng(42)
        
        # Simple (1+1) evolution strategy with adaptive sigma
        for _ in range(self.budget - 1):
            if time.time() - self.start_time > self.time_limit:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            try:
                fy = eval_func(y)
            except StopIteration:
                break
            except Exception:
                continue
            
            if fy < fx:
                # Accept move, increase step size slightly
                x, fx = y, fy
                sigma *= 1.1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Reject move, decrease step size
                sigma *= 0.95
                
            # Ensure sigma stays in reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)

        # Ensure we return something valid
        if not np.all(np.isfinite(best_x)):
            best_x = np.zeros(self.dim)
            if self.calls < self.budget:
                best_fx = eval_func(best_x)
            else:
                best_fx = 0.0
                
        return best_x, best_fx
# EVOLVE-BLOCK-END

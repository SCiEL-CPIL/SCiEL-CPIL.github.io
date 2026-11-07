"""GLOBAL-SEARCH: Use Differential Evolution for robust global optimization on multimodal functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    """Global optimizer using Differential Evolution."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        call_count = 0
        max_calls = self.budget
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def counted_func(x):
            nonlocal call_count, best_x, best_f
            # Time check
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            # Budget check
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            
            call_count += 1
            x = np.asarray(x, dtype=np.float64).ravel()[:self.dim]
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            return f

        bounds = [(-5.0, 5.0)] * self.dim
        
        # Configure Differential Evolution
        # popsize=15, maxiter=32 => ~495 evaluations
        # Latin Hypercube initialization for better global coverage
        try:
            differential_evolution(
                counted_func,
                bounds,
                maxiter=32,
                popsize=15,
                tol=1e-10,
                seed=42,
                init='latinhypercube'
            )
        except Exception:
            # If an exception occurs (e.g., budget exceeded), 
            # we return the best result found so far.
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import cma

# EVOLVE-BLOCK-START

class Optimizer:
    """CMA-ES optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # 330s limit, keep 10s margin
        
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise ValueError("Time limit exceeded")
            
            self.call_count += 1
            fx = func(x)
            
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            
            return fx

        # CMA-ES is highly effective for BBOB functions like Discus.
        # We start from the center of the box [0, 0, 0, 0, 0]
        # Initial sigma should be large enough to explore the [-5, 5] box.
        # A sigma of 2.0 covers a significant portion of the search space initially.
        
        x0 = np.zeros(self.dim)
        sigma0 = 2.0
        
        # CMA-ES options
        # maxiter is not the primary budget control; we use the wrapper to stop.
        # However, setting a reasonable maxiter prevents infinite loops if wrapper fails silently (unlikely).
        # popsize is typically ~4+2*ln(N) for N=5, so ~11. We can let CMA-ES choose or set it.
        
        options = {
            'maxiter': 1000,  # Internal iteration limit, but we stop via budget
            'verbose': -9,    # Suppress output
            'seed': 42,       # Deterministic seed for reproducibility if needed, but CMA-ES is stochastic
            'bounds': [-5.0, 5.0], # CMA-ES can handle bounds
            'CMA_adapt_sigma': 'auto', # Default
            'CMA_adaptation_of_alpha': True, # Adaptive step size
        }
        
        try:
            es = cma.CMAEvolutionStrategy(x0, sigma0, options)
            # es.optimize uses the function. It will call wrapped_func.
            # We pass wrapped_func directly.
            es.optimize(wrapped_func, maxfun=self.budget)
        except ValueError:
            # Budget or time exceeded
            pass
        except Exception:
            # Catch any other exceptions to be safe, though CMA-ES is robust
            pass

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END

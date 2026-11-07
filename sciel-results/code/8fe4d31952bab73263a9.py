import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # We will use a multi-start strategy.
        # Split budget into N restarts.
        # For dim=5, 25 restarts * 20 evals = 500 evals.
        n_restarts = 25
        evals_per_restart = self.budget // n_restarts
        
        best_x = None
        best_f = np.inf
        
        # Use a fixed seed for reproducibility if needed, but random is fine.
        rng = np.random.default_rng(42)
        
        # Generate starting points. 
        # To maximize diversity, we can use a grid or random.
        # Random uniform is simple.
        # Let's use a mix: some random, some on a coarse grid?
        # Actually, just random uniform is good enough for 25 points in 5D.
        
        for r in range(n_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # Initialize point
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Local (1+1)-ES
            sigma = 0.5  # Start with a moderate step
            # We have evals_per_restart - 1 evaluations left for this restart
            for i in range(evals_per_restart - 1):
                if time.time() > deadline:
                    break
                
                # Generate neighbor
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Adapt sigma: increase slightly on success
                    sigma *= 1.05 
                else:
                    # Adapt sigma: decrease on failure
                    sigma *= 0.95
                    
                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 1e-4, 5.0)

        return best_x, best_f

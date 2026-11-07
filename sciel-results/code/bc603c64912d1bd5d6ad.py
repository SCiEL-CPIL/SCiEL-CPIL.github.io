import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        budget = self.budget
        dim = self.dim

        calls = [0]
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(x):
            nonlocal best_x, best_f
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Global Exploration
        # Use a Latin Hypercube-like sampling or just uniform random for simplicity and robustness.
        # 20 initial samples to identify basins.
        n_init = 20
        initial_points = []
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            initial_points.append((x.copy(), f))
        
        # Sort initial points by fitness
        initial_points.sort(key=lambda t: t[1])

        # Phase 2: Multi-start Local Refinement using (1+1)-ES
        # We will attempt to refine the best K initial points.
        # If one gets stuck, we move to the next.
        
        max_evals_per_local_search = 100
        stagnation_limit = 30
        
        for i, (x0, f0) in enumerate(initial_points):
            if calls[0] >= budget or time.time() > deadline:
                break
            
            # Skip if this point is not better than the current global best
            if f0 >= best_f:
                continue
                
            x = x0.copy()
            f = f0
            sigma = 0.5
            no_improve_count = 0
            local_best_f = f
            
            # Run (1+1)-ES for this start point
            while calls[0] < budget and time.time() < deadline:
                if no_improve_count > stagnation_limit:
                    break
                
                # Generate candidate
                step = rng.standard_normal(dim) * sigma
                x_new = x + step
                x_new = np.clip(x_new, -5, 5)
                
                f_new = evaluate(x_new)
                
                if f_new < f:
                    # Accept
                    x = x_new
                    f = f_new
                    sigma *= 1.1
                    no_improve_count = 0
                    if f < local_best_f:
                        local_best_f = f
                else:
                    # Reject
                    sigma *= 0.9
                    no_improve_count += 1
                    
                # Stop if step size is too small
                if sigma < 1e-4:
                    break
        
        return best_x, best_f

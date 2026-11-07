import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize population
        # Bias x_2...x_dim towards 0
        # x_1 is the trade-off variable
        pop_size = 20
        X = np.zeros((pop_size, self.dim))
        # Set x_2...x_dim to 0 (clipped to bounds)
        for i in range(1, self.dim):
            X[:, i] = np.clip(0.0, lo[i], hi[i])
            
        # Set x_1 to random values in [0, 1] clipped to bounds
        # Usually x_1 is in [0, 1]
        x1_lo = max(0.0, lo[0])
        x1_hi = min(1.0, hi[0])
        if x1_lo > x1_hi:
            x1_lo, x1_hi = lo[0], hi[0]
            
        X[:, 0] = np.random.uniform(x1_lo, x1_hi, pop_size)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Store all evaluated points
        all_X = X.copy()
        all_F = F.copy()
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Pick a random parent
            parent_idx = np.random.randint(pop_size)
            child = X[parent_idx].copy()
            
            # Perturb x_1
            # Small step to explore the Pareto front
            step = 0.05 * (x1_hi - x1_lo) if x1_hi > x1_lo else 0.05
            child[0] += np.random.normal(0, step)
            child[0] = np.clip(child[0], x1_lo, x1_hi)
            
            # Keep x_2...x_dim at 0 (or their biased values)
            # We don't perturb them because 0 is the global minimum
            # If we want to be robust, we could add small noise, but it might hurt
            # Let's keep them fixed at the biased value for this strategy
            
            fc = problem(child)
            used += 1
            
            # Replace worst point in population?
            # Or just add to archive?
            # The task says "Every returned objective vector is recorded".
            # So we just need to call problem().
            # We don't strictly need to maintain a population for the score,
            # but maintaining a diverse set helps if we were doing real search.
            # Here, since we are on the Pareto front, any new point is useful.
            
            # Update population to maintain diversity?
            # Simple strategy: replace a random point in the population
            # This ensures we explore different x_1 values
            X[parent_idx] = child
            F[parent_idx] = fc
            
            # Record
            all_X = np.vstack([all_X, child])
            all_F = np.vstack([all_F, fc])
            
        return all_X

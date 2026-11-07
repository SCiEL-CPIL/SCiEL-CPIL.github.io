import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        pop_size = 20
        
        rng = np.random.default_rng(0)
        
        # Generate weight vectors on the simplex
        # Using random points ensures coverage of the Pareto front
        raw_weights = rng.random((pop_size, n_obj))
        weights = raw_weights / np.sum(raw_weights, axis=1, keepdims=True)
        
        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size
        
        # Ideal point (minimum of each objective seen so far)
        z_star = np.min(F, axis=0)
        
        # Helper: Tchebycheff scalarization
        def tchebycheff(f, w, z):
            # max_i (w_i * |f_i - z_i|)
            # Since we are minimizing and f_i >= z_i usually (z is min), 
            # but if f_i < z_i, z should have been updated. 
            # We update z_star before calculation, so f_i >= z_i is not guaranteed for the OLD point 
            # if z_star was updated by the NEW point.
            # Actually, z_star is the global min. So f_i - z_i >= 0.
            diffs = f - z
            # Ensure non-negative (safety)
            diffs = np.maximum(diffs, 0)
            return np.max(w * diffs)

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            for i in range(pop_size):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Mutation: Gaussian noise
                # Scale with span
                mutation_scale = 0.1
                noise = mutation_scale * span * rng.standard_normal(dim)
                x_new = X[i] + noise
                x_new = np.clip(x_new, lo, hi)
                
                # Evaluate
                f_new = np.array(problem(x_new), dtype=np.float64)
                used += 1
                
                # Update ideal point
                z_star = np.minimum(z_star, f_new)
                
                # Calculate Tchebycheff distances using the UPDATED z_star
                w_i = weights[i]
                
                g_new = tchebycheff(f_new, w_i, z_star)
                g_old = tchebycheff(F[i], w_i, z_star)
                
                # Replace if the new solution is better for this specific weight
                if g_new < g_old:
                    X[i] = x_new
                    F[i] = f_new

        return X

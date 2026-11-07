import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size: use 20 to allow 9 generations (20 init + 9*20 offspring = 200)
        pop_size = 20
        
        # Generate weight vectors for Tchebycheff decomposition
        # For 3 objectives, use systematic weight vectors
        n_sub = pop_size
        weights = []
        # Use a grid-like approach for 3 objectives
        # Generate points on a 2D simplex (since sum of weights = 1)
        # Use Fibonacci sphere or simple grid
        # For simplicity, use a structured set
        n_per_axis = int(np.ceil(n_sub ** 0.5))
        if n_per_axis < 2:
            n_per_axis = 2
        # Generate weight vectors on simplex
        w1_vals = np.linspace(0.05, 0.9, n_per_axis)
        w2_vals = np.linspace(0.05, 0.9, n_per_axis)
        for w1 in w1_vals:
            for w2 in w2_vals:
                w3 = 1.0 - w1 - w2
                if w3 > 0.05:
                    weights.append([w1, w2, w3])
        
        # If we don't have enough, add more
        while len(weights) < n_sub:
            # Add random weights
            w = np.random.dirichlet(np.ones(n_obj))
            weights.append(w.tolist())
        
        weights = np.array(weights[:n_sub])
        
        # Normalize weights
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Initialize population
        X = np.random.uniform(lo, hi, size=(pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Assign each individual to a weight vector (round-robin or random)
        # Each individual has a fixed weight vector
        individual_weights = weights  # one weight per individual
        
        def tchebycheff_scalarize(F_i, w, z_min, z_max):
            """Compute Tchebycheff scalarization value"""
            F_norm = (F_i - z_min) / (z_max - z_min + 1e-10)
            return np.max(w * F_norm)
        
        # Track ideal and nadir points
        z_min = np.min(F, axis=0)
        z_max = np.max(F, axis=0)
        
        # Neighborhood: each individual can mate with nearby weight vectors
        # Define neighborhood size
        neighborhood_size = 5
        
        def get_neighbors(idx):
            """Get indices of neighboring weight vectors"""
            # Sort by weight vector distance
            dists = np.linalg.norm(weights - weights[idx], axis=1)
            neighbors = np.argsort(dists)[1:1+neighborhood_size]
            return neighbors.tolist()
        
        # Precompute neighborhoods
        neighborhoods = [get_neighbors(i) for i in range(pop_size)]
        
        # Main loop
        while used < budget:
            if time.time() > deadline:
                break
            
            offspring_X = []
            offspring_F = []
            offspring_parent = []
            
            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select parent from neighborhood
                neighbors = neighborhoods[i]
                if len(neighbors) == 0:
                    parent_idx = i
                else:
                    parent_idx = np.random.choice(neighbors)
                
                # DE mutation
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
                
                F_weight = 0.5 + 0.5 * np.random.rand()
                C_rate = 0.3
                
                v = X[parent_idx] + F_weight * (X[r1] - X[r2])
                
                j_rand = np.random.randint(n_dim)
                mask = np.random.rand(n_dim) < C_rate
                mask[j_rand] = True
                
                child = np.where(mask, v, X[parent_idx])
                child = np.clip(child, lo, hi)
                
                fv = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fv)
                offspring_parent.append(i)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Update z_min and z_max
            z_min = np.minimum(z_min, np.min(offspring_F, axis=0))
            z_max = np.maximum(z_max, np.max(offspring_F, axis=0))
            
            # Selection: for each offspring, compare with its parent
            # Use Tchebycheff scalarization with the individual's weight
            for i in range(len(offspring_X)):
                parent_idx = offspring_parent[i]
                
                # Compute scalarization for child and parent
                child_val = tchebycheff_scalarize(offspring_F[i], individual_weights[parent_idx], z_min, z_max)
                parent_val = tchebycheff_scalarize(F[parent_idx], individual_weights[parent_idx], z_min, z_max)
                
                # Replace if child is better (lower scalarization value)
                if child_val < parent_val:
                    X[parent_idx] = offspring_X[i]
                    F[parent_idx] = offspring_F[i]
        
        return X

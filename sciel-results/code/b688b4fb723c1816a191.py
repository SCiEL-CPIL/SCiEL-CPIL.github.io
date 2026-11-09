import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 50 # Population size
        T = 5  # Neighborhood size
        
        # Generate weight vectors
        # For 3 objectives, generate points on the simplex
        weights = self._generate_weights(N, self.n_obj, rng)
        
        # Find neighbors
        neighbors = self._find_neighbors(weights, T)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Ideal point z*
        z_star = np.min(F, axis=0)
        
        # Calculate initial Tchebycheff values
        g = self._tchebycheff(F, z_star, weights)
        
        while used < self.budget:
            # Determine how many offspring to generate in this iteration
            # We can generate 1 offspring per individual, so N offspring per generation
            # But we must respect budget
            batch_size = min(N, self.budget - used)
            
            # Shuffle indices to process
            indices = rng.permutation(N)
            processed = 0
            
            for i in indices:
                if processed >= batch_size:
                    break
                
                # Select parents from neighborhood of i
                # Pick 2 distinct parents from neighbors[i]
                neigh = neighbors[i]
                p1_idx, p2_idx = rng.choice(neigh, size=2, replace=False)
                
                # Crossover
                if rng.random() < 0.5:
                    # Uniform crossover
                    mask = rng.random(self.dim) < 0.5
                    child = np.where(mask, X[p1_idx], X[p2_idx])
                else:
                    # Simple average
                    child = (X[p1_idx] + X[p2_idx]) / 2.0
                
                # Mutation
                # Gaussian mutation
                sigma = 0.2 * (hi - lo)
                child = child + rng.normal(0, sigma, self.dim)
                child = np.clip(child, lo, hi)
                
                # Evaluate
                f_child = problem(child)
                used += 1
                processed += 1
                
                # Update ideal point
                z_star = np.minimum(z_star, f_child)
                
                # Update individual i if better
                # Recalculate g for i with new z_star? 
                # Standard MOEA/D updates z* and then checks.
                # If we update z*, previous g values change. 
                # To keep it simple and consistent, we recompute g for the child against the current z_star.
                # And we compare it to the current g of individual i (which might be outdated if z_star changed).
                # A more robust approach: store g values and update them when z_star changes is expensive.
                # Approximation: Compare child's g against current individual's g using the NEW z_star.
                
                g_child = self._tchebycheff(np.array([f_child]), z_star, weights[i:i+1])[0]
                g_current = self._tchebycheff(np.array([F[i]]), z_star, weights[i:i+1])[0]
                
                if g_child < g_current:
                    X[i] = child
                    F[i] = f_child
                    # Note: We don't update g array directly, we recompute on the fly or just keep F and X.
                    # Since we only compare against the specific weight i, this is valid.
        
        return X

    def _generate_weights(self, N, n_obj, rng):
        # Generate N points uniformly on the (n_obj-1) simplex
        # Using Dirichlet distribution with alpha=1
        weights = rng.dirichlet(np.ones(n_obj), size=N)
        return weights

    def _find_neighbors(self, weights, T):
        N = len(weights)
        neighbors = []
        for i in range(N):
            # Distance to all other weights
            dists = np.linalg.norm(weights - weights[i], axis=1)
            # Sort indices by distance
            sorted_indices = np.argsort(dists)
            # Exclude self
            sorted_indices = sorted_indices[sorted_indices != i]
            # Take T nearest
            neigh = sorted_indices[:T].tolist()
            neighbors.append(neigh)
        return neighbors

    def _tchebycheff(self, F, z_star, weights):
        # F: (M, n_obj)
        # z_star: (n_obj,)
        # weights: (M, n_obj)
        # g(x, lambda) = max_i { lambda_i * | f_i(x) - z_i* | }
        
        # Broadcasting: F (M, 1, n_obj), weights (M, 1, n_obj), z_star (1, 1, n_obj)
        # Actually F is (M, n_obj), weights is (M, n_obj)
        # We want element-wise multiplication then max over objective axis.
        
        # f_diff = np.abs(F[:, np.newaxis, :] - z_star[np.newaxis, np.newaxis, :])
        # This is getting complicated with shapes.
        
        # Simpler:
        # F: (M, K)
        # weights: (M, K)
        # z_star: (K,)
        
        # term = weights * np.abs(F - z_star)
        # g = np.max(term, axis=1)
        
        term = weights * np.abs(F - z_star)
        g = np.max(term, axis=1)
        return g

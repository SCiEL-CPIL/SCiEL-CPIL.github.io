import numpy as np

class Optimizer:
    """MOEA/D multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # Budget 200. Init 100. 100 updates.
        N = 100
        
        # Generate weight vectors for MOEA/D
        # For 3 objectives, we can generate points on the simplex
        weights = self._generate_weights(N, self.n_obj, rng)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        
        # Precompute neighborhood for each weight vector
        # Distance between weight vectors
        # We use Euclidean distance in weight space
        # To save time, we can just use random neighbors or a fixed grid structure
        # For simplicity and speed, let's define neighbors as the K closest weight vectors
        K = 10
        neighbors = []
        for i in range(N):
            dists = np.sum((weights - weights[i])**2, axis=1)
            # Get K closest, excluding self
            idxs = np.argsort(dists)
            neighbors.append(idxs[1:K+1])
        
        # MOEA/D Loop
        # We have 100 evaluations left.
        # We will perform 100 updates.
        # Each update corresponds to one subproblem.
        
        # To ensure diversity and coverage, we can shuffle the order of subproblems
        # or just iterate. Iterating 0..N-1 is standard.
        
        for i in range(N):
            # Select best parent from neighborhood of i
            # Fitness is Tchebycheff scalarization
            w = weights[i]
            
            # Calculate fitness for neighbors
            # g = max_j { w_j * |f_j(x) - z_j^*| }
            # We need ideal point z*. We can use the current best per objective.
            z_star = np.min(F, axis=0)
            
            # Get fitness of neighbors
            neigh_idx = neighbors[i]
            neigh_F = F[neigh_idx]
            # Calculate Tchebycheff fitness for neighbors
            # f_j(x) - z_j*
            diff = np.abs(neigh_F - z_star)
            # Weighted
            weighted_diff = diff * w
            # Max
            fitness_vals = np.max(weighted_diff, axis=1)
            
            # Select best parent (lowest fitness)
            best_parent_local_idx = np.argmin(fitness_vals)
            parent_idx = neigh_idx[best_parent_local_idx]
            
            # Crossover
            # Select a second parent from neighborhood
            # Randomly pick another neighbor
            if len(neigh_idx) > 1:
                other_local_idx = rng.integers(0, len(neigh_idx))
                if other_local_idx == best_parent_local_idx:
                    other_local_idx = (other_local_idx + 1) % len(neigh_idx)
                other_idx = neigh_idx[other_local_idx]
            else:
                other_idx = parent_idx
                
            p1 = X[parent_idx]
            p2 = X[other_idx]
            
            # SBX Crossover
            child = self._sbx_crossover(p1, p2, rng, lo, hi, eta_c=20.0)
            
            # Polynomial Mutation
            child = self._poly_mutation(child, rng, lo, hi, eta_m=20.0)
            
            # Evaluate
            F_child = problem(child)
            
            # Update if better for subproblem i
            # Calculate fitness of child for subproblem i
            diff_child = np.abs(F_child - z_star)
            fitness_child = np.max(diff_child * w)
            
            # Current fitness of X[i]
            diff_cur = np.abs(F[i] - z_star)
            fitness_cur = np.max(diff_cur * w)
            
            if fitness_child < fitness_cur:
                X[i] = child
                F[i] = F_child
                # Update z_star if improved
                z_star = np.minimum(z_star, F_child)
            
        return X

    def _generate_weights(self, N, n_obj, rng):
        """Generate N weight vectors on the simplex."""
        weights = np.zeros((N, n_obj))
        # Use Dirichlet distribution for random weights
        # Alpha = 1 for uniform distribution on simplex
        alphas = np.ones(n_obj)
        weights = rng.dirichlet(alphas, size=N)
        return weights

    def _sbx_crossover(self, p1, p2, rng, lo, hi, eta_c=20.0):
        child = np.empty_like(p1)
        for i in range(len(p1)):
            if rng.random() < 0.5:
                child[i] = p1[i]
                if abs(p2[i] - p1[i]) > 1e-14:
                    r = rng.random()
                    if r <= 0.5:
                        beta = (2 * r) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - r))) ** (1.0 / (eta_c + 1.0))
                    c1 = 0.5 * (1 + beta) * p1[i] + 0.5 * (1 - beta) * p2[i]
                    child[i] = np.clip(c1, lo[i], hi[i])
            else:
                child[i] = p2[i]
        return child

    def _poly_mutation(self, x, rng, lo, hi, eta_m=20.0):
        for i in range(len(x)):
            if rng.random() < 1.0 / len(x):
                r = rng.random()
                if r <= 0.5:
                    delta = (2 * r) ** (1.0 / (eta_m + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2 * (1 - r)) ** (1.0 / (eta_m + 1.0))
                
                x[i] = x[i] + delta * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x

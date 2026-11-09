"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D style multi-objective decomposition optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Population size: Use a significant portion of the budget
        # For 200 budget, 50 pop size allows 4 generations
        pop_size = min(50, self.budget // 4)
        if pop_size < 5:
            pop_size = 5
            
        # Adjust pop_size to fit budget exactly if possible, or just run until budget exhausted
        # Let's aim for pop_size such that pop_size * generations <= budget
        # We will run in a loop until budget is exhausted.
        
        # Initialize weight vectors for decomposition
        # For 2 objectives, weights are on the line segment
        # lambda_i = [i/(N-1), 1 - i/(N-1)]
        if m == 2:
            weights = np.array([[i / (pop_size - 1), 1 - i / (pop_size - 1)] for i in range(pop_size)])
        else:
            # General simplex for more objectives (not needed here but good for robustness)
            weights = rng.dirichlet(np.ones(m), size=pop_size)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Ideal point (best value for each objective)
        # Initialize with infinity for minimization
        z_ideal = np.full(m, np.inf)
        for i in range(pop_size):
            for j in range(m):
                if F[i][j] < z_ideal[j]:
                    z_ideal[j] = F[i][j]
        
        # If any ideal point is inf (shouldn't happen with random init), set to max F
        for j in range(m):
            if np.isinf(z_ideal[j]):
                z_ideal[j] = np.max(F[:, j])

        # Neighborhood size
        T = min(10, pop_size - 1)
        
        # Precompute neighbors
        # For linear weights, neighbors are close in index
        neighbors = []
        for i in range(pop_size):
            # Indices closest to i
            dists = [(abs(i - j), j) for j in range(pop_size)]
            dists.sort()
            neighbors.append([j for _, j in dists[:T]])

        def tchebycheff_fitness(x, f, weight, z):
            """Compute Tchebycheff scalarization value."""
            # g(x, lambda) = max_j { lambda_j * |f_j(x) - z_j| }
            diff = np.abs(f - z)
            weighted_diff = weight * diff
            return np.max(weighted_diff)

        # Compute initial scalar fitnesses
        scalar_F = np.zeros(pop_size)
        for i in range(pop_size):
            scalar_F[i] = tchebycheff_fitness(X[i], F[i], weights[i], z_ideal)

        # SBX Crossover
        def sbx_crossover(x1, x2, bounds, eta=15.0):
            if rng.random() < 0.9: # Crossover probability
                child1 = x1.copy()
                child2 = x2.copy()
                for k in range(n):
                    if rng.random() < 0.5:
                        if abs(x1[k] - x2[k]) > 1e-14:
                            if 2 * rng.random() <= (x2[k] - x1[k]) / (hi[k] - lo[k]) + 1:
                                beta = (2 * rng.random() * (x1[k] - lo[k]) / (hi[k] - lo[k]) + 1) ** (1 / (eta + 1))
                            else:
                                beta = (2 * (1 - rng.random()) * (hi[k] - x1[k]) / (hi[k] - lo[k]) + 1) ** (1 / (eta + 1))
                            child1[k] = 0.5 * ((1 + beta) * x1[k] + (1 - beta) * x2[k])
                            child2[k] = 0.5 * ((1 + 1/beta) * x1[k] + (1 - 1/beta) * x2[k])
            return child1, child2

        # Polynomial Mutation
        def polynomial_mutation(x, bounds, eta=20.0):
            for k in range(n):
                if rng.random() < 1.0 / n:
                    delta = 2 * rng.random() - 1
                    if delta < 0:
                        xk = (x[k] - lo[k]) / (hi[k] - lo[k])
                        xl = (1 + 2 * (x[k] - lo[k]) / (hi[k] - lo[k])) ** (eta + 1)
                        delta1 = (1 + xl) ** (-1 / (eta + 1))
                    else:
                        xk = (hi[k] - x[k]) / (hi[k] - lo[k])
                        xr = (1 + 2 * (hi[k] - x[k]) / (hi[k] - lo[k])) ** (eta + 1)
                        delta1 = 1 - (1 + xr) ** (-1 / (eta + 1))
                    
                    delta1 = 2 * delta1
                    x[k] = x[k] + delta1 * (hi[k] - lo[k])
                    x[k] = np.clip(x[k], lo[k], hi[k])
            return x

        while used < self.budget:
            # Determine batch size for this generation
            # We can update one subproblem at a time or a batch
            # Standard MOEA/D updates one subproblem per iteration.
            # To use the budget efficiently, we'll update multiple subproblems.
            # Let's update min(pop_size, budget - used) subproblems in this "generation"
            batch_size = min(pop_size, self.budget - used)
            
            # Shuffle indices to update
            indices = rng.permutation(pop_size)
            
            for b in range(batch_size):
                i = indices[b]
                
                # Select parents from neighbors of i
                # Pick 2 distinct neighbors
                neigh = neighbors[i]
                p1_idx, p2_idx = rng.choice(neigh, size=2, replace=False)
                
                # Crossover
                child1, child2 = sbx_crossover(X[p1_idx], X[p2_idx], bounds)
                
                # Mutation
                child1 = polynomial_mutation(child1, bounds)
                
                # Evaluate
                f_child = problem(child1)
                used += 1
                
                # Update ideal point
                for j in range(m):
                    if f_child[j] < z_ideal[j]:
                        z_ideal[j] = f_child[j]
                    
                # Calculate scalar fitness for child using weight of subproblem i
                scalar_child = tchebycheff_fitness(child1, f_child, weights[i], z_ideal)
                
                # Also recalculate scalar fitness for current parent i (since z_ideal might have changed)
                # Note: z_ideal only decreases, so previous scalar values might be stale.
                # For strict MOEA/D, we usually update all affected or just compare locally.
                # Let's recalculate scalar_F[i] to be safe, or just compare if child is better.
                # If child is better, we replace.
                
                # Recalculate scalar fitness for i with updated z_ideal
                scalar_parent_i = tchebycheff_fitness(X[i], F[i], weights[i], z_ideal)
                
                # Replace if child is better
                if scalar_child < scalar_parent_i:
                    X[i] = child1
                    F[i] = f_child
                    scalar_F[i] = scalar_child
                    
                    # Update neighbors' scalar fitnesses? 
                    # In standard MOEA/D, we might update the neighborhood.
                    # For simplicity and speed, we only update the specific subproblem's stored value.
                    # The scalar_F array is used for potential future selections if we did tournament, 
                    # but here we select randomly from neighbors. 
                    # However, the "better" check uses the current z_ideal.
                    
            # End of batch

        return X
# EVOLVE-BLOCK-END

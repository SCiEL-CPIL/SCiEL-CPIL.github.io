import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # MOEA/D parameters
        N = 20  # Number of subproblems (population size)
        # Create weight vectors using a simple grid approach for 3 objectives
        # We need N points on the simplex. 
        # For n_obj=3, we can use a 2D grid.
        # Let's generate weights such that they sum to 1.
        # A simple way: use random points on the simplex and project, or use a grid.
        # Given small N, let's use a specific grid or random.
        # Random points on simplex:
        weights = np.zeros((N, self.n_obj))
        for i in range(N):
            # Generate random numbers and normalize
            r = np.random.rand(self.n_obj)
            weights[i] = r / np.sum(r)
        
        # Initialize population
        # Stratified initialization for better coverage
        X = np.random.uniform(lo, hi, (N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Archive for global best solutions
        archive_X = [X[0]]
        archive_F = [F[0]]
        
        # Helper: Check dominance
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        # Helper: Update archive
        def update_archive(x_new, f_new):
            nonlocal archive_X, archive_F
            # Check if f_new is dominated by any in archive
            for i in range(len(archive_F)):
                if dominates(archive_F[i], f_new):
                    return
            # Remove any solutions in archive dominated by f_new
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not dominates(f_new, archive_F[i]):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            # Add new solution
            new_archive_X.append(x_new)
            new_archive_F.append(f_new)
            # Limit archive size to avoid memory issues, keep non-dominated
            # For simplicity, if it grows too large, we can prune, but N is small
            # and 200 evals is small.
            archive_X = new_archive_X
            archive_F = new_archive_F
            
        # Initial archive update
        for i in range(N):
            update_archive(X[i], F[i])
            
        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring for each subproblem
            offspring_X = []
            offspring_F = []
            
            for i in range(N):
                # Select neighbors
                # Calculate distance to other weights
                dists = np.linalg.norm(weights - weights[i], axis=1)
                # Select 5 nearest neighbors including self? 
                # Standard MOEA/D uses a neighborhood size T. Let T=5.
                T = min(5, N)
                neighbor_indices = np.argsort(dists)[:T]
                
                # Differential Evolution
                # Select 3 distinct parents from neighbors
                # If T < 3, use random from population
                if T >= 3:
                    parents = np.random.choice(neighbor_indices, 3, replace=False)
                else:
                    parents = np.random.choice(N, 3, replace=False)
                
                p1, p2, p3 = parents[0], parents[1], parents[2]
                
                # DE formula: v = x_p1 + F * (x_p2 - x_p3)
                F_factor = 0.5
                v = X[p1] + F_factor * (X[p2] - X[p3])
                
                # Crossover
                # Binomial crossover
                CR = 0.5
                u = np.random.rand(self.dim)
                j_rand = np.random.randint(self.dim)
                
                child = X[i].copy()
                for j in range(self.dim):
                    if u[j] <= CR or j == j_rand:
                        child[j] = v[j]
                    else:
                        child[j] = X[i][j]
                
                # Mutation (Gaussian)
                # Small perturbation
                sigma = 0.1 * (hi - lo)
                child += np.random.normal(0, sigma, self.dim)
                child = np.clip(child, lo, hi)
                
                offspring_X.append(child)
                
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_F)
            
            # Selection
            # For each subproblem i, compare parent X[i], F[i] with offspring X_off[i], F_off[i]
            # using the Tchebycheff scalarization function
            new_X = X.copy()
            new_F = F.copy()
            
            for i in range(N):
                # Tchebycheff function: max_j (w_ij * |f_j - z_j^min|)
                # We need an ideal point z^min
                # Approximate z^min as the current min of the population
                z_min = np.min(F, axis=0)
                
                # Scalarize parent
                f_parent = F[i]
                # Avoid division by zero if weight is 0, but weights are normalized
                # Tchebycheff: max(w * |f - z|)
                # If w is 0, term is 0.
                # Weights sum to 1, so some might be small.
                # Standard formula: g(x, w) = max_{1<=j<=m} { w_j * |f_j(x) - z_j^min| }
                
                # To handle z_min updates robustly, we can use a fixed reference or dynamic.
                # Dynamic z_min is common.
                
                g_parent = np.max(weights[i] * np.abs(f_parent - z_min))
                
                # Scalarize offspring
                f_off = offspring_F[i]
                g_off = np.max(weights[i] * np.abs(f_off - z_min))
                
                # Accept offspring if it is better (lower) or equal
                if g_off <= g_parent:
                    new_X[i] = offspring_X[i]
                    new_F[i] = offspring_F[i]
                    update_archive(offspring_X[i], offspring_F[i])
            
            X = new_X
            F = new_F
            
            # Update z_min
            # We can update z_min to the min of the new population
            # This helps the scalarization adapt
            # Note: z_min is recalculated in the next iteration's scalarization step implicitly 
            # if we use the current F. But inside the loop we used the old z_min.
            # Let's update z_min for the next iteration.
            # Actually, the code above calculates z_min inside the loop for each i, 
            # which is inefficient but correct for the "current" state before update.
            # It's better to calculate z_min once per generation.
            
            # Optimization: Move z_min calculation outside the inner loop
            # Let's refactor slightly for efficiency if needed, but N=20 is small.
            
        return X

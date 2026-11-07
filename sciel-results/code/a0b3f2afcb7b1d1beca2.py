"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """MOEA/D style multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, leave 20s buffer
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Generate weight vectors for decomposition
        # We use a grid of weight vectors to cover the objective space
        # For 3 objectives, we can use a simplex-lattice approach
        n_obj = self.n_obj
        dim = self.dim
        
        # Generate weight vectors on a simplex
        # Number of weight vectors determines the population size
        # With budget 200, we can afford a population of ~100-150
        # Let's aim for a population size that allows for a few generations
        # Pop size 100 seems reasonable.
        
        # Generate weight vectors
        # For 3D, we can generate points on the surface of a simplex
        # Using a grid resolution
        resolution = 9 # 9x9 grid on the 2D simplex for 3 objectives
        weights = []
        for i in range(resolution):
            for j in range(resolution - i):
                k = resolution - i - j
                w = np.array([i, j, k], dtype=float)
                w = w / w.sum()
                weights.append(w)
        
        # If we have too many or too few, adjust
        # We want a population size that is manageable
        # Let's target a population size of 100
        target_pop_size = 100
        
        # If weights are too many, subsample
        if len(weights) > target_pop_size:
            indices = np.linspace(0, len(weights) - 1, target_pop_size).astype(int)
            weights = [weights[i] for i in indices]
        # If too few, duplicate or just use what we have
        # Actually, for 3 objectives, resolution 9 gives (9*10)/2 = 45 points.
        # That's a bit small for 200 budget. Let's increase resolution.
        
        # Let's just generate a fixed set of weight vectors that sum to 1
        # and are well-distributed.
        # A simple way: use a grid on the simplex
        # For 3 objectives:
        # w1 + w2 + w3 = 1, wi >= 0
        # We can iterate w1 from 0 to 1, w2 from 0 to 1-w1
        # Let's use a finer grid
        n_grid = 15
        weights = []
        for i in range(n_grid + 1):
            for j in range(n_grid - i + 1):
                k = n_grid - i - j
                w = np.array([i, j, k], dtype=float)
                w = w / w.sum()
                weights.append(w)
        
        # Subsample to target_pop_size
        if len(weights) > target_pop_size:
            indices = np.linspace(0, len(weights) - 1, target_pop_size).astype(int)
            weights = [weights[i] for i in indices]
        
        # Ensure we have exactly target_pop_size
        while len(weights) < target_pop_size:
            # Duplicate a random weight
            idx = np.random.randint(len(weights))
            weights.append(weights[idx])
            
        weights = np.array(weights[:target_pop_size])
        
        # 2. Initialize population
        # Use random initialization
        X = lo + (hi - lo) * np.random.random((target_pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = target_pop_size
        
        # 3. Determine neighborhood
        # Each weight vector has a neighborhood of T other weight vectors
        T = 20
        # Calculate distances between weight vectors to find neighbors
        # This is O(N^2), which is fine for N=100
        dists = np.zeros((target_pop_size, target_pop_size))
        for i in range(target_pop_size):
            for j in range(target_pop_size):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        neighbors = []
        for i in range(target_pop_size):
            # Sort by distance, take T closest (excluding self)
            sorted_indices = np.argsort(dists[i])
            # Exclude self (index 0 in sorted list usually, but check)
            neigh = [j for j in sorted_indices if j != i][:T]
            neighbors.append(neigh)
        
        # 4. Main Loop
        # We have budget - target_pop_size evaluations left
        # We will perform local searches
        # For each subproblem, we generate offspring using its neighbor solutions
        
        # We need to track which solutions are "best" for each subproblem
        # In MOEA/D, we typically keep the best solution for each weight vector
        # based on a scalarizing function (e.g., Tchebycheff)
        
        # Calculate ideal point (min of each objective) for normalization
        # We'll update this as we go, or just use the current min
        # For Tchebycheff: g(x, w) = max_i { w_i * |f_i(x) - z_i^*| }
        # z_i^* is the ideal point (min f_i)
        
        # Let's use the Tchebycheff approach
        # We need to estimate the ideal point. We can use the min of the current population.
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Update ideal point
            z_star = np.min(F, axis=0)
            
            # For each subproblem i, we try to improve its solution
            # We pick a random neighbor j for each i, or just iterate through all?
            # Standard MOEA/D: For each i, pick a random neighbor j,
            # create offspring from x_j (or x_i and x_j), and if it's better
            # for subproblem i, replace x_i.
            
            # To be efficient, we can iterate through all subproblems in random order
            # and for each, generate 1-2 offspring.
            
            # Let's do a pass: for each i, generate one offspring
            # This uses target_pop_size evaluations per generation.
            # We have ~100 evals left, so we can do ~1 generation.
            # To get more generations, we can generate fewer offspring per pass
            # or accept that we might only do 1-2 passes.
            
            # Let's try to do as many "local searches" as possible.
            # Each local search uses 1 evaluation (one offspring).
            # We will loop through subproblems and for each, generate an offspring
            # from a random neighbor.
            
            # To avoid bias, shuffle the order of subproblems
            order = np.random.permutation(target_pop_size)
            
            for i in order:
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Pick a random neighbor j
                if neighbors[i]:
                    j = neighbors[i][np.random.randint(len(neighbors[i]))]
                else:
                    j = np.random.randint(target_pop_size)
                
                # Create offspring from x_j (and maybe x_i)
                # Let's use SBX crossover between x_i and x_j, then mutation
                x1 = X[i]
                x2 = X[j]
                
                # SBX
                child = x1.copy()
                for d in range(dim):
                    if np.random.random() < 0.5:
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 20)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 20)
                        child[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                
                # Mutation
                # Adaptive sigma: start large, end small
                # Progress = used / budget
                progress = used / self.budget
                sigma = 0.3 * (1.0 - progress) + 0.05
                child = child + sigma * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                # Evaluate
                f_child = problem(child)
                used += 1
                
                # Check if child is better for subproblem i
                # Tchebycheff value
                def tchebycheff(f, w, z_star):
                    # Handle zero weights or zero range
                    # If w_i is 0, the term is 0? No, usually we normalize.
                    # Simple Tchebycheff: max_i w_i * |f_i - z_i|
                    # But if w_i is small, it ignores that objective.
                    # This is the intended behavior for decomposition.
                    vals = []
                    for k in range(self.n_obj):
                        # If w[k] is very small, we might want to avoid division by zero
                        # in some variants, but here we just multiply.
                        # However, if w[k] is 0, the term is 0, which means
                        # that objective is ignored for this subproblem.
                        # This is correct for MOEA/D.
                        vals.append(w[k] * abs(f[k] - z_star[k]))
                    return max(vals)
                
                w_i = weights[i]
                g_child = tchebycheff(f_child, w_i, z_star)
                g_parent = tchebycheff(F[i], w_i, z_star)
                
                # Replace if better
                if g_child < g_parent:
                    X[i] = child
                    F[i] = f_child
        
        return X
# EVOLVE-BLOCK-END

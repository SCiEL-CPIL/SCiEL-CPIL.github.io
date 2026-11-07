"""Multi-objective optimizer on DTLZ7 using SPEA2."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using Strength Pareto Evolutionary Algorithm 2 (SPEA2).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Track evaluations
        eval_count = 0
        all_points = []
        all_objectives = []
        
        def eval_point(x):
            nonlocal eval_count
            eval_count += 1
            f = np.array(problem(x))
            all_points.append(x.copy())
            all_objectives.append(f)
            return f

        def dominates(f1, f2):
            """Check if f1 dominates f2 (minimization)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # SPEA2 Parameters
        pop_size = 20
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))
        for i in range(pop_size):
            F[i] = eval_point(X[i])
            
        # SPEA2 Loop
        while eval_count < self.budget:
            # Generate offspring
            Q_X = np.zeros_like(X)
            Q_F = np.zeros_like(F)
            
            # We need to generate offspring until we have enough evaluations or pop_size offspring
            offspring_generated = 0
            while offspring_generated < pop_size and eval_count < self.budget:
                # Tournament Selection (size 2) based on SPEA2 fitness
                # First, we need to calculate fitness for the current population X
                # Standard SPEA2 combines P and Q, but for generation, we select parents from P.
                # We calculate a "strength" for each individual in P based on dominance within P.
                
                # Calculate dominance counts within P
                R = np.zeros(pop_size, dtype=int)
                for i in range(pop_size):
                    for j in range(pop_size):
                        if i != j and dominates(F[i], F[j]):
                            R[i] += 1
                
                # Fitness for selection:
                # If R[i] > 0, it is dominated, so it's bad.
                # If R[i] == 0, it is non-dominated.
                # We use a simple tournament on non-dominated points if available, 
                # otherwise we pick random or use a penalty.
                # To be robust, we assign a score:
                # score = -R[i] if R[i] == 0 else -pop_size - 1
                # Actually, let's use the standard SPEA2 fitness calculation for the combined set later.
                # For parent selection, we can just use a tournament on the "rank" or "strength".
                # Let's use a simple heuristic: prefer non-dominated.
                
                # Assign fitness for selection
                fitness = np.zeros(pop_size)
                for i in range(pop_size):
                    if R[i] == 0:
                        fitness[i] = 1.0 + R[i] # R[i] is 0, so 1.0
                    else:
                        fitness[i] = 1.0 + pop_size + 1 # Very bad score (higher is worse for minimization of score)
                
                # Invert for tournament (we want to select low fitness)
                # Tournament size 2
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                if fitness[i1] < fitness[i2]:
                    p1_idx = i1
                else:
                    p1_idx = i2
                    
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                if fitness[i1] < fitness[i2]:
                    p2_idx = i1
                else:
                    p2_idx = i2
                    
                # Crossover: Simulated Binary Crossover (SBX)
                # Or simple Blend Crossover (BLX-alpha)
                # Let's use BLX-alpha with alpha = 0.5
                alpha = 0.5
                c1 = X[p1_idx].copy()
                c2 = X[p2_idx].copy()
                
                # For each dimension
                for j in range(self.dim):
                    lo_c = min(c1[j], c2[j])
                    hi_c = max(c1[j], c2[j])
                    delta = hi_c - lo_c
                    u1 = rng.random()
                    u2 = rng.random()
                    
                    # BLX-alpha generates two children
                    # child1
                    if u1 < 0.5:
                        child1_j = lo_c + u1 * (1 + alpha) * delta
                    else:
                        child1_j = hi_c - (1 - u1) * (1 + alpha) * delta
                        
                    # child2
                    if u2 < 0.5:
                        child2_j = lo_c + u2 * (1 + alpha) * delta
                    else:
                        child2_j = hi_c - (1 - u2) * (1 + alpha) * delta
                        
                    c1[j] = np.clip(child1_j, lo[j], hi[j])
                    c2[j] = np.clip(child2_j, lo[j], hi[j])
                
                # Mutation: Polynomial Mutation
                # Or simple Gaussian mutation
                # Let's use Gaussian mutation with sigma = 0.1 * range
                sigma = 0.1 * (hi - lo)
                c1 += rng.normal(0, sigma)
                c2 += rng.normal(0, sigma)
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                
                # Evaluate offspring
                if offspring_generated < pop_size:
                    f1 = eval_point(c1)
                    Q_X[offspring_generated] = c1
                    Q_F[offspring_generated] = f1
                    offspring_generated += 1
                
                if eval_count < self.budget and offspring_generated < pop_size:
                    f2 = eval_point(c2)
                    Q_X[offspring_generated] = c2
                    Q_F[offspring_generated] = f2
                    offspring_generated += 1
            
            if offspring_generated == 0:
                break
                
            # Combine P and Q
            # P is X, F. Q is Q_X[:offspring_generated], Q_F[:offspring_generated]
            # Total size N = pop_size + offspring_generated
            N = pop_size + offspring_generated
            R_X = np.vstack([X, Q_X[:offspring_generated]])
            R_F = np.vstack([F, Q_F[:offspring_generated]])
            
            # Calculate SPEA2 fitness for R
            # 1. Dominance count R_i
            dom_count = np.zeros(N, dtype=int)
            for i in range(N):
                for j in range(N):
                    if i != j and dominates(R_F[i], R_F[j]):
                        dom_count[i] += 1
            
            # 2. Fitness
            # If dom_count[i] > 0, fitness is "bad" (infinite)
            # If dom_count[i] == 0, fitness is 1 + sum of dom_count of solutions dominated by i?
            # No, standard SPEA2:
            # F_i = -inf if R_i > 0
            # F_i = 1 + |{j in R | f_i dominates f_j}| if R_i == 0
            
            fitness = np.full(N, np.inf)
            for i in range(N):
                if dom_count[i] == 0:
                    # Count how many solutions i dominates
                    count = 0
                    for j in range(N):
                        if i != j and dominates(R_F[i], R_F[j]):
                            count += 1
                    fitness[i] = 1.0 + count
            
            # 3. Truncation
            # If number of non-dominated solutions (fitness < inf) is <= pop_size, select them all.
            # Otherwise, use density estimation to select the most diverse ones.
            
            non_dom_indices = np.where(fitness < np.inf)[0]
            
            if len(non_dom_indices) <= pop_size:
                # Select all non-dominated, and fill the rest with best dominated?
                # SPEA2 usually keeps only non-dominated if they fit.
                # If they don't fit, we truncate.
                # If they fit, we keep them. What about the rest of the population?
                # The new population P_{t+1} consists of the selected individuals.
                # So we just take the non-dominated ones.
                # If len < pop_size, we might need to add some dominated ones?
                # Standard SPEA2: The population size is fixed.
                # If non-dominated count < pop_size, we add the best dominated ones (lowest fitness among dominated).
                # But in SPEA2, dominated ones have fitness -inf (or very high).
                # Let's stick to: Select pop_size individuals from R based on fitness, using density for ties if necessary.
                
                # Actually, if len(non_dom_indices) < pop_size, we must select some dominated points.
                # We sort all indices by fitness.
                # Fitness: non-dom are 1+count (small), dom are inf (large).
                # So non-dom come first.
                
                # If there are more than pop_size non-dominated, we need to prune using density.
                if len(non_dom_indices) > pop_size:
                    # Density estimation
                    # For each non-dominated solution, find distance to k-th nearest neighbor in the non-dominated set.
                    # k = pop_size - len(non_dom_indices) + 1 ? No, usually k is chosen such that we can prune.
                    # Let's use k = 1 (nearest neighbor) or a fixed small k.
                    # Better: Calculate distance to the nearest neighbor for each point in the non-dominated set.
                    # Keep the ones with larger distance (more diverse).
                    
                    non_dom_F = R_F[non_dom_indices]
                    n_nd = len(non_dom_indices)
                    # Calculate pairwise distances
                    # This is O(n_nd^2 * n_obj)
                    dists = np.zeros((n_nd, n_nd))
                    for i in range(n_nd):
                        for j in range(n_nd):
                            if i != j:
                                dists[i, j] = np.linalg.norm(non_dom_F[i] - non_dom_F[j])
                    
                    # Find nearest neighbor distance for each point
                    # Set diagonal to inf
                    np.fill_diagonal(dists, np.inf)
                    nn_dists = np.min(dists, axis=1)
                    
                    # We want to keep points with LARGER nn_dists (more isolated/diverse)
                    # We need to remove (n_nd - pop_size) points.
                    # Sort nn_dists in ascending order. The smallest ones are the most crowded.
                    # Remove the ones with smallest nn_dists.
                    
                    # Indices in the non_dom set
                    sorted_indices = np.argsort(nn_dists)
                    # The first (n_nd - pop_size) indices in sorted_indices are the most crowded.
                    # We remove them.
                    to_remove = sorted_indices[:n_nd - pop_size]
                    
                    # Create a mask for non_dom_indices
                    keep_mask = np.ones(n_nd, dtype=bool)
                    keep_mask[to_remove] = False
                    
                    selected_indices = non_dom_indices[keep_mask]
                    
                else:
                    # len(non_dom_indices) <= pop_size
                    # Select all non-dominated
                    selected_indices = non_dom_indices.copy()
                    
                    # If we have space left, add best dominated points
                    if len(selected_indices) < pop_size:
                        dominated_indices = np.where(fitness == np.inf)[0]
                        # Sort dominated by... well, they all have inf fitness.
                        # We can just pick random or based on how close they are to non-dominated?
                        # For simplicity, pick random from dominated.
                        # Or better, pick the ones that are "closest" to being non-dominated?
                        # Let's just pick random to maintain diversity.
                        if len(dominated_indices) > 0:
                            n_needed = pop_size - len(selected_indices)
                            # Sample n_needed from dominated_indices
                            chosen_dominated = rng.choice(dominated_indices, size=n_needed, replace=False)
                            selected_indices = np.concatenate([selected_indices, chosen_dominated])
            
            # Update X and F
            X = R_X[selected_indices]
            F = R_F[selected_indices]
            
        return all_points
# EVOLVE-BLOCK-END

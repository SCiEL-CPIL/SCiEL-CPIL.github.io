"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-based multi-objective optimizer using weighted-sum guidance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Define islands based on weight vectors
        # For 3 objectives, we use a simplex grid of weights
        # To cover the front well, we pick specific weight vectors.
        # A simple strategy: use a set of weight vectors that sum to 1.
        # For 3D, we can use a grid on the simplex.
        # Let's use 5 islands with distinct weights to ensure diversity.
        # Weights: (1,0,0), (0,1,0), (0,0,1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
        # Actually, corners are often hard. Let's use a mix.
        # Standard WFG4 front is concave. 
        # Let's define 6 islands with weights:
        weights = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.3, 0.4],
            [0.2, 0.4, 0.4],
            [0.4, 0.2, 0.4],
            [0.4, 0.4, 0.2],
            [0.33, 0.33, 0.34],
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5]
        ])
        
        n_islands = len(weights)
        pop_per_island = max(5, self.budget // (n_islands * 10)) # Rough estimate
        # Total budget is 200. 16 islands * 5 = 80 initial. 120 left for evolution.
        # Let's just distribute budget evenly.
        
        # Initialize islands
        # Each island has a population of size P
        # We will evaluate P * n_islands points initially.
        # Then we have remaining budget for local search / DE within islands.
        
        P = 10 # Population size per island
        total_init = n_islands * P
        if total_init > self.budget:
             P = max(1, self.budget // n_islands)
             total_init = n_islands * P
             
        islands = []
        all_points = []
        
        for i in range(n_islands):
            # Initialize population for island i
            # Random initialization within bounds
            X = lo + span * rng.random((P, self.dim))
            F = np.array([problem(x) for x in X])
            all_points.extend(F)
            islands.append({'X': X, 'F': F, 'weight': weights[i]})
            
        used = total_init
        remaining = self.budget - used
        
        # Evolutionary loop
        # We perform DE-like updates within each island, guided by its weight vector.
        # Selection: pick parents based on weighted sum of objectives (approximate hypervolume contribution)
        # Mutation: standard DE
        # Crossover: binomial
        # Replacement: keep best based on weighted sum or non-dominated sorting within island?
        # To ensure coverage, we should keep diverse points. 
        # However, the "weighted sum" approach tends to cluster.
        # Let's use a hybrid: within each island, we maintain a small non-dominated set 
        # but guide mutation towards the weight direction.
        
        # Actually, a simpler and robust strategy for WFG4 (which is separable and has a known structure)
        # is to use the weights to bias the mutation.
        
        # Let's refine: 
        # For each island:
        # 1. Compute weighted scores: score = F @ weight
        # 2. Sort by score.
        # 3. Select top 3 as parents for DE.
        # 4. Generate offspring.
        # 5. Evaluate.
        # 6. Replace worst individual in the island based on weighted score.
        # This ensures each island converges to a specific point on the front.
        
        # We need to be careful: weighted sum only finds supported points. 
        # WFG4's Pareto front is convex? 
        # WFG4: f1 = x1, f2 = (1+0.1f1)(1 - sqrt(f1/f2)), f3 = ...
        # Actually WFG4 is not convex in all regions. Weighted sum might miss unsupported points.
        # However, for 3 objectives, the unsupported region is smaller.
        # To be safe, we should mix: some islands use weighted sum, some use pure random/DE with non-dominated sorting.
        
        # Revised Strategy:
        # Half the islands use Weighted Sum guidance (for convergence to specific areas).
        # Half the islands use standard NSGA-II style (non-dominated sorting + crowding) for exploration.
        
        n_ws_islands = n_islands // 2
        n_ns_islands = n_islands - n_ws_islands
        
        # Re-initialize or re-assign? 
        # Let's just run the loop.
        
        while used < self.budget:
            if time.time() - start_time > deadline:
                break
                
            # Process each island
            for i in range(n_islands):
                if used >= self.budget:
                    break
                    
                island = islands[i]
                X = island['X']
                F = island['F']
                
                if i < n_ws_islands:
                    # Weighted Sum Strategy
                    w = island['weight']
                    scores = F @ w
                    # Sort by score (lower is better)
                    sorted_idx = np.argsort(scores)
                    # Pick top 3 parents
                    parents_idx = sorted_idx[:3]
                    if len(parents_idx) < 3:
                        parents_idx = np.arange(len(parents_idx))
                        
                    # Generate 1 offspring
                    p1, p2, p3 = parents_idx[0], parents_idx[1], parents_idx[2]
                    F_factor = 0.5 + 0.5 * rng.random()
                    x_trial = X[p1] + F_factor * (X[p2] - X[p3])
                    x_trial = np.clip(x_trial, lo, hi)
                    
                    f_trial = problem(x_trial)
                    used += 1
                    all_points.append(f_trial)
                    
                    # Replace worst
                    worst_idx = sorted_idx[-1]
                    X[worst_idx] = x_trial
                    F[worst_idx] = f_trial
                else:
                    # Non-Dominated Sorting Strategy (simplified)
                    # Find non-dominated set within this island
                    n = len(F)
                    dominated = np.zeros(n, dtype=bool)
                    for j in range(n):
                        if dominated[j]:
                            continue
                        for k in range(n):
                            if j == k or dominated[k]:
                                continue
                            if np.all(F[j] <= F[k]) and np.any(F[j] < F[k]):
                                dominated[k] = True
                                
                    nd_indices = np.where(~dominated)[0]
                    if len(nd_indices) == 0:
                        nd_indices = [0]
                        
                    # Pick parent from non-dominated set
                    parent_idx = nd_indices[rng.integers(len(nd_indices))]
                    
                    # Pick 2 other random parents from island
                    others = [idx for idx in range(n) if idx != parent_idx]
                    if len(others) >= 2:
                        p2, p3 = rng.choice(others, size=2, replace=False)
                    else:
                        p2 = p3 = parent_idx
                        
                    F_factor = 0.5 + 0.5 * rng.random()
                    x_trial = X[parent_idx] + F_factor * (X[p2] - X[p3])
                    x_trial = np.clip(x_trial, lo, hi)
                    
                    f_trial = problem(x_trial)
                    used += 1
                    all_points.append(f_trial)
                    
                    # Replacement: if trial is non-dominated in the island, replace the one it dominates or the worst
                    # Simple: replace the individual with the worst hypervolume contribution? 
                    # Too expensive. Just replace the worst in terms of distance to origin?
                    # Or random replacement if trial is better than random member?
                    
                    # Let's just replace the worst member of the non-dominated set if trial is better, 
                    # or a random member if trial is non-dominated.
                    # Actually, let's keep it simple: replace the member with the largest sum of objectives.
                    sums = F.sum(axis=1)
                    worst_idx = np.argmax(sums)
                    
                    # Check if trial is better than worst
                    if np.all(f_trial <= F[worst_idx]) and np.any(f_trial < F[worst_idx]):
                        X[worst_idx] = x_trial
                        F[worst_idx] = f_trial
                    elif not dominated[np.argmin(sums)]: # If worst is dominated, replace it
                         # Find a dominated one to replace
                         dom_indices = np.where(dominated)[0]
                         if len(dom_indices) > 0:
                             replace_idx = dom_indices[0]
                             X[replace_idx] = x_trial
                             F[replace_idx] = f_trial

        # Combine all points
        all_points_array = np.array(all_points)
        return all_points_array
# EVOLVE-BLOCK-END

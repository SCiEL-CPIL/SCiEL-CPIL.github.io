import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Parameters
        n_islands = 8
        pop_size = 10
        total_init = n_islands * pop_size
        
        # Ensure we don't exceed budget
        if total_init >= budget:
            pop_size = max(1, budget // n_islands - 1)
            total_init = n_islands * pop_size
            
        # Use global random state to respect seed
        # The evaluator calls np.random.seed(seed)
        
        # Generate weight vectors for each island
        # Random points in the simplex
        weights = []
        for _ in range(n_islands):
            w = np.random.rand(n_obj)
            w = w / np.sum(w)
            weights.append(w)
            
        # Initialize islands
        islands = []
        for i in range(n_islands):
            X = lo + (hi - lo) * np.random.rand(pop_size, dim)
            F = np.array([problem(x) for x in X])
            islands.append({
                'X': X, 
                'F': F, 
                'w': weights[i]
            })
            
        used = total_init
        
        # Main Loop
        while used < budget:
            # Pick island
            island_idx = np.random.randint(n_islands)
            island = islands[island_idx]
            w = island['w']
            
            # Calculate weighted sums for selection
            # F is (N, 3), w is (3,)
            weighted_sums = np.sum(island['F'] * w, axis=1)
            
            # Select parent: Tournament Selection (size 2) on weighted sums
            # Pick 2 random indices
            idx1 = np.random.randint(pop_size)
            idx2 = np.random.randint(pop_size)
            if weighted_sums[idx1] < weighted_sums[idx2]:
                parent_idx = idx1
            else:
                parent_idx = idx2
                
            parent = island['X'][parent_idx]
            
            # Mutation
            # Adaptive step size
            progress = 1.0 - (used / budget)
            step_size = 0.1 * (0.5 + 0.5 * progress)
            mutant = parent + step_size * (hi - lo) * np.random.randn(dim)
            mutant = np.clip(mutant, lo, hi)
            
            f_mutant = problem(mutant)
            used += 1
            
            # Calculate weighted sum for mutant
            ws_mutant = np.sum(f_mutant * w)
            
            # Find worst individual in island (max weighted sum)
            worst_idx = np.argmax(weighted_sums)
            
            # Replacement
            if ws_mutant < weighted_sums[worst_idx]:
                # Replace worst
                island['X'][worst_idx] = mutant
                island['F'][worst_idx] = f_mutant
            else:
                # With small probability, replace a random individual to maintain diversity
                if np.random.rand() < 0.05:
                    rand_idx = np.random.randint(pop_size)
                    island['X'][rand_idx] = mutant
                    island['F'][rand_idx] = f_mutant

        return None

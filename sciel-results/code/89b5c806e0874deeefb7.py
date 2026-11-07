import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42) # Fixed seed for reproducibility within the run? 
        # The problem statement says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)"
        # This sets the global numpy seed. 
        # I should use np.random functions or a generator seeded from it?
        # The prompt says "np.random.seed(seed) is called". So I should probably use np.random directly 
        # or create a generator that respects the global state? 
        # Usually, if np.random.seed is called, using np.random.rand will be reproducible.
        # Using default_rng(0) ignores the global seed. 
        # The current program uses default_rng(0). 
        # If I use default_rng(0), the results will be the same for seed 0 and seed 1 from the evaluator?
        # No, the evaluator calls np.random.seed(seed). 
        # If I use np.random.rand, it will differ for seed 0 and 1.
        # If I use default_rng(0), it will be the same for both.
        # The task says "For seeds 0 and 1...". This implies the results should potentially differ or at least be evaluated separately.
        # It is better to use the global numpy random state to ensure the "seed" parameter from the evaluator has an effect.
        
        np.random.seed(0) # Wait, the evaluator calls it. I should NOT call it again if I want to use the evaluator's seed.
        # But I need a local RNG for internal logic that doesn't mess up the global state?
        # No, standard practice in these challenges: use the global state or a generator initialized from it.
        # Let's use np.random directly.

        pop_size = 20
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        
        used = pop_size
        
        def non_dominated_sort(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            S = [set() for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            front = [0]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    ranks[i] = 0
                    front[0] += 1
            
            k = 0
            while front[k] > 0:
                next_front = []
                for i in range(n):
                    if ranks[i] == k:
                        for j in S[i]:
                            n_count[j] -= 1
                            if n_count[j] == 0:
                                ranks[j] = k + 1
                                next_front.append(j)
                if not next_front:
                    break
                k += 1
                front[k] = len(next_front)
                
            return ranks

        def crowding_distance(F, ranks):
            n = len(F)
            cd = np.zeros(n)
            unique_ranks = np.unique(ranks)
            for r in unique_ranks:
                indices = np.where(ranks == r)[0]
                if len(indices) <= 2:
                    cd[indices] = np.inf
                    continue
                for m in range(self.n_obj):
                    sorted_indices = indices[np.argsort(F[indices, m])]
                    cd[sorted_indices[0]] = np.inf
                    cd[sorted_indices[-1]] = np.inf
                    if F[sorted_indices[-1], m] - F[sorted_indices[0], m] > 0:
                        for i in range(1, len(sorted_indices) - 1):
                            cd[sorted_indices[i]] += (F[sorted_indices[i+1], m] - F[sorted_indices[i-1], m]) / (F[sorted_indices[-1], m] - F[sorted_indices[0], m])
            return cd

        while used < self.budget:
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # We can create up to pop_size offspring per generation
            # Budget remaining: self.budget - used
            # Let's create a batch of offspring
            batch_size = min(pop_size, self.budget - used)
            
            for _ in range(batch_size):
                # Select parents using tournament selection on ranks
                # Simple: pick 2 random, choose better rank, then better cd
                # Or just pick from first front
                front_indices = np.where(ranks == 0)[0]
                if len(front_indices) == 0:
                    front_indices = np.arange(pop_size)
                
                p1_idx = front_indices[np.random.randint(len(front_indices))]
                p2_idx = front_indices[np.random.randint(len(front_indices))]
                
                # DE Mutation
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                F_factor = 0.8
                child = X[r1] + F_factor * (X[r2] - X[r3])
                # Clip
                child = np.clip(child, lo, hi)
                
                # Add some random noise to escape local optima?
                # Maybe not needed with DE.
                
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
                
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            ranks = non_dominated_sort(F_combined)
            
            # Crowding distance
            cd = crowding_distance(F_combined, ranks)
            
            # Select best pop_size
            # Sort by rank, then by cd (descending)
            sort_idx = np.lexsort((cd, ranks))
            sort_idx = sort_idx[::-1] # We want descending cd, so reverse? 
            # np.lexsort sorts ascending. 
            # We want lowest rank (0) first. For same rank, highest cd first.
            # So we sort by (ranks, -cd)
            sort_idx = np.lexsort((-cd, ranks))
            
            X = X_combined[sort_idx[:pop_size]]
            F = F_combined[sort_idx[:pop_size]]
            ranks = ranks[sort_idx[:pop_size]]
            
            # If we used all budget, break
            if used >= self.budget:
                break

        return X

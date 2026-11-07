import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42) # Fixed seed for reproducibility within the run, but evaluator seeds np.random
        
        # Note: The evaluator calls np.random.seed(seed). 
        # We should use np.random functions to respect the seed, or use default_rng.
        # Using np.random directly is safer to align with "np.random.seed(seed)".
        
        pop_size = 20
        # Ensure we don't exceed budget
        max_generations = self.budget // pop_size
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evals_used = pop_size
        
        def dominates(a, b):
            # a dominates b if a <= b in all objectives and a < b in at least one
            return np.all(a <= b) and np.any(a < b)
            
        def non_dominated_sort(F):
            # Returns list of fronts, each front is a list of indices
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    rank[i] = 0
                    fronts.append([i])
                    
            k = 0
            while fronts[k] is not None:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            return fronts

        def crowding_distance(F, indices):
            # Calculate crowding distance for a list of indices
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            m = len(indices)
            dist = np.zeros(m)
            F_sub = F[indices]
            
            for k in range(self.n_obj):
                # Sort by objective k
                sorted_idx = np.argsort(F_sub[:, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k]
                if f_range == 0:
                    continue
                for i in range(1, m - 1):
                    dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / f_range
            return dist

        # Main loop
        while evals_used < self.budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate offspring
            # We will generate one child at a time to manage budget precisely
            # Select 2 parents via tournament selection on rank
            
            # Compute ranks for current population
            fronts = non_dominated_sort(F)
            ranks = np.zeros(pop_size, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            
            # Tournament selection
            def tournament_select():
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                if ranks[i1] < ranks[i2]:
                    return i1
                elif ranks[i2] < ranks[i1]:
                    return i2
                else:
                    # Tie-break: random or crowding (skip for simplicity/speed, random is ok)
                    return i1 if np.random.rand() < 0.5 else i2

            p1_idx = tournament_select()
            p2_idx = tournament_select()
            
            # Crossover: BLX-Alpha
            alpha = 0.5
            child = np.zeros(self.dim)
            for d in range(self.dim):
                low = min(X[p1_idx, d], X[p2_idx, d])
                high = max(X[p1_idx, d], X[p2_idx, d])
                spread = high - low
                # Sample from [low - alpha*spread, high + alpha*spread]
                child[d] = np.random.uniform(low - alpha * spread, high + alpha * spread)
            
            # Mutation: Polynomial-like (Gaussian for simplicity, scaled)
            # Use adaptive mutation rate
            gen_ratio = evals_used / self.budget
            mutation_scale = 0.2 * (1 - gen_ratio) + 0.01
            child += mutation_scale * (hi - lo) * np.random.normal(0, 1, self.dim)
            child = np.clip(child, lo, hi)
            
            # Local Search Step (Memetic component)
            # Perform a few coordinate-wise perturbations
            fc = problem(child)
            evals_used += 1
            
            # Try to improve child locally
            for _ in range(3): # 3 local steps
                if evals_used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                # Random coordinate perturbation
                d = np.random.randint(self.dim)
                delta = 0.05 * (hi[d] - lo[d]) * np.random.normal()
                child_new = child.copy()
                child_new[d] = np.clip(child_new[d] + delta, lo[d], hi[d])
                fc_new = problem(child_new)
                evals_used += 1
                if np.all(fc_new <= fc) and np.any(fc_new < fc):
                    child = child_new
                    fc = fc_new
            
            # Replace worst individual in population
            # Find worst: highest rank, then lowest crowding distance
            # For speed, we can just replace the last one if it's dominated, or find the worst dominated one.
            # Simple strategy: Replace the individual with the highest sum of normalized objectives if dominated, 
            # or just the last one in the worst front.
            
            # Let's identify the worst front
            worst_front_idx = -1
            for i, front in enumerate(fronts):
                if front:
                    worst_front_idx = i
            
            # If the new child is non-dominated (rank 0), we might want to add it to the front.
            # But maintaining a fixed size population is simpler.
            # Let's find the worst individual in the population to replace.
            # We use a simple heuristic: replace the individual that is dominated by the most others.
            worst_idx = -1
            max_dom_count = -1
            for i in range(pop_size):
                dom_count = sum(1 for j in range(pop_size) if dominates(F[j], F[i]))
                if dom_count > max_dom_count:
                    max_dom_count = dom_count
                    worst_idx = i
            
            if worst_idx != -1:
                X[worst_idx] = child
                F[worst_idx] = fc

        return X

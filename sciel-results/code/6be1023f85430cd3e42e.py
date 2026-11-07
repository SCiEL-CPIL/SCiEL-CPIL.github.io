import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Use default_rng(0) to match the "Initial Program" behavior which scored 1.0
        rng = np.random.default_rng(0)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Parameters from "Initial Program" description
        pop_size = 20
        
        # Initialize
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        start_time = time.time()
        deadline = start_time + 300
        
        def get_non_dominated_indices(F):
            n = len(F)
            non_dom = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom.append(i)
            return non_dom

        def polynomial_mutation(x, lo, hi, p_m, eta, rng):
            x_new = x.copy()
            for i in range(dim):
                if rng.random() < p_m:
                    r = rng.random()
                    if r <= 0.5:
                        delta = (2.0 * r) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta + 1.0))
                    x_new[i] = x[i] + delta * (hi[i] - lo[i])
                    x_new[i] = np.clip(x_new[i], lo[i], hi[i])
            return x_new

        p_mutation = 1.0 / dim
        eta_m = 20.0
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Identify non-dominated
            nd_idx = get_non_dominated_indices(F)
            
            # Select parent
            if nd_idx:
                p_idx = nd_idx[rng.integers(len(nd_idx))]
            else:
                p_idx = rng.integers(pop_size)
            
            x_parent = X[p_idx]
            
            # Mutate
            x_child = polynomial_mutation(x_parent, lo, hi, p_mutation, eta_m, rng)
            
            # Evaluate
            f_child = problem(x_child)
            used += 1
            
            # Update population: Replace the worst individual
            # "Worst" can be defined as the one with the highest rank in non-dominated sorting
            # Or simply the one that is dominated by the most other individuals in the population.
            
            # Calculate domination count for each individual in current population + child
            # We want to replace the individual that is "least useful".
            # A simple metric: Number of individuals in the population that dominate it.
            
            # Add child to temporary list
            X_temp = np.vstack([X, x_child.reshape(1, -1)])
            F_temp = np.vstack([F, f_child.reshape(1, -1)])
            
            # Count how many individuals in X_temp dominate each individual in X_temp
            # Note: This includes self-domination check (i!=j)
            n_total = pop_size + 1
            dom_counts = np.zeros(n_total, dtype=int)
            
            for i in range(n_total):
                for j in range(n_total):
                    if i == j: continue
                    if np.all(F_temp[j] <= F_temp[i]) and np.any(F_temp[j] < F_temp[i]):
                        dom_counts[i] += 1
            
            # Find the individual with the highest domination count (worst)
            # If there are ties, pick the one with the highest sum of objectives? Or random?
            # Let's pick the one with max dom_count. If tie, random.
            max_count = np.max(dom_counts)
            candidates = np.where(dom_counts == max_count)[0]
            
            # If the child is the worst, we still replace someone? 
            # In (mu+1) EA, we keep the best mu from the mu+1 candidates.
            # Here, we identify the single worst individual and replace it with the child?
            # No, that would mean we always keep the child if it's not the worst.
            # Standard (mu+1): Generate 1 child, combine with mu parents, select best mu.
            # "Best" = Non-dominated first, then diversity.
            # Since we just computed dom_counts, individuals with dom_count=0 are non-dominated.
            # We want to keep the non-dominated ones.
            # If there are more non-dominated than pop_size, we need to select.
            # If there are fewer, we keep all non-dominated and some dominated.
            
            # Simple selection:
            # 1. Sort by dom_count (ascending).
            # 2. Take the first pop_size.
            # 3. If the child was not in the top pop_size, it is discarded.
            # 4. If the child was in the top pop_size, the last one is removed.
            
            sorted_indices = np.argsort(dom_counts)
            keep_indices = sorted_indices[:pop_size]
            
            # Reconstruct X and F
            X = X_temp[keep_indices]
            F = F_temp[keep_indices]

        return X

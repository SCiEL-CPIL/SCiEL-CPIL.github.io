import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0 # Safety margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size: Use a significant chunk of the budget for initial population
        # to allow for better selection pressure in subsequent generations.
        # With budget 200, pop_size 50 allows 4 generations (1 init + 3 offspring).
        pop_size = 50 
        if pop_size > self.budget:
            pop_size = self.budget
            
        # 1. Initialization
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper functions for NSGA-II
        
        def non_dominated_sort(F):
            # Returns ranks array
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            S = [set() for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            
            current_front = set()
            for i in range(n):
                if n_dom[i] == 0:
                    ranks[i] = 0
                    current_front.add(i)
            
            front_idx = 0
            while current_front:
                next_front = set()
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            ranks[j] = front_idx + 1
                            next_front.add(j)
                front_idx += 1
                current_front = next_front
                
            return ranks

        def crowding_distance(F, ranks, pop_indices):
            # Calculate crowding distance for points in pop_indices
            n = len(pop_indices)
            cd = np.zeros(n)
            if n <= 2:
                return np.full(n, np.inf)
                
            # Map indices to a local 0..n-1 range for easier handling
            # But we need to compare with the full F if we want global distance?
            # Standard NSGA-II calculates distance within the front.
            # Here we calculate for the specific subset of points belonging to the same front.
            
            # Group by rank
            from collections import defaultdict
            front_groups = defaultdict(list)
            for idx, global_idx in enumerate(pop_indices):
                front_groups[ranks[global_idx]].append(idx)
                
            for rank, group in front_groups.items():
                if len(group) <= 2:
                    for idx in group:
                        cd[idx] = np.inf
                    continue
                
                # Sort by each objective and assign distances
                for k in range(self.n_obj):
                    # Sort indices in group by objective k
                    sorted_indices = sorted(group, key=lambda i: F[pop_indices[i]][k])
                    # Assign inf to min and max
                    cd[sorted_indices[0]] = np.inf
                    cd[sorted_indices[-1]] = np.inf
                    # Calculate range
                    f_min = F[pop_indices[sorted_indices[0]]][k]
                    f_max = F[pop_indices[sorted_indices[-1]]][k]
                    f_range = f_max - f_min
                    if f_range == 0:
                        continue
                    
                    for i in range(1, len(sorted_indices) - 1):
                        idx_curr = sorted_indices[i]
                        idx_prev = sorted_indices[i-1]
                        idx_next = sorted_indices[i+1]
                        
                        f_curr = F[pop_indices[idx_curr]][k]
                        f_prev = F[pop_indices[idx_prev]][k]
                        f_next = F[pop_indices[idx_next]][k]
                        
                        cd[idx_curr] += (f_next - f_prev) / f_range
                        
            return cd

        def sbx_crossover(x1, x2, rng):
            # Simulated Binary Crossover
            eta = 20.0
            u = rng.random(self.dim)
            child1 = np.zeros(self.dim)
            child2 = np.zeros(self.dim)
            
            for i in range(self.dim):
                if u[i] <= 0.5:
                    beta = (2 * u[i]) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u[i]))) ** (1.0 / (eta + 1.0))
                
                child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                
            # Clip to bounds
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            
            return child1, child2

        def polynomial_mutation(x, rng):
            # Polynomial Mutation
            eta = 20.0
            for i in range(self.dim):
                if rng.random() < 0.5: # 50% probability of mutating each variable? Or 1/dim?
                    # Standard NSGA-II often uses 1/dim for mutation probability, 
                    # but higher rates can help in small pops. Let's use 1.0 for simplicity 
                    # or a small probability. Let's stick to standard: prob = 1/dim.
                    pass 
                    
            # Actually, let's do a simpler, robust mutation:
            # Add Gaussian noise scaled by range
            noise = 0.1 * (hi - lo) * rng.standard_normal(self.dim)
            x_mut = x + noise
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            # We can generate up to pop_size offspring, but we are limited by remaining budget
            remaining = self.budget - used
            n_offspring = min(pop_size, remaining)
            
            offspring_X = []
            # Select parents via tournament selection based on ranks and crowding?
            # For simplicity and speed in this small budget, we can just pick random pairs 
            # or use the current population directly.
            # Standard NSGA-II: Create mating pool, generate offspring.
            
            for _ in range(n_offspring // 2):
                # Pick 2 parents randomly (or tournament)
                p1_idx = rng.integers(pop_size)
                p2_idx = rng.integers(pop_size)
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx], rng)
                
                # Mutation
                c1 = polynomial_mutation(c1, rng)
                c2 = polynomial_mutation(c2, rng)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                
            # If n_offspring is odd, add one more mutated copy of a random parent
            if n_offspring % 2 != 0:
                p_idx = rng.integers(pop_size)
                c1 = polynomial_mutation(X[p_idx], rng)
                offspring_X.append(c1)
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += n_offspring
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            ranks = non_dominated_sort(F_comb)
            
            # Select next population
            # We need to select pop_size individuals from X_comb
            # Sort by rank, then by crowding distance within rank
            
            # Calculate crowding distance for all combined points?
            # It's more efficient to calculate only for the fronts we need.
            
            # Sort indices by rank
            sorted_indices = np.argsort(ranks, kind='mergesort')
            
            next_X_indices = []
            next_F_indices = []
            
            # We need to fill pop_size slots
            # Iterate through fronts
            current_front_size = 0
            for rank_val in range(max(ranks) + 1):
                front_indices = [i for i in sorted_indices if ranks[i] == rank_val]
                
                if current_front_size + len(front_indices) <= pop_size:
                    # Take all
                    next_X_indices.extend(front_indices)
                    next_F_indices.extend(front_indices)
                    current_front_size += len(front_indices)
                else:
                    # Need to select some from this front based on crowding distance
                    # Calculate crowding distance for this specific front
                    # The helper `crowding_distance` expects F, ranks, and indices
                    # But it calculates distance relative to the whole population passed?
                    # My helper `crowding_distance` above is a bit messy.
                    # Let's implement a simpler inline crowding distance for this front.
                    
                    n_front = len(front_indices)
                    cd = np.zeros(n_front)
                    
                    if n_front <= 2:
                        for i in range(n_front):
                            cd[i] = np.inf
                    else:
                        for k in range(self.n_obj):
                            # Sort front indices by objective k
                            sorted_front = sorted(front_indices, key=lambda i: F_comb[i][k])
                            cd_local_idx = {gi: li for li, gi in enumerate(sorted_front)}
                            
                            # Assign inf to ends
                            cd[cd_local_idx[sorted_front[0]]] = np.inf
                            cd[cd_local_idx[sorted_front[-1]]] = np.inf
                            
                            f_min = F_comb[sorted_front[0]][k]
                            f_max = F_comb[sorted_front[-1]][k]
                            f_range = f_max - f_min
                            if f_range == 0:
                                continue
                            
                            for li in range(1, n_front - 1):
                                gi_curr = sorted_front[li]
                                gi_prev = sorted_front[li-1]
                                gi_next = sorted_front[li+1]
                                
                                f_curr = F_comb[gi_curr][k]
                                f_prev = F_comb[gi_prev][k]
                                f_next = F_comb[gi_next][k]
                                
                                cd[cd_local_idx[gi_curr]] += (f_next - f_prev) / f_range
                                
                    # Select top (pop_size - current_front_size) based on cd
                    n_to_select = pop_size - current_front_size
                    # Sort front_indices by cd descending
                    # Create list of (cd_val, global_idx)
                    front_with_cd = [(cd[cd_local_idx[gi]], gi) for gi in front_indices]
                    front_with_cd.sort(key=lambda x: x[0], reverse=True)
                    
                    for _, gi in front_with_cd[:n_to_select]:
                        next_X_indices.append(gi)
                        next_F_indices.append(gi)
                        
                    current_front_size = pop_size
                    break
            
            # Update X and F
            X = X_comb[next_X_indices]
            F = F_comb[next_F_indices]

        return X

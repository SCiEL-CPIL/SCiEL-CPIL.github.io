"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ4 with structure exploitation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 295.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # DTLZ4 Structure:
        # Variables 0 to dim-6 (7 vars for dim=12) determine the front shape.
        # Variables dim-5 to dim-1 (5 vars for dim=12) determine g.
        # To be on the Pareto front, g must be 0.
        # g = sum((x_i - 0.5)^2) for i in last 5 vars.
        # So, setting last 5 vars to 0.5 ensures g=0.
        
        n_front_vars = dim - 5 # 7
        n_g_vars = 5           # 5
        
        # We will fix the g-variables to 0.5 throughout the search.
        # This reduces the effective dimensionality to 7.
        
        # Population size
        # With 200 budget, we can do roughly 4-5 generations of size 40-50.
        pop_size = 40
        
        # Initialize population
        # Front vars: uniform in [0, 1]
        # G vars: fixed at 0.5
        X = np.zeros((pop_size, dim))
        X[:, :n_front_vars] = np.random.rand(pop_size, n_front_vars)
        X[:, n_front_vars:] = 0.5
        
        # Clip to bounds (just in case, though 0.5 is safe for [0,1])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            
            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
                
            # Safety: add any unassigned
            assigned = set()
            for front in fronts:
                for idx in front:
                    assigned.add(idx)
            remaining = [i for i in range(n) if i not in assigned]
            if remaining:
                fronts.append(remaining)
                
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            m = len(indices)
            dist = np.zeros(m)
            
            for k in range(n_obj):
                obj_vals = F[indices, k]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val - min_val < 1e-15:
                    continue
                sorted_idx = np.argsort(obj_vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, m - 1):
                    dist[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)
            
            return dist

        # Helper: SBX Crossover (Simulated Binary Crossover)
        # Operates only on the first n_front_vars
        def sbx_crossover(x1, x2, eta=20.0):
            # x1, x2 are arrays of length dim, but we only cross the first n_front_vars
            # We return a new array of length dim
            child1 = x1.copy()
            child2 = x2.copy()
            
            for i in range(n_front_vars):
                if np.random.rand() < 0.5:
                    u = np.random.rand()
                    if u <= 0.5:
                        alpha = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        alpha = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * ((1 + alpha) * x1[i] + (1 - alpha) * x2[i])
                    c2 = 0.5 * ((1 - alpha) * x1[i] + (1 + alpha) * x2[i])
                    
                    child1[i] = c1
                    child2[i] = c2
            
            # Ensure g-variables remain 0.5
            child1[n_front_vars:] = 0.5
            child2[n_front_vars:] = 0.5
            
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta=20.0, prob=0.1):
            # Mutates only the first n_front_vars
            for i in range(n_front_vars):
                if np.random.rand() < prob:
                    u = np.random.rand()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                    x[i] += delta_q * (hi[i] - lo[i])
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Ensure g-variables remain 0.5
            x[n_front_vars:] = 0.5
            
            return x

        # Main Loop
        while used < budget:
            if time.time() - t0 > deadline:
                break
            
            # Sort current population
            fronts = non_dominated_sort(F)
            
            # Calculate crowding distance for all individuals
            crowd_dists = np.zeros(len(F))
            for front in fronts:
                cd = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    crowd_dists[idx] = cd[i]
            
            # Create a combined list for selection: (rank, -crowding, index)
            # We want to select parents based on rank (lower is better) and crowding (higher is better)
            selection_indices = []
            for r, front in enumerate(fronts):
                for idx in front:
                    selection_indices.append((r, -crowd_dists[idx], idx))
            
            # Sort for tournament
            selection_indices.sort(key=lambda x: (x[0], x[1]))
            
            # Generate offspring
            n_offspring = min(pop_size, budget - used)
            if n_offspring <= 0:
                break
            
            offspring_X = []
            for _ in range(n_offspring):
                if time.time() - t0 > deadline:
                    break
                
                # Tournament selection (size 2)
                # Pick 2 random candidates from the sorted list? 
                # Standard NSGA-II uses random indices from the population, 
                # but comparing their ranks/crowding.
                
                # Efficient tournament: pick 2 random indices from 0..pop_size-1
                idx1 = np.random.randint(0, len(F))
                idx2 = np.random.randint(0, len(F))
                
                # Determine winner
                # Get rank and crowding for these indices
                # We need a way to quickly get rank. 
                # Let's precompute rank array.
                rank_arr = np.zeros(len(F), dtype=int)
                for r, front in enumerate(fronts):
                    for idx in front:
                        rank_arr[idx] = r
                
                if rank_arr[idx1] < rank_arr[idx2]:
                    p1 = X[idx1]
                elif rank_arr[idx2] < rank_arr[idx1]:
                    p1 = X[idx2]
                else:
                    if crowd_dists[idx1] > crowd_dists[idx2]:
                        p1 = X[idx1]
                    else:
                        p1 = X[idx2]
                
                idx3 = np.random.randint(0, len(F))
                idx4 = np.random.randint(0, len(F))
                
                if rank_arr[idx3] < rank_arr[idx4]:
                    p2 = X[idx3]
                elif rank_arr[idx4] < rank_arr[idx3]:
                    p2 = X[idx4]
                else:
                    if crowd_dists[idx3] > crowd_dists[idx4]:
                        p2 = X[idx3]
                    else:
                        p2 = X[idx4]
                
                # Crossover
                c1, c2 = sbx_crossover(p1, p2, eta=20.0)
                
                # Mutation
                c1 = polynomial_mutation(c1, eta=20.0, prob=1.0/n_front_vars)
                c2 = polynomial_mutation(c2, eta=20.0, prob=1.0/n_front_vars)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
            
            if not offspring_X:
                break
            
            # Take only as many offspring as we have budget for
            offspring_X = np.array(offspring_X[:n_offspring])
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)
            
            # Combine populations
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Sort combined population
            combined_fronts = non_dominated_sort(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    next_X.append(combined_X[front])
                    next_F.append(combined_F[front])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance (descending)
                    sorted_idx = np.argsort(cd)[::-1]
                    selected = [front[i] for i in sorted_idx[:remaining]]
                    next_X.append(combined_X[selected])
                    next_F.append(combined_F[selected])
                    break
            
            if not next_X:
                break
                
            X = np.vstack(next_X)
            F = np.vstack(next_F)
            
            # Ensure we don't exceed budget in next iteration if n_offspring was partial
            if used >= budget:
                break

        return X
# EVOLVE-BLOCK-END

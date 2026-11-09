import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0 # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        # We want to use the budget efficiently.
        # Let pop_size be such that we can do several generations.
        # 200 budget. If pop_size=20, init=20, offspring=20/gen. (200-20)/20 = 9 gens.
        # If pop_size=10, init=10, offspring=10/gen. (200-10)/10 = 19 gens.
        # Smaller population, more generations might be better for convergence?
        # Or larger population, fewer generations for diversity?
        # For DTLZ1, diversity is key. Let's stick with pop_size=20.
        pop_size = 20
        
        # Latin Hypercube Sampling for initialization
        def lhs_sample(n, dim):
            # Generate n points in [0,1]^dim using LHS
            # Simple LHS: for each dimension, permute the grid
            indices = np.arange(n)
            samples = np.zeros((n, dim))
            for d in range(dim):
                perm = np.random.permutation(n)
                samples[:, d] = (perm + np.random.random(n)) / n
            return samples

        X = lhs_sample(pop_size, dim)
        X = lo + X * (hi - lo)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        if used >= budget:
            return F
            
        # NSGA-II parameters
        eta_c = 15.0
        pc = 1.0  # Crossover probability
        eta_m = 20.0
        pm = 1.0 / dim  # Mutation probability
        
        def non_dominated_sort(F):
            """
            Returns a list of fronts, each front is a list of indices.
            """
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []
            
            # Precompute dominance relationships? 
            # For n=40 (pop+offspring), O(n^2 * n_obj) is fine.
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            
            current_front = [i for i in range(n) if domination_count[i] == 0]
            rank = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
                rank += 1
                
            return fronts

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for points in front_indices.
            Returns a dictionary {index: distance}
            """
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            distances = {i: 0.0 for i in front_indices}
            for k in range(n_obj):
                f_k = F[front_indices, k]
                min_val = np.min(f_k)
                max_val = np.max(f_k)
                if max_val - min_val < 1e-10:
                    continue
                # Sort indices by objective k
                sorted_idx = np.argsort(f_k)
                # End points have infinite distance
                distances[front_indices[sorted_idx[0]]] = float('inf')
                distances[front_indices[sorted_idx[-1]]] = float('inf')
                
                for i in range(1, len(sorted_idx) - 1):
                    idx = front_indices[sorted_idx[i]]
                    prev_val = f_k[sorted_idx[i-1]]
                    next_val = f_k[sorted_idx[i+1]]
                    distances[idx] += (next_val - prev_val) / (max_val - min_val)
                    
            return distances

        def tournament_select(X, F, ranks, distances, k=2):
            """
            Select two parents using binary tournament.
            Returns indices of two parents.
            """
            n = len(F)
            idx1 = np.random.randint(n)
            idx2 = np.random.randint(n)
            # Compare idx1 and idx2
            if ranks[idx1] < ranks[idx2]:
                p1 = idx1
            elif ranks[idx1] > ranks[idx2]:
                p1 = idx2
            else:
                if distances[idx1] > distances[idx2]:
                    p1 = idx1
                else:
                    p1 = idx2
                    
            idx3 = np.random.randint(n)
            idx4 = np.random.randint(n)
            if ranks[idx3] < ranks[idx4]:
                p2 = idx3
            elif ranks[idx3] > ranks[idx4]:
                p2 = idx4
            else:
                if distances[idx3] > distances[idx4]:
                    p2 = idx3
                else:
                    p2 = idx4
                    
            return p1, p2

        def sbx_crossover(x1, x2):
            if np.random.random() > pc:
                return x1.copy(), x2.copy()
            
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            
            for i in range(dim):
                if np.random.random() < 0.5:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
            return child1, child2

        def polynomial_mutation(x):
            x_mut = x.copy()
            for i in range(dim):
                if np.random.random() < pm:
                    u = np.random.random()
                    if u < 0.5:
                        delta1 = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    
                    x_mut[i] += delta1 * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        while used < budget:
            if time.time() > deadline:
                break
                
            # Determine offspring size
            # We can use the remaining budget, but capped by pop_size to keep population stable
            # Actually, standard NSGA-II uses offspring_size = pop_size
            offspring_size = min(pop_size, budget - used)
            if offspring_size == 0:
                break
            
            # Generate offspring
            offspring_X = np.zeros((offspring_size, dim))
            for i in range(offspring_size):
                p1, p2 = tournament_select(X, F, ranks, distances)
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                offspring_X[i] = c1
                # Note: We are discarding c2 to save evaluations, 
                # effectively using a "1-child" mutation/crossover scheme 
                # or just taking one child. 
                # Standard NSGA-II generates 2 children per pair. 
                # If we generate 2 children, we need 2 evals.
                # Let's stick to generating 1 child per iteration to simplify 
                # and ensure we don't exceed budget.
                # Or we can generate 2 and evaluate both if budget allows.
                # Let's try generating 2 children if budget allows 2 more evals.
                # But for simplicity and strict budget control:
                # We will just take c1. This is a form of mutation/crossover.
            
            # Evaluate offspring
            F_offspring = np.array([problem(x) for x in offspring_X])
            used += offspring_size
            
            if used >= budget:
                # Combine and select best pop_size from combined
                X_all = np.vstack([X, offspring_X])
                F_all = np.vstack([F, F_offspring])
                
                fronts = non_dominated_sort(F_all)
                ranks = np.zeros(len(F_all), dtype=int)
                for rank_idx, front in enumerate(fronts):
                    for idx in front:
                        ranks[idx] = rank_idx
                
                # Select best pop_size
                selected_indices = []
                for rank_idx, front in enumerate(fronts):
                    if len(selected_indices) + len(front) <= pop_size:
                        selected_indices.extend(front)
                    else:
                        # Need to pick some from this front
                        needed = pop_size - len(selected_indices)
                        # Calculate crowding distance for this front
                        dists = crowding_distance(F_all, front)
                        # Sort front by distance descending
                        front_sorted = sorted(front, key=lambda idx: dists[idx], reverse=True)
                        selected_indices.extend(front_sorted[:needed])
                        break
                        
                X = X_all[selected_indices]
                F = F_all[selected_indices]
                break
            
            # If we haven't hit budget, we update population
            # Combine parents and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_offspring])
            
            # Non-dominated sort
            fronts = non_dominated_sort(F_all)
            
            # Assign ranks
            ranks = np.zeros(len(F_all), dtype=int)
            for rank_idx, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank_idx
            
            # Calculate crowding distance for the entire population?
            # Only needed for fronts that might be selected.
            # For selection in next step, we need distances for all points.
            distances = np.zeros(len(F_all))
            for front in fronts:
                dists = crowding_distance(F_all, front)
                for idx in front:
                    distances[idx] = dists[idx]
            
            # Select new population
            selected_indices = []
            for rank_idx, front in enumerate(fronts):
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    needed = pop_size - len(selected_indices)
                    dists = crowding_distance(F_all, front)
                    front_sorted = sorted(front, key=lambda idx: dists[idx], reverse=True)
                    selected_indices.extend(front_sorted[:needed])
                    break
                    
            X = X_all[selected_indices]
            F = F_all[selected_indices]
            
        return F

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0
        
        # Use global random state to respect evaluator's np.random.seed(seed)
        # This ensures different seeds yield different runs.
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size 20 allows 10 generations with budget 200.
        # This is a good balance for 12D WFG7.
        pop_size = 20
        
        # Initialize with stratified sampling for better coverage
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            for d in range(self.dim):
                X[i, d] = lo[d] + range_[d] * ((i + np.random.rand()) / pop_size)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Non-dominated sort
        def fast_non_dominated_sort(F_arr):
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[j].append(i)
                        domination_count[i] += 1
                        
            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F_arr, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            distances = np.zeros(n)
            for m in range(self.n_obj):
                f_vals = F_arr[front_indices, m]
                min_val = f_vals.min()
                max_val = f_vals.max()
                if max_val - min_val < 1e-15:
                    continue
                sorted_idx = np.argsort(f_vals)
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    distances[sorted_idx[k]] += (f_vals[sorted_idx[k + 1]] - f_vals[sorted_idx[k - 1]]) / (max_val - min_val)
            return distances

        # Helper: SBX Crossover (Single child version for 1:1 ratio in loop, or 2:1 if we wanted, but here we generate 1 child per parent pair in a loop to fill pop)
        # Actually, standard NSGA-II generates lambda offspring. 
        # If we use pop_size=20, and generate 20 children, we use 20 evals.
        # To generate 20 children, we need 10 parent pairs.
        # Let's generate 2 children per pair to be more standard, or just 1 child per pair and loop pop_size/2 times?
        # The inspiration 1 uses a 1:1 replacement strategy (generate 1 child, evaluate, replace worst). 
        # That uses 1 eval per step. 200 steps total.
        # My current structure is generation-based.
        # Let's stick to generation-based but optimize the generation loop.
        # Generate pop_size children.
        
        def sbx_crossover(x1, x2, lo, hi, eta=20.0):
            # Returns two children
            c1 = np.empty_like(x1)
            c2 = np.empty_like(x2)
            for d in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(x2[d] - x1[d]) > 1e-15:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                        c1[d] = 0.5 * ((1.0 + beta) * x1[d] + (1.0 - beta) * x2[d])
                        c2[d] = 0.5 * ((1.0 - beta) * x1[d] + (1.0 + beta) * x2[d])
                    else:
                        c1[d] = x1[d]
                        c2[d] = x2[d]
                else:
                    c1[d] = x2[d]
                    c2[d] = x1[d]
            return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

        # Helper: Tournament selection
        def tournament_select(X_arr, F_arr, fronts, crowding_dists):
            # Map index to front rank and crowding distance
            rank = np.zeros(len(F_arr), dtype=int)
            cd = np.zeros(len(F_arr))
            for r, front in enumerate(fronts):
                cd_front = crowding_dists[r]
                for i, idx in enumerate(front):
                    rank[idx] = r
                    cd[idx] = cd_front[i]
                    
            idx1 = np.random.randint(len(F_arr))
            idx2 = np.random.randint(len(F_arr))
            
            if rank[idx1] < rank[idx2]:
                return idx1
            elif rank[idx2] < rank[idx1]:
                return idx2
            else:
                return idx1 if cd[idx1] > cd[idx2] else idx2

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Calculate fronts and crowding for current population
            fronts = fast_non_dominated_sort(F)
            crowding_dists = []
            for front in fronts:
                crowding_dists.append(crowding_distance(F, front))
                
            # Generate offspring
            Q = np.empty((pop_size, self.dim))
            for i in range(0, pop_size, 2):
                parent1_idx = tournament_select(X, F, fronts, crowding_dists)
                parent2_idx = tournament_select(X, F, fronts, crowding_dists)
                parent1 = X[parent1_idx]
                parent2 = X[parent2_idx]
                
                if np.random.rand() < 0.9:
                    c1, c2 = sbx_crossover(parent1, parent2, lo, hi)
                else:
                    c1 = parent1.copy()
                    c2 = parent2.copy()
                
                # Adaptive Gaussian Mutation
                progress = used / self.budget
                # Start with larger steps, reduce over time
                # Inspiration 1 used: step = 0.3 * (1.0 - 0.8 * progress)
                step_scale = 0.3 * (1.0 - 0.8 * progress)
                
                c1 = c1 + step_scale * range_ * np.random.randn(self.dim)
                c2 = c2 + step_scale * range_ * np.random.randn(self.dim)
                
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                
                Q[i] = c1
                if i + 1 < pop_size:
                    Q[i+1] = c2
            # If pop_size is odd, the last one might be uninitialized? 
            # pop_size=20 is even.
            
            F_q = np.array([problem(x) for x in Q])
            used += pop_size
            
            # Combine
            P = np.vstack([X, Q])
            F_p = np.vstack([F, F_q])
            
            # Select next population
            fronts_new = fast_non_dominated_sort(F_p)
            crowding_dists_new = []
            for front in fronts_new:
                crowding_dists_new.append(crowding_distance(F_p, front))
                
            # Fill next population
            X_next = []
            F_next = []
            for i, front in enumerate(fronts_new):
                if len(X_next) + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        X_next.append(P[idx])
                        F_next.append(F_p[idx])
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(X_next)
                    cd_front = crowding_dists_new[i]
                    # Sort indices in front by crowding distance descending
                    sorted_indices = np.argsort(-cd_front)
                    for k in range(remaining):
                        idx = front[sorted_indices[k]]
                        X_next.append(P[idx])
                        F_next.append(F_p[idx])
                    break
                    
            X = np.array(X_next)
            F = np.array(F_next)
            
        return X

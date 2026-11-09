import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        pop_size = 20
        
        # Initialize
        X = lo + np.random.rand(pop_size, self.dim) * range_
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
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

        # Helper: Tournament selection
        def tournament_select(F_arr, fronts, crowding_dists):
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
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Calculate fronts and crowding for current population
            fronts = fast_non_dominated_sort(F)
            crowding_dists = []
            for front in fronts:
                crowding_dists.append(crowding_distance(F, front))
                
            # Select parent
            parent_idx = tournament_select(F, fronts, crowding_dists)
            parent = X[parent_idx]
            
            # Mutate
            progress = evals / self.budget
            # Step size: start wide, narrow down
            step = 0.5 * (1.0 - 0.8 * progress)
            # Add noise to step size
            step *= np.random.uniform(0.5, 1.5)
            
            child = parent + step * range_ * np.random.randn(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            evals += 1
            
            # Combine population and child
            X_new = np.vstack([X, child.reshape(1, -1)])
            F_new = np.vstack([F, fc.reshape(1, -1)])
            
            # Select next population
            fronts_new = fast_non_dominated_sort(F_new)
            crowding_dists_new = []
            for front in fronts_new:
                crowding_dists_new.append(crowding_distance(F_new, front))
                
            # Fill next population
            X_next = []
            F_next = []
            for i, front in enumerate(fronts_new):
                if len(X_next) + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        X_next.append(X_new[idx])
                        F_next.append(F_new[idx])
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(X_next)
                    cd_front = crowding_dists_new[i]
                    # Sort indices in front by crowding distance descending
                    sorted_indices = np.argsort(-cd_front)
                    for k in range(remaining):
                        idx = front[sorted_indices[k]]
                        X_next.append(X_new[idx])
                        F_next.append(F_new[idx])
                    break
                    
            X = np.array(X_next)
            F = np.array(F_next)
            
        return X

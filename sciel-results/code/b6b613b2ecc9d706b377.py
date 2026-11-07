import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        N = 20 # Population size
        
        # Initialize population
        X = lo + range_ * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_dominated = np.zeros(n)
            rank = np.zeros(n)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dominated[i] += 1
                        
            current_front = []
            front_index = 0
            for i in range(n):
                if n_dominated[i] == 0:
                    rank[i] = front_index
                    current_front.append(i)
                    
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = front_index + 1
                            next_front.append(j)
                front_index += 1
                current_front = next_front
                
            return rank

        # Helper: Crowding distance
        def crowding_distance(F, rank, indices):
            # indices are indices of points in the same front
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            cd = np.zeros(len(indices))
            for m in range(self.n_obj):
                f_vals = F[indices, m]
                min_f = np.min(f_vals)
                max_f = np.max(f_vals)
                if max_f == min_f:
                    cd[:] = np.inf
                    continue
                sorted_idx = np.argsort(f_vals)
                sorted_indices = indices[sorted_idx]
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, len(sorted_idx) - 1):
                    i = sorted_idx[k]
                    cd[i] += (f_vals[sorted_idx[k+1]] - f_vals[sorted_idx[k-1]]) / (max_f - min_f)
            return cd

        # Main loop
        while used < self.budget and (time.time() - start_time) < deadline:
            # Check time before generating offspring to avoid partial work
            if (time.time() - start_time) > deadline - 5:
                break
                
            # Select parents
            rank = non_dominated_sort(F)
            
            # Calculate crowding distance for each front
            # Group indices by rank
            fronts = {}
            for i, r in enumerate(rank):
                if r not in fronts:
                    fronts[r] = []
                fronts[r].append(i)
                
            cd = np.zeros(N)
            for r, idxs in fronts.items():
                cd[idxs] = crowding_distance(F, r, idxs)
                
            # Tournament selection for offspring
            offspring_X = np.zeros((N, self.dim))
            for i in range(N):
                p1 = self._tournament(F, rank, cd, rng)
                p2 = self._tournament(F, rank, cd, rng)
                
                # Crossover (SBX or simple blend)
                # Let's use a simple convex combination (blend crossover) for simplicity and robustness
                alpha = rng.random(self.dim)
                child = alpha * X[p1] + (1 - alpha) * X[p2]
                
                # Mutation
                # Adaptive sigma based on range
                sigma = 0.1 * range_
                child = child + sigma * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                offspring_X[i] = child
                
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += N
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Select next population
            rank_comb = non_dominated_sort(F_comb)
            
            # We need to select N individuals from 2N
            # Sort by rank, then by crowding distance within rank
            # We will greedily pick from best rank to worst
            selected_indices = []
            # Get unique ranks sorted
            unique_ranks = sorted(list(set(rank_comb)))
            
            for r in unique_ranks:
                idxs_in_front = np.where(rank_comb == r)[0]
                if len(selected_indices) + len(idxs_in_front) <= N:
                    selected_indices.extend(idxs_in_front.tolist())
                else:
                    # Need to select only some from this front
                    # Calculate crowding distance for this front
                    front_indices = idxs_in_front
                    # We need crowding distance for F_comb
                    # Re-use logic
                    cd_front = crowding_distance(F_comb, r, front_indices)
                    # Sort by crowding distance descending
                    sorted_local = np.argsort(-cd_front)
                    needed = N - len(selected_indices)
                    selected_indices.extend(front_indices[sorted_local[:needed]].tolist())
                    break
                    
            X = X_comb[selected_indices]
            F = F_comb[selected_indices]

        return X

    def _tournament(self, F, rank, cd, rng, k=2):
        indices = rng.integers(0, len(F), size=k)
        best_idx = indices[0]
        for i in indices[1:]:
            # Compare rank, then crowding distance
            if rank[i] < rank[best_idx]:
                best_idx = i
            elif rank[i] == rank[best_idx]:
                if cd[i] > cd[best_idx]:
                    best_idx = i
        return best_idx

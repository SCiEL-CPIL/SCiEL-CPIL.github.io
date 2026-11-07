import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        # With 200 budget, P=20 allows 10 generations.
        # P=10 allows 20 generations.
        # Let's try P=20 for better diversity in DE.
        pop_size = 20
        
        rng = np.random.default_rng(0)
        
        # Initialize
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size
        
        # Helper: Fast Non-Dominated Sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            
            # Vectorized dominance check
            # diff[i, j, k] = F[i, k] - F[j, k]
            # i dominates j if F[i] <= F[j] and F[i] != F[j]
            diff = F[:, np.newaxis, :] - F[np.newaxis, :, :]
            less_eq = np.all(diff <= 0, axis=2)
            not_equal = np.any(diff != 0, axis=2)
            dominates = less_eq & not_equal
            
            # dominates[i, j] is True if i dominates j
            for i in range(n):
                for j in range(n):
                    if dominates[i, j]:
                        dominated_set[i].append(j)
                        domination_count[j] += 1
            
            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            if current_front:
                fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            return fronts

        # Helper: Compute Crowding Distance
        def compute_crowding(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[front_indices, m]
                min_val = f_vals.min()
                max_val = f_vals.max()
                if max_val - min_val < 1e-14:
                    dist[:] = np.inf
                    continue
                order = np.argsort(f_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (f_vals[order[k + 1]] - f_vals[order[k - 1]]) / (max_val - min_val)
            return dist

        # Helper: Tournament Selection
        def tournament_select(rank, cd, size=2):
            indices = rng.integers(0, pop_size, size=size)
            best_idx = indices[0]
            for i in range(1, size):
                idx = indices[i]
                if rank[idx] < rank[best_idx]:
                    best_idx = idx
                elif rank[idx] == rank[best_idx]:
                    if cd[idx] > cd[best_idx]:
                        best_idx = idx
            return best_idx

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Calculate Ranks and Crowding Distances
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(pop_size, dtype=int)
            cd = np.zeros(pop_size, dtype=np.float64)
            
            for idx, front in enumerate(fronts):
                for i in front:
                    rank[i] = idx
                cd_front = compute_crowding(F, front)
                for k, i in enumerate(front):
                    cd[i] = cd_front[k]
            
            # 2. Generate Offspring
            V = np.empty((pop_size, dim))
            
            for i in range(pop_size):
                # Tournament selection for two parents
                p1_idx = tournament_select(rank, cd)
                p2_idx = tournament_select(rank, cd)
                
                # Ensure distinct parents for DE mutation
                if p1_idx == p2_idx:
                    p2_idx = (p2_idx + 1) % pop_size
                    if p1_idx == p2_idx: # unlikely but safe
                        p2_idx = (p2_idx + 1) % pop_size

                # DE/Current-to-Best/1 or similar? 
                # Let's use standard DE/rand/1 with crossover, but guided by selection.
                # Actually, let's stick to DE/rand/1 structure but use selected parents as base?
                # Or use selected parents for BLX-alpha?
                # Let's use a hybrid: Use selected parents for BLX-alpha crossover, then DE-style mutation?
                # Or simpler: Use DE/rand/1 where r1 is selected parent, r2, r3 are random?
                
                # Let's try: Base = Selected Parent 1. 
                # Mutation: Base + F * (Random1 - Random2)
                # This combines elitism (selection) with exploration (DE).
                
                r1 = p1_idx
                r2 = rng.integers(0, pop_size)
                r3 = rng.integers(0, pop_size)
                while r2 == r1 or r3 == r1 or r2 == r3:
                    r2 = rng.integers(0, pop_size)
                    r3 = rng.integers(0, pop_size)
                
                F_factor = 0.8
                V[i] = X[r1] + F_factor * (X[r2] - X[r3])
                
                # Crossover (Binomial)
                # For each dimension, with probability CR, take from mutant, else from parent
                CR = 0.5
                cross_mask = rng.random(dim) < CR
                # Ensure at least one dimension is taken from mutant
                if not np.any(cross_mask):
                    k = rng.integers(0, dim)
                    cross_mask[k] = True
                    
                V[i][cross_mask] = V[i][cross_mask]
                V[i][~cross_mask] = X[r1][~cross_mask]
                
                # Clip to bounds
                V[i] = np.clip(V[i], lo, hi)
            
            # 3. Evaluate Offspring
            F_v = np.array([problem(x) for x in V], dtype=np.float64)
            used += pop_size
            
            # 4. Environmental Selection (NSGA-II style)
            # Combine parents and offspring
            X_all = np.vstack([X, V])
            F_all = np.vstack([F, F_v])
            
            # Sort by rank, then crowding distance
            fronts_all = fast_non_dominated_sort(F_all)
            
            X_new = []
            F_new = []
            for front in fronts_all:
                if len(X_new) + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                else:
                    # Need to truncate
                    needed = pop_size - len(X_new)
                    # Compute crowding for this front
                    cd_front = compute_crowding(F_all, front)
                    # Sort by crowding distance (descending)
                    order = np.argsort(-cd_front)
                    for k in range(needed):
                        idx = front[order[k]]
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
            
            # Ensure we have exactly pop_size (safety)
            if len(X) < pop_size:
                # Fill with random points if somehow we lost too many (shouldn't happen)
                missing = pop_size - len(X)
                X_miss = lo + span * rng.random((missing, dim))
                F_miss = np.array([problem(x) for x in X_miss])
                used += missing
                X = np.vstack([X, X_miss])
                F = np.vstack([F, F_miss])
                
        return X

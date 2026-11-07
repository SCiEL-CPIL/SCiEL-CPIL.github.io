import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 40
        if self.budget < pop_size:
            pop_size = self.budget
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Fast Non-Dominated Sort
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            n_dominates = np.zeros(n, dtype=int)
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dominates[p] += 1
            
            for p in range(n):
                if n_dominates[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dominates[q] -= 1
                        if n_dominates[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            
            if not fronts[-1]:
                fronts.pop()
                
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            dist = np.zeros(len(indices))
            m = self.n_obj
            
            for k in range(m):
                sorted_idx = sorted(indices, key=lambda i: F[i, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                f_min = F[sorted_idx[0], k]
                f_max = F[sorted_idx[-1], k]
                if f_max == f_min:
                    continue
                
                for j in range(1, len(sorted_idx) - 1):
                    idx_j = sorted_idx[j]
                    idx_jm1 = sorted_idx[j-1]
                    idx_jp1 = sorted_idx[j+1]
                    dist[idx_j] += (F[idx_jp1, k] - F[idx_jm1, k]) / (f_max - f_min)
            
            return dist

        # Helper: Tournament Selection
        def tournament_select(F, fronts, k=2):
            best = -1
            best_rank = float('inf')
            best_cd = -1
            
            for _ in range(k):
                idx = np.random.randint(len(F))
                rank = 0
                for i, front in enumerate(fronts):
                    if idx in front:
                        rank = i
                        break
                # If rank is better (smaller), or same rank and higher crowding distance
                if rank < best_rank:
                    best = idx
                    best_rank = rank
                    best_cd = 0 # placeholder, we need cd for same rank
                elif rank == best_rank:
                    # Need crowding distance for comparison
                    # To avoid recalculating, we can just pick random if same rank?
                    # Better: precompute cd for the front?
                    # For efficiency in tournament, let's just pick the one with better rank,
                    # and if same rank, pick random from the tied ones for now?
                    # No, proper NSGA-II uses CD.
                    # Let's refactor to pass CD array or compute on the fly.
                    # Computing CD on the fly for every tournament is expensive.
                    # Let's precompute CD for the current population fronts.
                    pass
            
            # This implementation is slightly inefficient if we don't have CD precomputed.
            # Let's just return best based on rank, and if tie, random among tied.
            # Actually, let's simplify: pick k random, return the one with lowest rank.
            # If tie, pick random. This is a valid variant (rank-based tournament).
            
            candidates = [np.random.randint(len(F)) for _ in range(k)]
            best_idx = candidates[0]
            best_rank = float('inf')
            for idx in candidates:
                for i, front in enumerate(fronts):
                    if idx in front:
                        if i < best_rank:
                            best_rank = i
                            best_idx = idx
                        break
            return best_idx

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = []
            for _ in range(pop_size):
                # Selection
                fronts = fast_non_dominated_sort(F)
                p1_idx = tournament_select(F, fronts)
                p2_idx = tournament_select(F, fronts)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover: SBX (Simulated Binary Crossover)
                child = np.zeros(self.dim)
                for i in range(self.dim):
                    if np.random.random() < 0.5:
                        # SBX
                        if np.random.random() < 0.5:
                            beta = (2 * np.random.random()) ** (1.0 / (10 + 1))
                        else:
                            beta = (1.0 / (2 * np.random.random())) ** (1.0 / (10 + 1))
                        c1 = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                        c2 = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                        child[i] = c1 if np.random.random() < 0.5 else c2
                    else:
                        child[i] = p1[i] if np.random.random() < 0.5 else p2[i]
                
                # Mutation: Polynomial
                for i in range(self.dim):
                    if np.random.random() < 1.0 / self.dim:
                        if child[i] == lo[i] or child[i] == hi[i]:
                            # If at boundary, move inward
                            if child[i] == lo[i]:
                                child[i] = lo[i] + (hi[i] - lo[i]) * np.random.random()
                            else:
                                child[i] = hi[i] - (hi[i] - lo[i]) * np.random.random()
                        else:
                            if np.random.random() < 0.5:
                                beta = (2 * (child[i] - lo[i]) / (hi[i] - lo[i])) ** (1.0 / (20 + 1))
                            else:
                                beta = (2 * (hi[i] - child[i]) / (hi[i] - lo[i])) ** (1.0 / (20 + 1))
                            delta = (beta - 1) * (hi[i] - lo[i])
                            child[i] += delta
                        child[i] = np.clip(child[i], lo[i], hi[i])
                
                offspring_X.append(child)
            
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)
            
            # Combine
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            # Selection: NSGA-II
            fronts = fast_non_dominated_sort(F_all)
            
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    remaining = pop_size - len(new_indices)
                    cd = crowding_distance(F_all, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    new_indices.extend(sorted_front[:remaining])
                    break
            
            X = X_all[new_indices]
            F = F_all[new_indices]
            
        return X

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Parameters
        N = 10  # Population size
        if N > budget:
            N = budget

        # Initialize: mix of random interior and boundary/corner points
        X = np.empty((N, dim))
        n_rand = N // 2
        X[:n_rand] = lo + np.random.random((n_rand, dim)) * span
        for i in range(n_rand, N):
            corners = np.random.random(dim) < 0.5
            X[i] = np.where(corners, lo, hi)
        
        F = np.array([problem(x) for x in X])
        used = N

        # Helper: Fast non-dominated sort
        def fast_non_dominated_sort(F_mat):
            n = len(F_mat)
            S = [[] for _ in range(n)]
            ndom = [0] * n
            rank = [0] * n

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].append(j)
                        ndom[j] += 1
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        S[j].append(i)
                        ndom[i] += 1

            fronts = []
            current_front = [i for i in range(n) if ndom[i] == 0]
            for i in current_front:
                rank[i] = 0
            if current_front:
                fronts.append(current_front)

            k = 0
            while k < len(fronts) and fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        ndom[j] -= 1
                        if ndom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F_mat, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F_mat[indices, m]
                min_f = f_vals.min()
                max_f = f_vals.max()
                if max_f == min_f:
                    dist[:] = np.inf
                    continue
                sorted_indices = sorted(indices, key=lambda i: F_mat[i][m])
                dist[sorted_indices[0]] = np.inf
                dist[sorted_indices[-1]] = np.inf
                for k in range(1, n - 1):
                    prev = sorted_indices[k-1]
                    next_ = sorted_indices[k+1]
                    dist[sorted_indices[k]] += (F_mat[next_][m] - F_mat[prev][m]) / (max_f - min_f)
            return dist

        # Main Loop
        while used < budget and (time.time() < deadline):
            # Compute ranks and crowding distance
            fronts = fast_non_dominated_sort(F)
            ranks = np.zeros(N, dtype=int)
            for rank, front in enumerate(fronts):
                for i in front:
                    ranks[i] = rank
            
            cd = crowding_distance(F, list(range(N)))
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            n_offspring = min(N, budget - used)
            
            for _ in range(n_offspring):
                if used >= budget or time.time() > deadline:
                    break
                
                # Tournament selection of size 3
                best_idx = -1
                best_rank = float('inf')
                best_cd = -float('inf')
                
                for _ in range(3):
                    idx = np.random.randint(N)
                    if ranks[idx] < best_rank or (ranks[idx] == best_rank and cd[idx] > best_cd):
                        best_idx = idx
                        best_rank = ranks[idx]
                        best_cd = cd[idx]
                
                p1 = best_idx
                
                # Second parent
                best_idx2 = -1
                best_rank2 = float('inf')
                best_cd2 = -float('inf')
                
                for _ in range(3):
                    idx = np.random.randint(N)
                    if ranks[idx] < best_rank2 or (ranks[idx] == best_rank2 and cd[idx] > best_cd2):
                        best_idx2 = idx
                        best_rank2 = ranks[idx]
                        best_cd2 = cd[idx]
                
                p2 = best_idx2
                
                # Crossover: blend
                alpha = np.random.rand()
                child = alpha * X[p1] + (1 - alpha) * X[p2]
                
                # Adaptive mutation
                progress = used / budget
                sigma = 0.08 * (1.0 - 0.5 * progress)
                child = child + sigma * span * np.random.randn(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
                
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
            
            if used >= budget or time.time() > deadline:
                break
            
            # Combine parents and offspring
            X_new = np.vstack([X, np.array(offspring_X)])
            F_new = np.array(list(F) + offspring_F)
            total = len(X_new)
            
            # Selection for next generation: NSGA-II style
            fronts_new = fast_non_dominated_sort(F_new)
            next_X = []
            next_F = []
            
            for front in fronts_new:
                if len(next_X) + len(front) <= N:
                    for i in front:
                        next_X.append(X_new[i])
                        next_F.append(F_new[i])
                else:
                    need = N - len(next_X)
                    cd_front = crowding_distance(F_new, front)
                    sorted_front = sorted(front, key=lambda i: cd_front[i], reverse=True)
                    for i in sorted_front[:need]:
                        next_X.append(X_new[i])
                        next_F.append(F_new[i])
                    break
                    
            X = np.array(next_X)
            F = np.array(next_F)
        
        return X

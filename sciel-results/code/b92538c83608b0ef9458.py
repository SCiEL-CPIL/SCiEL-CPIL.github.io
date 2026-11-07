import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Parameters
        N = 20  # Population size
        if N > budget:
            N = budget
            
        # Initialize with random points
        X = lo + np.random.random((N, dim)) * span
        F = np.array([problem(x) for x in X])
        used = N
        
        # Track best per objective for adaptive mutation
        best_obj = F.min(axis=0)
        worst_obj = F.max(axis=0)
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F_mat):
            n = len(F_mat)
            fronts = []
            remaining = list(range(n))
            while remaining:
                non_dom = []
                for i in remaining:
                    dominated = False
                    for j in remaining:
                        if i == j:
                            continue
                        if np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                            dominated = True
                            break
                    if not dominated:
                        non_dom.append(i)
                fronts.append(non_dom)
                remaining = [i for i in remaining if i not in non_dom]
            return fronts
        
        # Helper: Crowding distance
        def crowding_distance(F_mat, indices):
            if len(indices) == 0:
                return {}
            dist = {i: 0.0 for i in indices}
            for m in range(n_obj):
                f_vals = [F_mat[i][m] for i in indices]
                min_f = min(f_vals)
                max_f = max(f_vals)
                if max_f == min_f:
                    for i in indices:
                        dist[i] = float('inf')
                    continue
                sorted_indices = sorted(indices, key=lambda i: F_mat[i][m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                for k in range(1, len(sorted_indices) - 1):
                    prev = sorted_indices[k-1]
                    next_ = sorted_indices[k+1]
                    dist[sorted_indices[k]] += (F_mat[next_][m] - F_mat[prev][m]) / (max_f - min_f)
            return dist
        
        # Main Loop
        while used < budget and (time.time() < deadline):
            # Compute ranks and crowding distance ONCE per generation
            fronts = non_dominated_sort(F)
            ranks = np.zeros(N, dtype=int)
            for rank, front in enumerate(fronts):
                for i in front:
                    ranks[i] = rank
            cd = crowding_distance(F, list(range(N)))
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(N):
                # Tournament selection of size 2 on rank, then crowding
                p1_idx = np.random.randint(N)
                p2_idx = np.random.randint(N)
                if ranks[p1_idx] < ranks[p2_idx]:
                    p1 = p1_idx
                elif ranks[p2_idx] < ranks[p1_idx]:
                    p1 = p2_idx
                else:
                    p1 = p1_idx if cd[p1_idx] > cd[p2_idx] else p2_idx
                
                p3_idx = np.random.randint(N)
                p4_idx = np.random.randint(N)
                if ranks[p3_idx] < ranks[p4_idx]:
                    p2 = p3_idx
                elif ranks[p4_idx] < ranks[p3_idx]:
                    p2 = p4_idx
                else:
                    p2 = p3_idx if cd[p3_idx] > cd[p4_idx] else p4_idx
                
                # Crossover: uniform blend
                mask = np.random.rand(dim) > 0.5
                child = np.where(mask, X[p1], X[p2])
                
                # Adaptive mutation
                progress = used / budget
                sigma = 0.15 * (1.0 - 0.7 * progress)
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
            fronts_new = non_dominated_sort(F_new)
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
            
            # Update best/worst
            best_obj = np.minimum(best_obj, F.min(axis=0))
            worst_obj = np.maximum(worst_obj, F.max(axis=0))
        
        return X

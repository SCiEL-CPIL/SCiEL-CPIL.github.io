import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return [], []
            
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = []
            front0 = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        nd[i] += 1
                if nd[i] == 0:
                    front0.append(i)
                    fronts.append([])
            
            if not front0:
                return [], [0] * n
            
            fronts[0] = front0
            k = 1
            while fronts[k - 1]:
                next_front = []
                for i in fronts[k - 1]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                    k += 1
                else:
                    break
            
            ranks = np.zeros(n, dtype=int)
            for rank, front in enumerate(fronts):
                for i in front:
                    ranks[i] = rank
            
            return fronts, ranks
        
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            idx = np.array(indices)
            F_sub = F[idx]
            dist = np.zeros(n)
            
            for k in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, k])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                f_range = F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k]
                if f_range > 0:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], k] - F_sub[sorted_idx[i - 1], k]) / f_range
            
            return dist
        
        def tournament_select(X, F, fronts, k=2):
            indices = np.random.randint(0, len(F), size=k)
            best = indices[0]
            for i in indices[1:]:
                # Compare by front rank first, then crowding
                # For simplicity in tournament, use front rank
                front_rank_best = -1
                front_rank_i = -1
                for r, front in enumerate(fronts):
                    if best in front:
                        front_rank_best = r
                        break
                    if i in front:
                        front_rank_i = r
                        break
                
                if front_rank_i < front_rank_best:
                    best = i
                elif front_rank_i == front_rank_best:
                    # Simple tie-break: lower sum of objectives
                    if np.sum(F[i]) < np.sum(F[best]):
                        best = i
            
            return X[best]
        
        def sbx_crossover(x1, x2, eta=20.0):
            x = np.zeros_like(x1)
            for i in range(dim):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x[i] = x1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x1[i] - x2[i]))
                else:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x[i] = x1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x1[i] - x2[i]))
            return np.clip(x, lo, hi)
        
        def polynomial_mutation(x, pmut=1.0/dim, eta=20.0):
            x_new = x.copy()
            for i in range(dim):
                if np.random.rand() < pmut:
                    range_i = hi[i] - lo[i]
                    if range_i < 1e-14:
                        continue
                    y = (x_new[i] - lo[i]) / range_i
                    y = np.clip(y, 0, 1)
                    u = np.random.rand()
                    if u < 0.5:
                        delta1 = (1 + 2 * (y - 0)) ** (eta + 1)
                        delta = (1 / delta1) ** (1 / (eta + 1)) - 1
                        y_new = y + delta * (1 - y)
                    else:
                        delta2 = (1 + 2 * (1 - y)) ** (eta + 1)
                        delta = 1 - (1 / delta2) ** (1 / (eta + 1))
                        y_new = y + delta * y
                    y_new = np.clip(y_new, 0, 1)
                    x_new[i] = lo[i] + y_new * range_i
            return np.clip(x_new, lo, hi)
        
        # Initialize
        pop_size = 30
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            n_off = min(pop_size, self.budget - evaluations)
            X_off = []
            for _ in range(n_off):
                fronts, _ = fast_non_dominated_sort(F)
                p1 = tournament_select(X, F, fronts)
                p2 = tournament_select(X, F, fronts)
                child = sbx_crossover(p1, p2)
                child = polynomial_mutation(child)
                X_off.append(child)
            
            X_off = np.array(X_off)
            F_off = np.array([problem(x) for x in X_off])
            evaluations += n_off
            
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            fronts, ranks = fast_non_dominated_sort(F_all)
            
            # Calculate crowding for each front
            all_indices = list(range(len(F_all)))
            # Sort by rank then crowding
            front_list = []
            for front in fronts:
                cd = crowding_distance(F_all, front)
                front_with_cd = list(zip(front, cd))
                front_with_cd.sort(key=lambda x: -x[1])
                front_list.extend(front_with_cd)
            
            # Take top pop_size
            selected = front_list[:pop_size]
            X = np.array([F_all[i] for i, _ in selected])  # Wait, this is F not X
            # Correct selection
            selected_idx = [i for i, _ in selected]
            X = X_all[selected_idx]
            F = F_all[selected_idx]
        
        return np.zeros((1, dim))

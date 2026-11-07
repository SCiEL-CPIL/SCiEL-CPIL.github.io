import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(15, max(5, self.budget // 12))
        
        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [0] * n
            front = [0]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    rank[i] = 0
                    front[0] += 1
            
            k = 0
            while front[k] > 0:
                next_front = []
                for i in S_range(front, rank, k):
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if k < len(front):
                    front[k] = len(next_front)
                else:
                    front.append(len(next_front))
            
            return rank

        def S_range(front, rank, k):
            return [i for i in range(len(rank)) if rank[i] == k]

        def crowding_distance(F, rank, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[indices, m]
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                span = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
                if span > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (vals[sorted_idx[i+1]] - vals[sorted_idx[i-1]]) / span
            return cd

        def tournament_select(F, rank, cd, k=2):
            idx1 = rng.integers(len(rank))
            idx2 = rng.integers(len(rank))
            if rank[idx1] < rank[idx2]:
                return idx1
            elif rank[idx2] < rank[idx1]:
                return idx2
            else:
                return idx1 if cd[idx1] > cd[idx2] else idx2

        def sbx_crossover(x1, x2, eta=20.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        u = rng.random()
                        if u <= 1.0 / (eta + 1):
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        child1[i] = c1
                        child2[i] = c2
            return child1, child2

        def polynomial_mutate(x, eta=20.0):
            x_mut = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    x_mut[i] += delta * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            rank = non_dominated_sort(F)
            
            # Crowding distance for the worst front
            max_rank = max(rank)
            worst_front_idx = [i for i in range(len(rank)) if rank[i] == max_rank]
            cd = np.zeros(len(rank))
            if len(worst_front_idx) > 2:
                cd_worst = crowding_distance(F, rank, worst_front_idx)
                for i, idx in enumerate(worst_front_idx):
                    cd[idx] = cd_worst[i]
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            for _ in range(pop_size):
                p1 = tournament_select(F, rank, cd)
                p2 = tournament_select(F, rank, cd)
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                if len(offspring_X) < pop_size:
                    offspring_X.append(c2)
                    offspring_F.append(problem(c2))
                    used += 1
                    if used >= self.budget:
                        break
            
            # Combine and select
            if offspring_X:
                X_comb = np.vstack([X, np.array(offspring_X)])
                F_comb = np.vstack([F, np.array(offspring_F)])
                
                rank_comb = non_dominated_sort(F_comb)
                max_rank_comb = max(rank_comb)
                worst_idx = [i for i in range(len(rank_comb)) if rank_comb[i] == max_rank_comb]
                cd_comb = np.zeros(len(rank_comb))
                if len(worst_idx) > 2:
                    cd_w = crowding_distance(F_comb, rank_comb, worst_idx)
                    for i, idx in enumerate(worst_idx):
                        cd_comb[idx] = cd_w[i]
                
                # Select best pop_size
                selected = []
                for r in range(max_rank_comb + 1):
                    front_members = [i for i in range(len(rank_comb)) if rank_comb[i] == r]
                    if len(selected) + len(front_members) <= pop_size:
                        selected.extend(front_members)
                    else:
                        remaining = pop_size - len(selected)
                        front_cd = [(cd_comb[i], i) for i in front_members]
                        front_cd.sort(reverse=True)
                        selected.extend([i for _, i in front_cd[:remaining]])
                        break
                
                X = X_comb[selected]
                F = F_comb[selected]
            
            if used >= self.budget:
                break

        return X

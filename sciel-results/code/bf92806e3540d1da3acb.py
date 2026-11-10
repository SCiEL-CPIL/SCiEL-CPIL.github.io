import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Setup
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        N = 20  # Population size
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        # 1. Initial Population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        evals_used = N
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            S = [set() for _ in range(n)]
            n_dominated = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dominated[i] += 1
            
            current_front = [i for i in range(n) if n_dominated[i] == 0]
            front_id = 0
            while current_front:
                for i in current_front:
                    ranks[i] = front_id
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                current_front = next_front
                front_id += 1
            return ranks

        # Helper: Crowding Distance
        def crowding_distance(F, ranks, n_points):
            cd = np.zeros(n_points)
            unique_ranks = np.unique(ranks)
            for r in unique_ranks:
                idx = np.where(ranks == r)[0]
                if len(idx) <= 2:
                    cd[idx] = np.inf
                    continue
                for m in range(self.n_obj):
                    vals = F[idx, m]
                    min_val, max_val = np.min(vals), np.max(vals)
                    if max_val - min_val < 1e-12:
                        continue
                    sorted_idx = idx[np.argsort(vals)]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    for k in range(1, len(sorted_idx) - 1):
                        i_curr = sorted_idx[k]
                        i_prev = sorted_idx[k-1]
                        i_next = sorted_idx[k+1]
                        cd[i_curr] += (F[i_next, m] - F[i_prev, m]) / (max_val - min_val)
            return cd

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta_c=20):
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(self.dim):
                if np.random.rand() <= 0.5:
                    x1_i, x2_i = x1[i], x2[i]
                else:
                    x1_i, x2_i = x2[i], x1[i]
                
                if abs(x1_i - x2_i) < 1e-14:
                    child1[i] = child2[i] = x1_i
                    continue
                
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta_c + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                
                c1 = 0.5 * ((1 + beta) * x1_i + (1 - beta) * x2_i)
                c2 = 0.5 * ((1 - beta) * x1_i + (1 + beta) * x2_i)
                
                c1 = np.clip(c1, lo[i], hi[i])
                c2 = np.clip(c2, lo[i], hi[i])
                
                child1[i] = c1
                child2[i] = c2
            return child1, child2

        # Helper: Mutation
        def mutate(x, eta_m=20):
            y = x.copy()
            for i in range(self.dim):
                if np.random.rand() <= 1.0 / self.dim:
                    y_i = y[i]
                    lo_i, hi_i = lo[i], hi[i]
                    if hi_i - lo_i < 1e-14:
                        continue
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    
                    y_i = y_i + delta * (hi_i - lo_i)
                    y[i] = np.clip(y_i, lo_i, hi_i)
            return y

        # Helper: Tournament Selection
        def tournament_select(X, F, ranks, cd, k=2):
            idxs = np.random.randint(0, len(X), k)
            best_idx = idxs[0]
            for idx in idxs[1:]:
                if ranks[idx] < ranks[best_idx]:
                    best_idx = idx
                elif ranks[idx] == ranks[best_idx] and cd[idx] > cd[best_idx]:
                    best_idx = idx
            return best_idx

        # Main Loop
        while evals_used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Compute ranks and crowding distance for current population
            ranks = non_dominated_sort(F)
            cd = crowding_distance(F, ranks, N)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # We want to generate N offspring.
            # Each crossover produces 2 children.
            # So we need N/2 crossovers.
            # If N is odd, we handle the last one differently or just make N even.
            # Let's assume N is even for simplicity in this loop structure, 
            # or we can generate pairs and stop when we have N.
            
            num_pairs = N // 2
            for _ in range(num_pairs):
                # Select parents
                p1_idx = tournament_select(X, F, ranks, cd)
                p2_idx = tournament_select(X, F, ranks, cd)
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                
                evals_used += 2
                
                if evals_used >= self.budget:
                    break
            
            if evals_used >= self.budget:
                break

            # Combine parents and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.array([F, np.array(offspring_F)])
            
            # Sort and select next generation
            ranks_c = non_dominated_sort(F_combined)
            cd_c = crowding_distance(F_combined, ranks_c, len(X_combined))
            
            sort_idx = np.lexsort((-cd_c, ranks_c))
            X = X_combined[sort_idx[:N]]
            F = F_combined[sort_idx[:N]]
            
        return X

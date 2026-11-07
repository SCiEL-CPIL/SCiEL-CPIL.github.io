import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280 # Safety margin

        # Population size
        # For 3 objectives, N=20-30 is often good.
        # Budget 200.
        # If N=20, ~9 generations.
        # If N=30, ~6 generations.
        # Let's try N=20 for more exploration generations.
        pop_size = 20
        
        if budget < pop_size:
            # Fallback to random
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main Loop
        while used + pop_size <= budget and time.time() < deadline:
            # 1. Non-dominated sort and Crowding Distance
            ranks, fronts = self.non_dominated_sort(F)
            cd = self.crowding_distance(F, fronts)
            
            # 2. Selection
            # Tournament selection for parents
            parents_idx = []
            for _ in range(pop_size // 2):
                p1 = self.tournament_select(ranks, cd)
                p2 = self.tournament_select(ranks, cd)
                parents_idx.append((p1, p2))
            
            # 3. Crossover and Mutation
            offspring_X = np.empty((pop_size, dim))
            for i in range(pop_size // 2):
                idx1, idx2 = parents_idx[i]
                x1, x2 = X[idx1], X[idx2]
                c1, c2 = self.sbx_crossover(x1, x2, lo, hi, 0.9, 20)
                c1 = self.polynomial_mutation(c1, lo, hi, 1.0/dim, 20)
                c2 = self.polynomial_mutation(c2, lo, hi, 1.0/dim, 20)
                offspring_X[i*2] = c1
                offspring_X[i*2+1] = c2
            
            # 4. Evaluate Offspring
            F_off = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # 5. Combine
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])
            
            # 6. Selection for next generation
            ranks_all, fronts_all = self.non_dominated_sort(F_all)
            cd_all = self.crowding_distance(F_all, fronts_all)
            
            # Sort by rank, then crowding distance
            # We want to keep the best pop_size individuals
            # Create a list of (rank, -cd, index) and sort
            sort_keys = list(zip(ranks_all, -cd_all, range(len(ranks_all))))
            sort_keys.sort(key=lambda x: (x[0], x[1]))
            
            next_indices = [idx for _, _, idx in sort_keys[:pop_size]]
            
            X = X_all[next_indices]
            F = F_all[next_indices]

        # Return the best non-dominated points found so far?
        # The problem says "Every returned objective vector is recorded".
        # The return value of __call__ is ignored.
        # So we just return X or F, it doesn't matter.
        return X

    def non_dominated_sort(self, F):
        """
        Fast Non-Dominated Sorting
        Returns ranks (array) and fronts (list of lists of indices)
        """
        n = F.shape[0]
        ranks = np.zeros(n, dtype=int)
        fronts = []
        
        # Dominance count and set of points dominated by i
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif self.dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        # First front
        front = [i for i in range(n) if n_dom[i] == 0]
        ranks[front] = 0
        fronts.append(front)
        
        k = 1
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        ranks[j] = k
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
                k += 1
            else:
                break
                
        return ranks, fronts

    def dominates(self, f1, f2):
        """
        Returns True if f1 dominates f2 (minimization)
        """
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def crowding_distance(self, F, fronts):
        """
        Calculate crowding distance
        """
        n = F.shape[0]
        n_obj = F.shape[1]
        cd = np.zeros(n)
        
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    cd[i] = np.inf
                continue
                
            F_front = F[front]
            
            for k in range(n_obj):
                # Sort by k-th objective
                sorted_indices = np.argsort(F_front[:, k])
                
                # Get the actual indices in the original array
                orig_indices = front[sorted_indices]
                
                # Calculate distances
                # First and last are inf
                cd[orig_indices[0]] = np.inf
                cd[orig_indices[-1]] = np.inf
                
                if len(front) > 2:
                    range_k = F_front[sorted_indices[-1], k] - F_front[sorted_indices[0], k]
                    if range_k == 0:
                        continue
                        
                    for i in range(1, len(front) - 1):
                        dist = (F_front[sorted_indices[i+1], k] - F_front[sorted_indices[i-1], k]) / range_k
                        cd[orig_indices[i]] += dist
                        
        return cd

    def tournament_select(self, ranks, cd, k=2):
        """
        Tournament selection
        """
        n = len(ranks)
        candidates = np.random.choice(n, k, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and cd[c] > cd[best]:
                best = c
        return best

    def sbx_crossover(self, x1, x2, lo, hi, p_c=0.9, eta_c=20):
        """
        Simulated Binary Crossover
        """
        if np.random.random() > p_c:
            return x1.copy(), x2.copy()
            
        c1, c2 = x1.copy(), x2.copy()
        for i in range(len(x1)):
            if np.random.random() < 0.5:
                if x1[i] != x2[i]:
                    x_min = min(x1[i], x2[i])
                    x_max = max(x1[i], x2[i])
                    if x_max - x_min <= 1e-14:
                        continue
                        
                    u = np.random.random()
                    beta = 1 + 2 * (x1[i] - x_min) / (x_max - x_min)
                    alpha = 2 - beta ** (-eta_c - 1)
                    if u <= 1 / alpha:
                        beta_q = (u * alpha) ** (1 / (eta_c + 1))
                    else:
                        beta_q = (1 / (2 - u * alpha)) ** (1 / (eta_c + 1))
                    
                    c1[i] = 0.5 * ((x1[i] + x2[i]) - beta_q * (x1[i] - x2[i]))
                    
                    u = np.random.random()
                    beta = 1 + 2 * (x2[i] - x_min) / (x_max - x_min)
                    alpha = 2 - beta ** (-eta_c - 1)
                    if u <= 1 / alpha:
                        beta_q = (u * alpha) ** (1 / (eta_c + 1))
                    else:
                        beta_q = (1 / (2 - u * alpha)) ** (1 / (eta_c + 1))
                    
                    c2[i] = 0.5 * ((x1[i] + x2[i]) + beta_q * (x1[i] - x2[i]))
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def polynomial_mutation(self, x, lo, hi, p_m, eta_m):
        """
        Polynomial Mutation
        """
        x_mut = x.copy()
        for i in range(len(x)):
            if np.random.random() < p_m:
                x_min = lo[i]
                x_max = hi[i]
                if x_max - x_min <= 1e-14:
                    continue
                    
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                    delta_q = (x[i] - x_min) * delta
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    delta_q = (x_max - x[i]) * delta
                
                x_mut[i] = x[i] + delta_q
                x_mut[i] = np.clip(x_mut[i], x_min, x_max)
        return x_mut

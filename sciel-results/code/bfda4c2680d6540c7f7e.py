import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size
        # We want to maximize generations.
        # N = 20 is a good balance.
        N = 20
        if self.budget < N:
            N = self.budget
            
        # Initialize with Latin Hypercube Sampling
        # Simple LHS: divide each dimension into N bins, pick one random point per bin per dim
        X = np.zeros((N, self.dim))
        for d in range(self.dim):
            # Create bins
            edges = np.linspace(lo[d], hi[d], N + 1)
            # For each bin, pick a random point
            for i in range(N):
                X[i, d] = np.random.uniform(edges[i], edges[i+1])
        
        F = np.array([problem(x) for x in X])
        used = N
        
        # NSGA-II Loop
        while used + N <= self.budget:
            # 1. Non-dominated sort
            ranks, fronts = self._fast_non_dominated_sort(F)
            
            # 2. Crowding distance
            crowd = np.zeros(N)
            for i in range(len(fronts)):
                if len(fronts[i]) > 1:
                    front_idx = fronts[i]
                    for k in range(self.n_obj):
                        sorted_idx = np.argsort(F[front_idx, k])
                        crowd[front_idx[sorted_idx[0]]] = np.inf
                        crowd[front_idx[sorted_idx[-1]]] = np.inf
                        min_val = F[front_idx, k].min()
                        max_val = F[front_idx, k].max()
                        range_val = max_val - min_val
                        if range_val == 0: continue
                        for j in range(1, len(front_idx)-1):
                            idx = front_idx[sorted_idx[j]]
                            idx_prev = front_idx[sorted_idx[j-1]]
                            idx_next = front_idx[sorted_idx[j+1]]
                            crowd[idx] += (F[idx_next, k] - F[idx_prev, k]) / range_val
            
            # 3. Generate Offspring
            offspring_X = []
            # Generate N/2 pairs
            for _ in range(N // 2):
                p1 = self._tournament_select(ranks, crowd)
                p2 = self._tournament_select(ranks, crowd)
                
                # SBX Crossover
                if np.random.rand() < 0.9:
                    c1, c2 = self._sbx_crossover(X[p1], X[p2], lo, hi)
                else:
                    c1, c2 = X[p1].copy(), X[p2].copy()
                
                # Polynomial Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                
            # If N is odd, add one more mutated parent? 
            # For N=20, it's even. If N is odd, we can just mutate a random parent.
            if N % 2 != 0:
                p1 = self._tournament_select(ranks, crowd)
                c1 = self._polynomial_mutation(X[p1].copy(), lo, hi)
                offspring_X.append(c1)

            # 4. Evaluate Offspring
            F_off = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)
            
            # 5. Combine and Select
            X_all = np.vstack([X, np.array(offspring_X)])
            F_all = np.vstack([F, F_off])
            
            ranks_all, fronts_all = self._fast_non_dominated_sort(F_all)
            
            new_X = []
            new_F = []
            current_size = 0
            for i in range(len(fronts_all)):
                front = fronts_all[i]
                if current_size + len(front) <= N:
                    for idx in front:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                    current_size += len(front)
                else:
                    # Calculate crowding for this front
                    crowd_front = np.zeros(len(front))
                    for k in range(self.n_obj):
                        sorted_idx = np.argsort(F_all[front, k])
                        crowd_front[sorted_idx[0]] = np.inf
                        crowd_front[sorted_idx[-1]] = np.inf
                        min_val = F_all[front, k].min()
                        max_val = F_all[front, k].max()
                        range_val = max_val - min_val
                        if range_val == 0: continue
                        for j in range(1, len(front)-1):
                            idx = front[sorted_idx[j]]
                            idx_prev = front[sorted_idx[j-1]]
                            idx_next = front[sorted_idx[j+1]]
                            crowd_front[sorted_idx[j]] += (F_all[idx_next, k] - F_all[idx_prev, k]) / range_val
                    
                    sorted_front_idx = np.argsort(crowd_front)[::-1]
                    needed = N - current_size
                    for j in range(needed):
                        idx = front[sorted_front_idx[j]]
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                    current_size += needed
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        S = [[] for _ in range(n)] # S[i] = indices dominated by i
        n_d = np.zeros(n, dtype=int) # n_d[i] = number of individuals dominating i
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_d[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_d[i] += 1
        
        current_front = []
        for i in range(n):
            if n_d[i] == 0:
                ranks[i] = 0
                current_front.append(i)
        
        k = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            current_front = next_front
            k += 1
            
        return ranks, fronts

    def _tournament_select(self, ranks, crowd):
        # Tournament size 2
        i = np.random.randint(len(ranks))
        j = np.random.randint(len(ranks))
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            if crowd[i] > crowd[j]:
                return i
            else:
                return j

    def _sbx_crossover(self, x1, x2, lo, hi):
        # Simulated Binary Crossover
        # eta = 20
        eta = 20
        c1 = x1.copy()
        c2 = x2.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(x2[i] - x1[i]) > 1e-14:
                    if x1[i] < x2[i]:
                        x1_i, x2_i = x1[i], x2[i]
                    else:
                        x1_i, x2_i = x2[i], x1[i]
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    
                    c1_i = 0.5 * ((1 + beta) * x1_i + (1 - beta) * x2_i)
                    c2_i = 0.5 * ((1 - beta) * x1_i + (1 + beta) * x2_i)
                    
                    c1[i] = np.clip(c1_i, lo[i], hi[i])
                    c2[i] = np.clip(c2_i, lo[i], hi[i])
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        # Polynomial Mutation
        eta = 20
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5: # Mutation probability 0.5? Or 1/dim?
                # Standard is 1/dim. Let's use 1/dim.
                pass
        
        # Correct mutation prob is 1/dim
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                u = np.random.rand()
                if u < 0.5:
                    delta1 = (2 * u + (x[i] - lo[i]) / (hi[i] - lo[i])) ** (1 / (eta + 1)) - 1
                else:
                    delta2 = (2 * (1 - u) + (hi[i] - x[i]) / (hi[i] - lo[i])) ** (1 / (eta + 1)) - 1
                    delta1 = -delta2
                x_mut[i] = x[i] + delta1 * (hi[i] - lo[i])
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut

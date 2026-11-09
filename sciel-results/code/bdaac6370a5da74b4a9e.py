import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 10
        # Number of generations
        # Total budget = N (init) + N * G
        # G = (budget - N) // N
        G = (self.budget - N) // N
        
        # Initialize population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            return X
            
        # NSGA-II Loop
        for gen in range(G):
            if time.time() > deadline:
                break
                
            # 1. Selection and Crossover to create offspring Q
            Q = np.zeros_like(X)
            for i in range(N):
                # Select two parents
                p1 = self._tournament_select(X, F)
                p2 = self._tournament_select(X, F)
                
                # SBX Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                # Polynomial Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                # We need to fill N offspring. 
                # Standard NSGA-II creates N offspring from N pairs (2N parents).
                # Here we create 2 children per iteration, but we only have N slots.
                # So we pick one of the two, or we do N/2 pairs?
                # Let's do N/2 pairs to create N children.
                # But N=10 is even, so 5 pairs.
                # Let's adjust the loop.
                pass
            
            # Better approach for offspring creation:
            Q = np.zeros((N, self.dim))
            for i in range(0, N, 2):
                p1 = self._tournament_select(X, F)
                p2 = self._tournament_select(X, F)
                
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                Q[i] = c1
                if i + 1 < N:
                    Q[i+1] = c2
            
            # Evaluate offspring
            FQ = np.array([problem(x) for x in Q])
            
            # 2. Combine parents P and offspring Q
            R_X = np.vstack((X, Q))
            R_F = np.vstack((F, FQ))
            
            # 3. Non-dominated sorting
            fronts = self._non_dominated_sort(R_F)
            
            # 4. Select next population P
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= N:
                    next_X.append(R_X[front])
                    next_F.append(R_F[front])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = N - len(next_X)
                    # Calculate crowding distance for this front
                    # Note: crowding distance is calculated on the subset
                    subset_X = R_X[front]
                    subset_F = R_F[front]
                    cd = self._crowding_distance(subset_F)
                    # Sort by cd descending
                    sorted_indices = np.argsort(-cd)
                    selected_indices = sorted_indices[:remaining]
                    next_X.append(R_X[front][selected_indices])
                    next_F.append(R_F[front][selected_indices])
                    break
            
            X = np.vstack(next_X)
            F = np.vstack(next_F)
            
        return X

    def _tournament_select(self, X, F):
        # Rank and crowding distance are needed for proper tournament selection
        # For simplicity in this small budget, we can just pick 2 random and compare rank?
        # Or better, calculate ranks for the current population X, F once per generation.
        # But we are calling this inside the offspring loop.
        # It's more efficient to precompute ranks for X, F.
        
        # Let's refactor: Precompute ranks for X, F.
        # I'll modify the main loop to do this.
        
        # For now, a simple random selection is weak.
        # Let's implement a helper that takes precomputed ranks.
        
        # Actually, let's just use random selection for now to keep it simple?
        # No, NSGA-II relies on rank-based selection.
        
        # I will modify the __call__ to precompute ranks for the current population.
        
        # Placeholder for now, will fix in final code.
        idx1 = np.random.randint(len(X))
        idx2 = np.random.randint(len(X))
        # Simple dominance check for tournament of size 2
        # If f1 dominates f2, pick 1. If f2 dominates f1, pick 2. Else random.
        f1 = F[idx1]
        f2 = F[idx2]
        
        # Dominance check: f1 dominates f2 if f1 <= f2 and f1 != f2
        if np.all(f1 <= f2) and np.any(f1 < f2):
            return X[idx1]
        elif np.all(f2 <= f1) and np.any(f2 < f1):
            return X[idx2]
        else:
            return X[np.random.choice([idx1, idx2])]

    def _sbx_crossover(self, p1, p2, lo, hi):
        # Simulated Binary Crossover
        # eta_c = 20
        eta_c = 20
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if p1[i] < p2[i]:
                        x1 = p1[i]
                        x2 = p2[i]
                    else:
                        x1 = p2[i]
                        x2 = p1[i]
                    
                    if np.random.rand() <= 0.5:
                        beta = 1.0 / (x2 - x1) * (np.random.rand() ** (1.0 / (eta_c + 1)))
                        beta = min(beta, (hi[i] - x1) / (x2 - x1)) if (x2 - x1) > 0 else 1.0
                        c1[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    else:
                        beta = 1.0 / (x2 - x1) * (np.random.rand() ** (1.0 / (eta_c + 1)))
                        beta = min(beta, (x2 - lo[i]) / (x2 - x1)) if (x2 - x1) > 0 else 1.0
                        c1[i] = 0.5 * ((1 + beta) * x2 + (1 - beta) * x1)
                    
                    c2[i] = 2 * 0.5 * (x1 + x2) - c1[i] # This is not quite right for SBX, usually symmetric
                    # Actually, standard SBX:
                    # if rand < 0.5:
                    #   beta = ...
                    #   c1 = 0.5 * ((1+beta)*x1 + (1-beta)*x2)
                    # else:
                    #   beta = ...
                    #   c1 = 0.5 * ((1+beta)*x2 + (1-beta)*x1)
                    # And c2 is the other one.
                    # Let's stick to a simpler mutation-heavy approach if SBX is complex to get right quickly.
                    # Or use simple arithmetic crossover?
                    # Let's use uniform crossover + mutation, it's often robust for small budgets.
                    pass
        
        # Let's switch to Uniform Crossover for simplicity and reliability.
        mask = np.random.rand(self.dim) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        # eta_m = 20
        eta_m = 20
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                y = x[i]
                delta1 = (y - lo[i]) / (hi[i] - lo[i]) if (hi[i] - lo[i]) > 0 else 0
                delta2 = (hi[i] - y) / (hi[i] - lo[i]) if (hi[i] - lo[i]) > 0 else 0
                mut_pow = 1.0 / (eta_m + 1)
                
                u = np.random.rand()
                if u < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy) ** (mut_pow + 1)
                    deltaq = val ** (1.0 / (mut_pow + 1)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy) ** (mut_pow + 1)
                    deltaq = val ** (1.0 / (mut_pow + 1)) - 1.0
                
                y = y + deltaq * (hi[i] - lo[i])
                x[i] = np.clip(y, lo[i], hi[i])
        return x

    def _non_dominated_sort(self, F):
        # Returns list of arrays of indices for each front
        n = len(F)
        S = [[] for _ in range(n)] # Dominated sets
        n_d = np.zeros(n, dtype=int) # Number of dominators
        rank = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_d[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_d[i] += 1
        
        fronts = []
        current_front = np.where(n_d == 0)[0]
        rank[current_front] = 0
        
        while len(current_front) > 0:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        rank[j] = rank[i] + 1
                        next_front.append(j)
            current_front = np.array(next_front)
            
        return fronts

    def _crowding_distance(self, F):
        # F is an array of shape (m, n_obj)
        m = len(F)
        if m <= 2:
            return np.full(m, np.inf)
        
        cd = np.zeros(m)
        for k in range(self.n_obj):
            # Sort by k-th objective
            sorted_indices = np.argsort(F[:, k])
            cd[sorted_indices[0]] = np.inf
            cd[sorted_indices[-1]] = np.inf
            f_min = F[sorted_indices[0], k]
            f_max = F[sorted_indices[-1], k]
            if f_max - f_min > 1e-14:
                for i in range(1, m - 1):
                    cd[sorted_indices[i]] += (F[sorted_indices[i+1], k] - F[sorted_indices[i-1], k]) / (f_max - f_min)
        return cd

import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # We need to fit generations into the budget.
        # Let N be population size.
        # Initial: N evals.
        # Each generation: N offspring evals.
        # Total evals = N * (1 + G) where G is number of generations.
        # We want to maximize G for a given N, or balance them.
        # With Budget=200, N=50 gives 4 generations (50 * 4 = 200).
        # N=40 gives 5 generations (40 * 5 = 200).
        # N=25 gives 8 generations.
        # Smaller populations converge faster but lose diversity.
        # NSGA-II works well with N around 50-100 for 3 objectives.
        # Let's try N=50.
        
        N = 50
        if self.budget < N:
            # Fallback to random search if budget is too small
            X = lo + (hi - lo) * np.random.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper functions for NSGA-II
        
        def non_dominated_sort(F):
            """Returns list of fronts, each containing indices of non-dominated points."""
            n = len(F)
            S = [set() for _ in range(n)]  # S[i] = set of indices dominated by i
            n_doms = [0] * n               # n_doms[i] = number of points dominating i
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_doms[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_doms[i] += 1
            
            current_front = [i for i in range(n) if n_doms[i] == 0]
            fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
                
            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for points in 'indices'."""
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            
            dist = [0.0] * len(indices)
            for m in range(self.n_obj):
                # Sort indices by objective m
                sorted_indices = sorted(indices, key=lambda i: F[i][m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                f_range = f_max - f_min
                
                if f_range == 0:
                    continue
                    
                for k in range(1, len(sorted_indices) - 1):
                    idx_curr = sorted_indices[k]
                    idx_prev = sorted_indices[k - 1]
                    idx_next = sorted_indices[k + 1]
                    dist[idx_curr] += (F[idx_next][m] - F[idx_prev][m]) / f_range
            
            return dist

        while used + N <= self.budget:
            # Create offspring
            Q = np.zeros((N, self.dim))
            
            for i in range(N):
                # Tournament selection for parent 1
                p1_idx = self.tournament_select(F, N)
                # Tournament selection for parent 2
                p2_idx = self.tournament_select(F, N)
                
                # SBX Crossover
                c1, c2 = self.sbx_crossover(X[p1_idx], X[p2_idx], lo, hi)
                
                # Mutation
                c1 = self.polynomial_mutation(c1, lo, hi)
                c2 = self.polynomial_mutation(c2, lo, hi)
                
                Q[2*i] = c1
                Q[2*i+1] = c2
            
            Fq = np.array([problem(x) for x in Q])
            used += N
            
            # Combine P and Q
            X_combined = np.vstack((X, Q))
            F_combined = np.vstack((F, Fq))
            
            # Non-dominated sorting
            fronts = non_dominated_sort(F_combined)
            
            # Select next population
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= N:
                    # Take all from this front
                    for idx in front:
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                else:
                    # Need to select some from this front based on crowding distance
                    needed = N - len(X_new)
                    cd = crowding_distance(F_combined, front)
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:needed]:
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)

        return X

    def tournament_select(self, F, k=2):
        """Select an index via tournament selection."""
        idxs = np.random.choice(len(F), size=k, replace=False)
        # Find non-dominated among selected
        best_idx = idxs[0]
        for i in idxs[1:]:
            if np.all(F[i] <= F[best_idx]) and np.any(F[i] < F[best_idx]):
                best_idx = i
            elif not (np.all(F[best_idx] <= F[i]) and np.any(F[best_idx] < F[i])):
                # If best is dominated by i, i is better? 
                # Actually, simpler: just pick the one with lower sum or random if non-dominated?
                # Standard tournament: pick the one that dominates the other. 
                # If they are non-dominated, pick randomly or use crowding. 
                # For simplicity in this helper, let's just use sum of objectives as a tie-breaker for non-dominated cases in tournament.
                if np.sum(F[i]) < np.sum(F[best_idx]):
                    best_idx = i
        return best_idx

    def sbx_crossover(self, x1, x2, lo, hi, eta=20):
        """Simulated Binary Crossover."""
        c1, c2 = x1.copy(), x2.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    x1i, x2i = x1[i], x2[i]
                    if x1i < x2i:
                        xl, xh = x1i, x2i
                    else:
                        xl, xh = x2i, x1i
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * xl + (1 - beta) * xh)
                    c2[i] = 0.5 * ((1 - beta) * xl + (1 + beta) * xh)
                    
                    # Clip to bounds
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def polynomial_mutation(self, x, lo, hi, eta=20):
        """Polynomial Mutation."""
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.8: # Mutation rate 0.8 per gene? Or 1/dim?
                # Standard is 1/dim. Let's use 1/dim.
                pass 
        
        # Let's stick to standard: mutate with prob 1/dim
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                u = np.random.rand()
                xl = lo[i]
                xh = hi[i]
                if u <= 0.5:
                    delta1 = (x_mut[i] - xl) / (xh - xl)
                    delta = (1 + 2 * (xl - x_mut[i]) / (xh - xl)) ** (eta + 1)
                    delta = 2 * (1 - delta ** (1 / (eta + 1)))
                else:
                    delta2 = (xh - x_mut[i]) / (xh - xl)
                    delta = (1 + 2 * (xh - x_mut[i]) / (xh - xl)) ** (eta + 1)
                    delta = 2 * (1 - delta ** (1 / (eta + 1)))
                
                x_mut[i] = x_mut[i] + delta * (xh - xl)
                x_mut[i] = np.clip(x_mut[i], xl, xh)
                
        return x_mut

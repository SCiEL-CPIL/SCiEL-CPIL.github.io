import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        N = 10  # Population size
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Parameters
        eta_c = 15.0
        eta_m = 20.0
        
        while evals < self.budget:
            # Selection: Binary Tournament
            parents = []
            # Pre-compute crowding distance for current population for tie-breaking
            cd = self._crowding_distance(F)
            for _ in range(N):
                i1 = np.random.randint(N)
                i2 = np.random.randint(N)
                if self._dominates(F[i1], F[i2]):
                    parents.append(i1)
                elif self._dominates(F[i2], F[i1]):
                    parents.append(i2)
                else:
                    if cd[i1] > cd[i2]:
                        parents.append(i1)
                    else:
                        parents.append(i2)
            
            # Crossover and Mutation
            X_off = []
            for i in range(0, N, 2):
                p1 = X[parents[i]]
                p2 = X[parents[i+1]]
                # SBX Crossover
                u = np.random.random(self.dim)
                beta = np.where(u < 0.5, (2*u)**(1/(eta_c+1)), (2*(1-u))**(1/(eta_c+1)))
                c1 = 0.5 * ((1+beta)*p1 + (1-beta)*p2)
                c2 = 0.5 * ((1-beta)*p1 + (1+beta)*p2)
                X_off.append(c1)
                X_off.append(c2)
            
            # Mutation
            for i in range(len(X_off)):
                x = X_off[i]
                u = np.random.random(self.dim)
                delta = np.zeros(self.dim)
                mask = u < 0.5
                delta[mask] = (2*u[mask])**(1/(eta_m+1))
                delta[~mask] = (2*(1-u[~mask]))**(1/(eta_m+1))
                sign = np.where(u < 0.5, 1, -1)
                x = x + (hi - lo) * delta * sign
                X_off[i] = np.clip(x, lo, hi)
            
            X_off = np.array(X_off)
            F_off = np.array([problem(x) for x in X_off])
            evals += len(X_off)
            
            # Combine and Select
            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            
            # Non-dominated sort
            ranks = self._non_dominated_sort(F_comb)
            
            # Crowding distance
            cd_comb = self._crowding_distance(F_comb)
            
            # Sort by rank then crowding distance
            # We want lowest rank (0 is best), highest crowding distance
            sort_idx = np.lexsort((-cd_comb, ranks))
            # Take first N
            X = X_comb[sort_idx[:N]]
            F = F_comb[sort_idx[:N]]
            
        return X

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        remaining = list(range(n))
        front_idx = 0
        while remaining:
            current_front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i != j and self._dominates(F[j], F[i]):
                        dominated = True
                        break
                if not dominated:
                    current_front.append(i)
            
            for i in current_front:
                ranks[i] = front_idx
            remaining = [i for i in remaining if i not in current_front]
            front_idx += 1
        return ranks

    def _crowding_distance(self, F):
        n = len(F)
        cd = np.zeros(n)
        if n <= 2:
            return cd
        for m in range(self.n_obj):
            idx = np.argsort(F[:, m])
            cd[idx[0]] = np.inf
            cd[idx[-1]] = np.inf
            f_min = F[idx[0], m]
            f_max = F[idx[-1], m]
            if f_max == f_min:
                continue
            for i in range(1, n-1):
                cd[idx[i]] += (F[idx[i+1], m] - F[idx[i-1], m]) / (f_max - f_min)
        return cd

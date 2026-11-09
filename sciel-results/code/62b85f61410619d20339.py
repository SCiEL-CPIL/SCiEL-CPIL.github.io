"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Multi-Objective Differential Evolution Optimizer (NSGA-II selection)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 20
        
        if self.budget < N:
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        # DE Parameters
        F_factor = 0.5 # Differential weight
        CR = 1.0       # Crossover rate (1.0 means always take from mutant if better, or use binomial crossover)
        # Standard DE uses binomial crossover. Let's use binomial crossover with CR=0.9
        CR = 0.9

        while used < self.budget:
            offspring_X = np.zeros((N, self.dim))
            
            for i in range(N):
                # Select 3 distinct random parents
                r1, r2, r3 = rng.choice(N, size=3, replace=False)
                
                # Differential Evolution Mutation
                # v = x_r1 + F * (x_r2 - x_r3)
                v = X[r1] + F_factor * (X[r2] - X[r3])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Binomial Crossover
                # Create a random binary mask
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() <= CR or j == j_rand:
                        offspring_X[i][j] = v[j]
                    else:
                        offspring_X[i][j] = X[i][j]
                        
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += N
            
            # Combine parent and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Selection for next generation using NSGA-II logic
            X, F = self._nsga2_select(X_combined, F_combined, N, rng)

        return X

    def _nsga2_select(self, X, F, N, rng):
        ranks, crowding_dist = self._fast_non_dominated_sort(F)
        
        # Sort by rank, then by crowding distance (descending)
        # We want low rank (better) and high crowding (better).
        # np.lexsort sorts by the last key first.
        # So keys: (-crowding_dist, ranks)
        sorted_indices = np.lexsort((-crowding_dist, ranks))
        
        selected_indices = sorted_indices[:N]
        
        return X[selected_indices], F[selected_indices]

    def _fast_non_dominated_sort(self, F):
        N = len(F)
        n_obj = self.n_obj
        ranks = np.zeros(N, dtype=int)
        crowding_dist = np.zeros(N)
        
        S = [[] for _ in range(N)] # Dominated sets
        n = np.zeros(N, dtype=int) # Number of dominating solutions
        
        # Step 1: Dominance relations
        for i in range(N):
            for j in range(i + 1, N):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n[i] += 1
        
        # Step 2: Assign ranks
        front = []
        for i in range(N):
            if n[i] == 0:
                ranks[i] = 0
                front.append(i)
        
        k = 0
        while len(front) > 0:
            next_front = []
            for i in front:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            front = next_front
            k += 1
        
        # Step 3: Crowding distance
        for r in range(k + 1):
            front_indices = [i for i in range(N) if ranks[i] == r]
            if len(front_indices) <= 2:
                for i in front_indices:
                    crowding_dist[i] = np.inf
                continue
            
            for m in range(n_obj):
                # Sort front by objective m
                sorted_front = sorted(front_indices, key=lambda x: F[x][m])
                crowding_dist[sorted_front[0]] = np.inf
                crowding_dist[sorted_front[-1]] = np.inf
                
                f_min = F[sorted_front[0]][m]
                f_max = F[sorted_front[-1]][m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                    
                for i in range(1, len(sorted_front) - 1):
                    idx = sorted_front[i]
                    idx_prev = sorted_front[i - 1]
                    idx_next = sorted_front[i + 1]
                    crowding_dist[idx] += (F[idx_next][m] - F[idx_prev][m]) / f_range

        return ranks, crowding_dist

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        N = 20
        
        rng = np.random.default_rng(42)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Calculate initial ranks
        fronts, rank = self._fast_non_dominated_sort(F)
        
        # Helper: Fast Non-Dominated Sorting
        def _fast_non_dominated_sort(self, F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_d = [0] * n
            rank = [0] * n
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
                if n_d[i] == 0:
                    rank[i] = 0
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
                if not fronts[k]:
                    break
            return fronts, np.array(rank)

        # Helper: Crowding Distance
        def _crowding_distance(self, F, indices):
            if not indices:
                return {}
            m = len(indices)
            if m <= 2:
                return {i: float('inf') for i in indices}
            
            cd = {i: 0.0 for i in indices}
            for k in range(n_obj):
                sorted_indices = sorted(indices, key=lambda i: F[i, k])
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                f_min = F[sorted_indices[0], k]
                f_max = F[sorted_indices[-1], k]
                if f_max - f_min == 0:
                    continue
                for i in range(1, m - 1):
                    cd[sorted_indices[i]] += (F[sorted_indices[i + 1], k] - F[sorted_indices[i - 1], k]) / (f_max - f_min)
            return cd

        # Helper: SBX Crossover
        def _sbx_crossover(self, x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1, x2
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x1)
            for i in range(dim):
                if x1[i] == x2[i]:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                else:
                    if rng.random() < 0.5:
                        u = rng.random()
                        alpha = 1 + 2 * (min(x1[i], x2[i]) - lo[i]) / (x1[i] - x2[i])
                        beta = 1 + 2 * (hi[i] - max(x1[i], x2[i])) / (x1[i] - x2[i])
                        if u <= 1 / (alpha + 1):
                            factor = (2 * u) ** (1 / (eta + 1))
                        elif u >= 1 - 1 / (beta + 1):
                            factor = (2 * (1 - u)) ** (1 / (eta + 1))
                        else:
                            factor = 1
                        child1[i] = 0.5 * ((1 + factor) * x1[i] + (1 - factor) * x2[i])
                        child2[i] = 0.5 * ((1 - factor) * x1[i] + (1 + factor) * x2[i])
                    else:
                        u = rng.random()
                        alpha = 1 + 2 * (min(x1[i], x2[i]) - lo[i]) / (x1[i] - x2[i])
                        beta = 1 + 2 * (hi[i] - max(x1[i], x2[i])) / (x1[i] - x2[i])
                        if u <= 1 / (alpha + 1):
                            factor = (2 * u) ** (1 / (eta + 1))
                        elif u >= 1 - 1 / (beta + 1):
                            factor = (2 * (1 - u)) ** (1 / (eta + 1))
                        else:
                            factor = 1
                        child1[i] = 0.5 * ((1 - factor) * x1[i] + (1 + factor) * x2[i])
                        child2[i] = 0.5 * ((1 + factor) * x1[i] + (1 - factor) * x2[i])
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        # Helper: Polynomial Mutation
        def _polynomial_mutation(self, x, eta=20.0):
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x_mut[i] = x[i] + delta_q * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        # Helper: Tournament Selection
        def _tournament_selection(self, rank, cd=None, k=2):
            idxs = rng.integers(0, len(rank), k)
            best = idxs[0]
            for i in range(1, k):
                if rank[idxs[i]] < rank[best]:
                    best = idxs[i]
                elif rank[idxs[i]] == rank[best]:
                    if cd is not None:
                        if cd[idxs[i]] > cd[best]:
                            best = idxs[i]
                    else:
                        # Random tie-break
                        if rng.random() < 0.5:
                            best = idxs[i]
            return best

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Calculate crowding distance for current population for selection
            # We need crowding distance for all fronts or just the last one?
            # Standard NSGA-II calculates it for the entire combined population, 
            # but for selection of parents, we can use the crowding distance of the current population.
            
            # For simplicity, let's just use rank for tournament selection.
            # Diversity is maintained by the selection of the next generation.
            
            # Create offspring
            offspring_X = np.zeros((N, dim))
            for i in range(N // 2):
                p1_idx = _tournament_selection(rank)
                p2_idx = _tournament_selection(rank)
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                c1, c2 = _sbx_crossover(x1, x2)
                c1 = _polynomial_mutation(c1)
                c2 = _polynomial_mutation(c2)
                
                offspring_X[2 * i] = c1
                offspring_X[2 * i + 1] = c2
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += N
            
            # Combine parents and offspring
            X_comb = np.vstack((X, offspring_X))
            F_comb = np.vstack((F, offspring_F))
            
            # Non-dominated sorting
            fronts, rank_comb = _fast_non_dominated_sort(F_comb)
            
            # Select next generation
            next_X = []
            next_F = []
            next_rank = []
            
            for front in fronts:
                if len(next_X) + len(front) <= N:
                    # Add all
                    for idx in front:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                        next_rank.append(rank_comb[idx])
                else:
                    # Need to select based on crowding distance
                    cd = _crowding_distance(F_comb, front)
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = N - len(next_X)
                    for idx in sorted_front[:needed]:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                        next_rank.append(rank_comb[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            rank = np.array(next_rank)

        return X

# EVOLVE-BLOCK-END

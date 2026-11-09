"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9 with vectorized sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280  # 300s limit, keep 20s safety margin
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: ~20% of budget, min 20, max 50 to ensure enough generations
        pop_size = max(20, min(50, self.budget // 4))
        
        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Vectorized fast non-dominated sorting. Returns list of fronts (lists of indices)."""
            n = len(F)
            if n == 0:
                return []
            
            # S[i] contains indices of solutions dominated by i
            # n_count[i] contains number of solutions that dominate i
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            
            # Compute domination relationships
            # F[i] <= F[j] and F[i] != F[j] means i dominates j
            # We use broadcasting to compute all pairwise comparisons
            # This is O(N^2) memory, but N is small (<=50)
            
            # M[i, j] is True if i dominates j
            M = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        M[i, j] = True
            
            for i in range(n):
                for j in range(n):
                    if M[i, j]:
                        S[i].append(j)
                        n_count[j] += 1
            
            fronts = []
            front = []
            for i in range(n):
                if n_count[i] == 0:
                    front.append(i)
            
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                front = next_front
            
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            F_sub = F[indices]

            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / f_range

            return dist

        def select_parents(X, F, pop_size):
            """Binary tournament selection based on non-dominated rank and crowding distance."""
            fronts = fast_non_dominated_sort(F)

            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for front_idx, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = front_idx

            # Compute crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                cd_sub = crowding_distance(F, front)
                for k, idx in enumerate(front):
                    cd[idx] = cd_sub[k]

            # Tournament selection
            parents = []
            for _ in range(pop_size):
                i = rng.integers(len(F))
                j = rng.integers(len(F))
                if rank[i] < rank[j]:
                    parents.append(i)
                elif rank[j] < rank[i]:
                    parents.append(j)
                else:
                    if cd[i] > cd[j]:
                        parents.append(i)
                    else:
                        parents.append(j)
            return parents

        def sbx_crossover(p1, p2):
            """Simulated Binary Crossover."""
            eta = 15.0
            c1 = np.zeros(self.dim)
            c2 = np.zeros(self.dim)
            for k in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    c1[k] = 0.5 * ((1 + beta) * p1[k] + (1 - beta) * p2[k])
                    c2[k] = 0.5 * ((1 - beta) * p1[k] + (1 + beta) * p2[k])
                else:
                    c1[k] = p2[k]
                    c2[k] = p1[k]
            return c1, c2

        def polynomial_mutation(x):
            """Polynomial Mutation."""
            eta = 20.0
            x_mut = x.copy()
            for k in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x_mut[k] += delta * span[k]
            return np.clip(x_mut, lo, hi)

        # Main optimization loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Select parents
            parents = select_parents(X, F, pop_size)

            # Generate offspring
            offspring_X = np.zeros((pop_size, self.dim))
            for i in range(0, pop_size, 2):
                p1_idx = parents[i]
                p2_idx = parents[i + 1]
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                offspring_X[i] = c1
                if i + 1 < pop_size:
                    offspring_X[i + 1] = c2

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size

            # Combine parent and offspring populations
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])

            # Selection: keep best pop_size individuals
            fronts = fast_non_dominated_sort(F_combined)

            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_combined, front)
                    sorted_front = sorted(range(len(front)), key=lambda k: cd[k], reverse=True)
                    for k in range(remaining):
                        idx = front[sorted_front[k]]
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s limit

        # NSGA-II parameters
        # Budget 200, Pop size 20 -> 10 generations (including init)
        # This allows for sufficient exploration and exploitation
        pop_size = 20

        if pop_size > budget:
            pop_size = budget

        if pop_size < 2:
            # Fallback for very small budgets
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Main loop
        while used + pop_size <= budget:
            if time.time() > deadline:
                break

            # 1. Non-dominated sort on current population
            ranks, fronts = self.non_dominated_sort(F)

            # 2. Crowding distance
            cd = self.crowding_distance(F, fronts)

            # 3. Create offspring
            offspring_X = np.empty((pop_size, dim))
            n_pairs = pop_size // 2

            for i in range(n_pairs):
                # Tournament selection for two parents
                p1_idx = self.tournament_select(ranks, cd, 2)
                p2_idx = self.tournament_select(ranks, cd, 2)
                
                x1, x2 = X[p1_idx], X[p2_idx]
                c1, c2 = self.sbx_crossover(x1, x2, lo, hi, 0.9, 20)
                c1 = self.polynomial_mutation(c1, lo, hi, 1.0/dim, 20)
                c2 = self.polynomial_mutation(c2, lo, hi, 1.0/dim, 20)
                
                offspring_X[i*2] = c1
                offspring_X[i*2+1] = c2

            # Handle odd population size
            if pop_size % 2 == 1:
                p1_idx = self.tournament_select(ranks, cd, 2)
                x1 = X[p1_idx]
                c1 = x1.copy()
                c1 = self.polynomial_mutation(c1, lo, hi, 1.0/dim, 20)
                offspring_X[-1] = c1

            # 4. Evaluate offspring
            F_off = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += pop_size

            # 5. Combine parent and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])

            # 6. Non-dominated sort on combined population
            ranks_all, fronts_all = self.non_dominated_sort(F_all)
            cd_all = self.crowding_distance(F_all, fronts_all)

            # 7. Select next population
            # Sort by rank (ascending) and crowding distance (descending)
            # np.lexsort sorts by the last key first, so we put -cd first?
            # Actually, lexsort keys: (k1, k2, ...) sorts by k1, then k2.
            # We want primary key: rank (asc), secondary key: cd (desc).
            # So keys should be (-cd, rank) if we use standard sort?
            # No, lexsort uses the order of keys provided.
            # Let's use np.argsort on a structured array or just simple sorting logic.
            # Simpler: create indices sorted by rank, then stable sort by -cd within ranks.
            
            # Efficient selection:
            # Create an array of indices sorted by rank
            # Then for each rank, keep the ones with highest cd
            
            # Alternative simple approach:
            # Create a list of (rank, -cd) tuples and sort
            # But vectorized is better.
            
            # Let's use the lexsort approach carefully.
            # We want to minimize rank, and maximize cd.
            # So we sort by (rank, -cd).
            sort_keys = np.lexsort((-cd_all, ranks_all))
            next_idx = sort_keys[:pop_size]

            X = X_all[next_idx]
            F = F_all[next_idx]

        return X

    def non_dominated_sort(self, F):
        """
        Fast non-dominated sorting.
        Returns ranks (array of int, 0-based) and fronts (list of lists of indices).
        """
        n = F.shape[0]
        ranks = np.zeros(n, dtype=int)
        fronts = []

        # S[i] = set of indices that i dominates
        # n_i = number of individuals that dominate i
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)

        # Compute dominance relations
        # O(n^2 * m)
        for i in range(n):
            for j in range(i + 1, n):
                if self.dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif self.dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dom[i] += 1

        # First front: all non-dominated points
        front = []
        for i in range(n):
            if n_dom[i] == 0:
                ranks[i] = 0
                front.append(i)

        while front:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        ranks[j] = len(fronts)
                        next_front.append(j)
            front = next_front

        return ranks, fronts

    def dominates(self, a, b):
        """
        Returns True if a dominates b (a is better than or equal in all objectives,
        and strictly better in at least one).
        """
        return np.all(a <= b) and np.any(a < b)

    def crowding_distance(self, F, fronts):
        """
        Calculates crowding distance for each point.
        """
        n = F.shape[0]
        n_obj = F.shape[1]
        cd = np.zeros(n)

        for front in fronts:
            m = len(front)
            if m <= 2:
                for i in front:
                    cd[i] = np.inf
                continue

            # For each objective, sort and compute distances
            for k in range(n_obj):
                # Sort front indices by k-th objective
                sorted_idx = sorted(front, key=lambda idx: F[idx, k])

                # First and last get infinite distance
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                # For intermediate points
                if m > 2:
                    # Get the range of the k-th objective in this front
                    f_min = F[sorted_idx[0], k]
                    f_max = F[sorted_idx[-1], k]
                    f_range = f_max - f_min

                    if f_range < 1e-14:
                        continue

                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / f_range

        return cd

    def tournament_select(self, ranks, cd, k=2):
        """
        Select one index based on tournament selection.
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
        Simulated Binary Crossover.
        Returns two children.
        """
        if np.random.random() > p_c:
            return x1.copy(), x2.copy()

        c1, c2 = x1.copy(), x2.copy()
        for i in range(len(x1)):
            if np.random.random() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    x_min = min(x1[i], x2[i])
                    x_max = max(x1[i], x2[i])
                    diff = x_max - x_min

                    u = np.random.random()
                    beta = 1.0 + 2.0 * (x1[i] - x_min) / diff
                    alpha = 2.0 - beta ** (-eta_c - 1.0)
                    if u <= 1.0 / alpha:
                        beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

                    c1[i] = 0.5 * ((x1[i] + x2[i]) - beta_q * (x1[i] - x2[i]))

                    u = np.random.random()
                    beta = 1.0 + 2.0 * (x2[i] - x_min) / diff
                    alpha = 2.0 - beta ** (-eta_c - 1.0)
                    if u <= 1.0 / alpha:
                        beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

                    c2[i] = 0.5 * ((x1[i] + x2[i]) + beta_q * (x1[i] - x2[i]))

        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)

        return c1, c2

    def polynomial_mutation(self, x, lo, hi, pm, eta_m=20):
        """
        Polynomial Mutation.
        """
        x_mut = x.copy()
        for i in range(len(x)):
            if np.random.random() < pm:
                x_min = lo[i]
                x_max = hi[i]
                if x_max - x_min < 1e-14:
                    continue

                y = x[i]
                u = np.random.random()
                delta1 = (y - x_min) / (x_max - x_min)
                delta2 = (x_max - y) / (x_max - x_min)

                if u < 0.5:
                    xy = 1.0 - delta1
                    val = 1.0 - xy ** (eta_m + 1.0)
                    delta_q = (2.0 * val) ** (1.0 / (eta_m + 1.0))
                else:
                    xy = 1.0 - delta2
                    val = 1.0 - xy ** (eta_m + 1.0)
                    delta_q = (2.0 * val) ** (1.0 / (eta_m + 1.0))

                x_mut[i] = y + delta_q * (x_max - x_min)
                x_mut[i] = np.clip(x_mut[i], x_min, x_max)

        return x_mut

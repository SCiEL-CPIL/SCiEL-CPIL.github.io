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

        # Population size: N=15 gives ~13 generations with budget=200
        pop_size = min(15, budget)

        if pop_size < 2:
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # NSGA-II Loop
        while used + pop_size <= budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            ranks, fronts = self.non_dominated_sort(F)

            # Crowding distance
            cd = self.crowding_distance(F, fronts)

            # Create offspring
            offspring_X = np.empty((pop_size, dim))
            n_pairs = pop_size // 2

            for i in range(n_pairs):
                # Tournament selection for two parents
                p1_idx = self.tournament_select(ranks, cd, 3)
                p2_idx = self.tournament_select(ranks, cd, 3)

                x1, x2 = X[p1_idx], X[p2_idx]
                c1, c2 = self.sbx_crossover(x1, x2, lo, hi, 0.8, 20)
                c1 = self.polynomial_mutation(c1, lo, hi, 2.0/dim, 20)
                c2 = self.polynomial_mutation(c2, lo, hi, 2.0/dim, 20)

                offspring_X[i*2] = c1
                offspring_X[i*2+1] = c2

            # If pop_size is odd, handle the last one
            if pop_size % 2 == 1:
                p1_idx = self.tournament_select(ranks, cd, 3)
                x1 = X[p1_idx]
                c1 = x1.copy()
                c1 = self.polynomial_mutation(c1, lo, hi, 2.0/dim, 20)
                offspring_X[-1] = c1

            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += pop_size

            # Combine parent and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])

            # Non-dominated sort on combined population
            ranks_all, fronts_all = self.non_dominated_sort(F_all)
            cd_all = self.crowding_distance(F_all, fronts_all)

            # Select next population
            sort_keys = np.lexsort((-cd_all, ranks_all))
            next_idx = sort_keys[:pop_size]

            X = X_all[next_idx]
            F = F_all[next_idx]

            # Update archive: keep only non-dominated points from combined set
            combined_X = np.vstack([archive_X, X])
            combined_F = np.vstack([archive_F, F])
            # Find non-dominated points in the combined archive
            n_combined = len(combined_F)
            dominated = np.zeros(n_combined, dtype=bool)
            for i in range(n_combined):
                if dominated[i]:
                    continue
                for j in range(i + 1, n_combined):
                    if dominated[j]:
                        continue
                    if self.dominates(combined_F[i], combined_F[j]):
                        dominated[j] = True
                    elif self.dominates(combined_F[j], combined_F[i]):
                        dominated[i] = True
                        break
            mask = ~dominated
            archive_X = combined_X[mask]
            archive_F = combined_F[mask]

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

    def polynomial_mutation(self, x, lo, hi, p_m, eta_m):
        """
        Polynomial Mutation.
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
                    delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    delta_q = (x[i] - x_min) * delta
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    delta_q = (x_max - x[i]) * delta

                x_mut[i] = x[i] + delta_q
                x_mut[i] = np.clip(x_mut[i], x_min, x_max)
        return x_mut

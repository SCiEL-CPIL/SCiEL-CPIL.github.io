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

        # Population size: 20 gives good balance for 200 budget
        pop_size = 20

        if pop_size >= budget:
            # If budget is too small, just do random sampling
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X], dtype=np.float64)
            return X

        # Initialize population
        X = np.empty((pop_size, dim))
        # Use a mix of random and Latin hypercube for better initial spread
        # First, create a Latin hypercube sample
        n_lh = min(pop_size, dim)
        # Simple LHS: for each dimension, divide into pop_size intervals and pick one random point
        lh = np.empty((pop_size, dim))
        for d in range(dim):
            # Create pop_size intervals
            intervals = np.linspace(lo[d], hi[d], pop_size + 1)
            # Pick a random point in each interval
            lh[:, d] = np.random.uniform(intervals[:-1], intervals[1:])
        
        # Use the LHS for the first min(pop_size, dim) points, rest random
        n_use_lh = min(pop_size, dim)
        X[:n_use_lh] = lh[:n_use_lh]
        if pop_size > n_use_lh:
            X[n_use_lh:] = lo + (hi - lo) * np.random.random((pop_size - n_use_lh, dim))

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Main evolutionary loop
        generation = 0
        while used + pop_size <= budget:
            if time.time() > deadline:
                break

            generation += 1

            # Adaptive parameters
            progress = min(generation / 15.0, 1.0)
            p_c = 0.9
            eta_c = 15 + 10 * (1 - progress)
            p_m = 1.0 / dim
            eta_m = 20

            # Non-dominated sort
            ranks, fronts = self._fast_non_dominated_sort(F)

            # Crowding distance
            cd = self._crowding_distance(F, fronts)

            # Create offspring
            offspring_X = np.empty((pop_size, dim))
            n_pairs = pop_size // 2

            for i in range(n_pairs):
                # Tournament selection
                p1_idx = self._tournament_select(ranks, cd, 2)
                p2_idx = self._tournament_select(ranks, cd, 2)

                x1, x2 = X[p1_idx], X[p2_idx]
                c1, c2 = self._sbx_crossover(x1, x2, lo, hi, p_c, eta_c)
                c1 = self._polynomial_mutation(c1, lo, hi, p_m, eta_m)
                c2 = self._polynomial_mutation(c2, lo, hi, p_m, eta_m)

                offspring_X[i * 2] = c1
                offspring_X[i * 2 + 1] = c2

            if pop_size % 2 == 1:
                p1_idx = self._tournament_select(ranks, cd, 2)
                x1 = X[p1_idx]
                c1 = self._polynomial_mutation(x1.copy(), lo, hi, p_m * 2, eta_m)
                offspring_X[-1] = c1

            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += pop_size

            # Combine parent and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])

            # Non-dominated sort on combined population
            ranks_all, fronts_all = self._fast_non_dominated_sort(F_all)
            cd_all = self._crowding_distance(F_all, fronts_all)

            # Select next population
            sort_keys = np.lexsort((-cd_all, ranks_all))
            next_idx = sort_keys[:pop_size]

            X = X_all[next_idx]
            F = F_all[next_idx]

        # If we have remaining budget, do local refinement
        if used < budget:
            remaining = budget - used
            if time.time() < deadline - 10:
                n_refine = min(remaining, 10)
                for _ in range(n_refine):
                    if time.time() > deadline:
                        break
                    # Pick a random point from current population and refine
                    idx = np.random.choice(len(X))
                    x_base = X[idx].copy()
                    # Small perturbation
                    perturbation = 0.005 * (hi - lo)
                    x_new = x_base + perturbation * np.random.random(dim) * (np.random.random(dim) * 2 - 1)
                    x_new = np.clip(x_new, lo, hi)
                    f_new = np.array(problem(x_new), dtype=np.float64)
                    used += 1
                    # Check if it's non-dominated
                    is_dominated = False
                    for j in range(len(X)):
                        if self._dominates(F[j], f_new):
                            is_dominated = True
                            break
                    if not is_dominated:
                        # Replace the worst point in the population
                        # Find the point with the highest rank and lowest crowding distance
                        ranks_cur, fronts_cur = self._fast_non_dominated_sort(F)
                        cd_cur = self._crowding_distance(F, fronts_cur)
                        worst_idx = np.lexsort((cd_cur, ranks_cur))[-1]
                        X[worst_idx] = x_new
                        F[worst_idx] = f_new

        return X

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sorting.
        Returns ranks (array of int, 0-based) and fronts (list of lists of indices).
        """
        n = F.shape[0]
        if n == 0:
            return np.array([], dtype=int), []
        
        ranks = np.zeros(n, dtype=int)
        fronts = []

        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                fi = F[i]
                fj = F[j]
                # fi dominates fj?
                if np.all(fi <= fj) and np.any(fi < fj):
                    S[i].append(j)
                    n_dom[j] += 1
                # fj dominates fi?
                elif np.all(fj <= fi) and np.any(fj < fi):
                    S[j].append(i)
                    n_dom[i] += 1

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

    def _dominates(self, a, b):
        """
        Returns True if a dominates b.
        """
        return np.all(a <= b) and np.any(a < b)

    def _crowding_distance(self, F, fronts):
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

            for k in range(n_obj):
                sorted_idx = sorted(front, key=lambda idx: F[idx, k])

                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                if m > 2:
                    f_min = F[sorted_idx[0], k]
                    f_max = F[sorted_idx[-1], k]
                    f_range = f_max - f_min

                    if f_range < 1e-14:
                        continue

                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i + 1], k] - F[sorted_idx[i - 1], k]) / f_range

        return cd

    def _tournament_select(self, ranks, cd, k=2):
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

    def _sbx_crossover(self, x1, x2, lo, hi, p_c=0.9, eta_c=20):
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

        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, p_m, eta_m):
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

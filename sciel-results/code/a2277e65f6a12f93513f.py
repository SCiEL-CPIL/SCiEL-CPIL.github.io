"""Combined optimizer: Sobol init + adaptive SBX-NSGA-II with archive and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: Sobol init, adaptive SBX-NSGA-II with archive."""

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
        deadline = start_time + 270

        pop_size = min(24, budget)
        if pop_size < 2:
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X

        # Sobol initialization for space-filling coverage
        X, F, used = self._sobol_init(problem, lo, hi, pop_size, budget)

        # Archive to track all non-dominated points found
        archive_X = X.copy()
        archive_F = F.copy()

        # Track stagnation
        prev_archive_size = len(archive_X)
        stagnation_counter = 0
        F_adapt = 0.5  # DE scale factor, starts moderate
        CR = 0.7  # Crossover rate (SBX probability)
        eta_c = 20  # SBX distribution index
        eta_m = 20  # Mutation distribution index

        while used + pop_size <= budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            ranks, fronts = self._non_dominated_sort(F)

            # Crowding distance
            cd = self._crowding_distance(F, fronts)

            # Adaptive: increase exploration if stagnating
            if len(archive_X) <= prev_archive_size:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter >= 3:
                F_adapt = min(0.9, F_adapt + 0.1)
                CR = min(0.95, CR + 0.05)
                eta_c = max(10, eta_c - 2)  # More diverse crossover
                eta_m = max(10, eta_m - 2)  # More diverse mutation
            elif stagnation_counter == 0:
                F_adapt = max(0.3, F_adapt - 0.05)
                CR = max(0.5, CR - 0.03)
                eta_c = min(30, eta_c + 1)  # More focused crossover
                eta_m = min(30, eta_m + 1)  # More focused mutation

            # Create offspring using SBX crossover + polynomial mutation
            offspring_X = np.empty((pop_size, dim))
            n_pairs = pop_size // 2

            for i in range(n_pairs):
                # Select parents via tournament
                p1 = self._tournament_select(ranks, cd, 2)
                p2 = self._tournament_select(ranks, cd, 2)

                x1, x2 = X[p1], X[p2]
                c1, c2 = self._sbx_crossover(x1, x2, lo, hi, CR, eta_c)

                # Polynomial mutation
                c1 = self._polynomial_mutation(c1, lo, hi, 1.0 / dim, eta_m)
                c2 = self._polynomial_mutation(c2, lo, hi, 1.0 / dim, eta_m)

                offspring_X[i * 2] = c1
                offspring_X[i * 2 + 1] = c2

            if pop_size % 2 == 1:
                p1 = self._tournament_select(ranks, cd, 2)
                c1 = X[p1].copy()
                c1 = self._polynomial_mutation(c1, lo, hi, 1.0 / dim, eta_m)
                offspring_X[-1] = c1

            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += pop_size

            # Combine parent and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])

            # Update archive with any new non-dominated points
            archive_X, archive_F = self._update_archive(archive_X, archive_F, X_all, F_all)
            prev_archive_size = len(archive_X)

            # Select next population using non-dominated sort + crowding
            ranks_all, fronts_all = self._non_dominated_sort(F_all)
            cd_all = self._crowding_distance(F_all, fronts_all)
            sort_keys = np.lexsort((-cd_all, ranks_all))
            next_idx = sort_keys[:pop_size]

            X = X_all[next_idx]
            F = F_all[next_idx]

        # Use remaining budget for local refinement of archive points
        remaining = budget - used
        if remaining > 0 and len(archive_X) > 0:
            self._local_refinement(problem, lo, hi, archive_X, archive_F, remaining, deadline)

        return archive_X

    def _sobol_init(self, problem, lo, hi, pop_size, budget):
        """Generate Sobol-based initial population."""
        dim = hi.shape[0]
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=dim, scramble=True, seed=0)
            sobol_n = 1
            while sobol_n < pop_size:
                sobol_n *= 2
            sobol_pts = sobol.random(sobol_n)
            X = lo + (hi - lo) * sobol_pts[:pop_size]
        except Exception:
            X = lo + (hi - lo) * np.random.random((pop_size, dim))

        F = np.array([problem(x) for x in X], dtype=np.float64)
        return X, F, pop_size

    def _update_archive(self, archive_X, archive_F, X_new, F_new):
        """Update archive with new non-dominated points."""
        # Combine archive and new points
        X_all = np.vstack([archive_X, X_new])
        F_all = np.vstack([archive_F, F_new])

        # Find non-dominated points
        n = len(X_all)
        is_non_dom = np.ones(n, dtype=bool)

        for i in range(n):
            if not is_non_dom[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F_all[j], F_all[i]):
                    is_non_dom[i] = False
                    break

        return X_all[is_non_dom], F_all[is_non_dom]

    def _local_refinement(self, problem, lo, hi, archive_X, archive_F, remaining, deadline):
        """Local refinement using scalarized objectives."""
        dim = hi.shape[0]
        n_obj = archive_F.shape[1]
        n_archive = len(archive_X)
        if n_archive == 0:
            return

        per_point = max(1, remaining // n_archive)
        for idx in range(n_archive):
            if time.time() > deadline:
                break
            if remaining <= 0:
                break

            for _ in range(min(per_point, remaining)):
                if time.time() > deadline:
                    break
                # Random weighted sum direction
                weights = np.random.dirichlet(np.ones(n_obj))
                base = archive_X[idx].copy()
                step = 0.03 * (hi - lo)
                candidate = base + step * np.random.randn(dim)
                candidate = np.clip(candidate, lo, hi)

                fc = problem(candidate)
                remaining -= 1

                # Accept if non-dominated or improves scalarized objective
                scalar_old = np.dot(weights, archive_F[idx])
                scalar_new = np.dot(weights, fc)

                if scalar_new < scalar_old:
                    archive_X[idx] = candidate
                    archive_F[idx] = fc
                elif not self._is_dominated(fc, archive_F):
                    # Add to archive if non-dominated
                    archive_X = np.vstack([archive_X, candidate])
                    archive_F = np.vstack([archive_F, fc])
                    n_archive = len(archive_X)

    def _is_dominated(self, f_new, F_archive):
        """Check if f_new is dominated by any point in archive."""
        for f in F_archive:
            if self._dominates(f, f_new):
                return True
        return False

    def _non_dominated_sort(self, F):
        n = F.shape[0]
        ranks = np.zeros(n, dtype=int)
        fronts = []

        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif self._dominates(F[j], F[i]):
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
        return np.all(a <= b) and np.any(a < b)

    def _crowding_distance(self, F, fronts):
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

    def _polynomial_mutation(self, x, lo, hi, p_m=1.0 / 12, eta_m=20):
        x = x.copy()
        for i in range(len(x)):
            if np.random.random() < p_m:
                y = x[i]
                y_min = lo[i]
                y_max = hi[i]
                diff = y_max - y_min
                if diff < 1e-14:
                    continue

                u = np.random.random()
                if u < 0.5:
                    xy = (y - y_min) / diff
                    beta = 1.0 + 2.0 * (1.0 - xy)
                    alpha = 2.0 + beta * (eta_m + 1.0)
                    delta = (2.0 * (u + 1.0 / alpha) ** (1.0 / (eta_m + 1.0)) - 1.0)
                else:
                    xy = (y_max - y) / diff
                    beta = 1.0 + 2.0 * xy
                    alpha = 2.0 + beta * (eta_m + 1.0)
                    delta = 1.0 - (2.0 * (1.0 - u + 1.0 / alpha)) ** (1.0 / (eta_m + 1.0))

                y = y + delta * diff
                x[i] = np.clip(y, y_min, y_max)
        return x
# EVOLVE-BLOCK-END

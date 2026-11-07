"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size: balance exploration vs exploitation
        # With budget 200, pop_size=40 gives ~5 generations
        pop_size = 40

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size

        while evals < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size and evals < self.budget:
                # Tournament selection for two parents
                p1 = self._tournament_select(X, F, rng, k=2)
                p2 = self._tournament_select(X, F, rng, k=2)

                # SBX crossover
                if rng.random() < 0.9:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Polynomial mutation
                c1 = self._polynomial_mutation(c1, lo, hi, rng, eta=20)
                c2 = self._polynomial_mutation(c2, lo, hi, rng, eta=20)

                # Evaluate offspring
                if evals + 1 > self.budget:
                    break

                f1 = problem(c1)
                offspring_X.append(c1)
                offspring_F.append(f1)
                evals += 1

                if evals + 1 > self.budget or len(offspring_X) >= pop_size:
                    break

                f2 = problem(c2)
                offspring_X.append(c2)
                offspring_F.append(f2)
                evals += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])

            # Non-dominated sorting
            fronts = self._non_dominated_sort(F_comb)

            # Select next generation
            next_X = []
            next_F = []
            front_idx = 0

            while front_idx < len(fronts) and len(next_X) < pop_size:
                front = fronts[front_idx]
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                else:
                    # Need crowding distance to select from this front
                    F_front = F_comb[front]
                    dist = self._crowding_distance(F_front)
                    sorted_idx = np.argsort(-dist)
                    needed = pop_size - len(next_X)
                    for i in range(needed):
                        idx = front[sorted_idx[i]]
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                    break
                front_idx += 1

            X = np.array(next_X)
            F = np.array(next_F)

        return X

    def _tournament_select(self, X, F, rng, k=2):
        """Tournament selection based on non-dominated rank."""
        n = len(X)
        if n <= k:
            return X[rng.integers(n)].copy()

        # Quick non-dominated sorting for ranking
        ranks = self._fast_non_dominated_sort(F)

        # Select k random individuals, pick best rank
        indices = rng.integers(0, n, size=k)
        best_idx = indices[0]
        best_rank = ranks[best_idx]
        for i in indices[1:]:
            if ranks[i] < best_rank:
                best_rank = ranks[i]
                best_idx = i
            elif ranks[i] == best_rank:
                # Tie-break by crowding distance approximation
                # For speed, just pick randomly among ties
                if rng.random() < 0.5:
                    best_idx = i
        return X[best_idx].copy()

    def _sbx_crossover(self, p1, p2, lo, hi, rng, eta=15.0):
        """Simulated Binary Crossover."""
        c1, c2 = p1.copy(), p2.copy()
        for i in range(self.dim):
            if rng.random() < 0.5:
                continue
            if abs(p1[i] - p2[i]) < 1e-14:
                    c1[i] = c2[i] = p1[i]
                    continue

            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))

            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])

            # Clip to bounds
            c1[i] = np.clip(c1[i], lo[i], hi[i])
            c2[i] = np.clip(c2[i], lo[i], hi[i])

        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, rng, eta=20.0):
        """Polynomial mutation."""
        x_mut = x.copy()
        for i in range(self.dim):
            if rng.random() < 0.5:
                continue

            u = rng.random()
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))

            x_range = hi[i] - lo[i]
            x_mut[i] = x[i] + delta * x_range
            x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])

        return x_mut

    def _non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []

        # dominance count and dominated set
        S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
        n_dominates = np.zeros(n, dtype=int)  # number of solutions that dominate i
        fronts = []

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominates[i] += 1

        # First front
        front = [i for i in range(n) if n_dominates[i] == 0]
        if not front:
            # All dominated? Shouldn't happen, but fallback
            front = list(range(n))
        fronts.append(front)

        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front

        return fronts

    def _fast_non_dominated_sort(self, F):
        """Return rank array (lower is better) for tournament selection."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)

        S = [[] for _ in range(n)]
        n_dominates = np.zeros(n, dtype=int)
        ranks = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominates[i] += 1

        front = [i for i in range(n) if n_dominates[i] == 0]
        if not front:
            front = list(range(n))

        rank = 0
        while front:
            for i in front:
                ranks[i] = rank
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        next_front.append(j)
            front = next_front
            rank += 1

        return ranks

    def _crowding_distance(self, F):
        """Calculate crowding distance for a front."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        for m in range(self.n_obj):
            sorted_idx = np.argsort(F[:, m])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf

            f_min = F[sorted_idx[0], m]
            f_max = F[sorted_idx[-1], m]
            f_range = f_max - f_min
            if f_range < 1e-14:
                continue

            for i in range(1, n - 1):
                dist[sorted_idx[i]] += (F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]) / f_range

        return dist
# EVOLVE-BLOCK-END

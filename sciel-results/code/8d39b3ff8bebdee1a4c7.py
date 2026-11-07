# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II style search with Halton initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, dim, base_offset=0):
        """Generate Halton sequence points in [0,1]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2  # primes: 2, 3, 5, 7, ...
            for i in range(n):
                f = 1.0
                idx = i + 1 + base_offset
                res = 0.0
                while idx > 0:
                    f /= base
                    res += f * (idx % base)
                    idx //= base
                points[i, d] = res
        return points

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sort.
        Returns list of fronts, each front is a list of indices.
        """
        n = len(F)
        S = [[] for _ in range(n)]
        n_count = np.zeros(n, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i dominates j?
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_count[j] += 1
                # j dominates i?
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_count[i] += 1

        for i in range(n):
            if n_count[i] == 0:
                fronts.append([i])

        k = 0
        while k < len(fronts):
            current_front = fronts[k]
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        cd = np.zeros(n)
        F_sub = F[indices]

        for m in range(self.n_obj):
            # Sort by objective m
            order = np.argsort(F_sub[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            if F_sub[order[-1], m] - F_sub[order[0], m] == 0:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / (
                    F_sub[order[-1], m] - F_sub[order[0], m])
        return cd

    def _tournament_select(self, ranks, F, k=2):
        """Tournament selection based on rank and crowding distance."""
        idx1 = np.random.randint(len(ranks))
        idx2 = np.random.randint(len(ranks))
        if ranks[idx1] < ranks[idx2]:
            return idx1
        elif ranks[idx2] < ranks[idx1]:
            return idx2
        else:
            # Same rank, use crowding distance
            front_indices = [i for i, r in enumerate(ranks) if r == ranks[idx1]]
            cd = self._crowding_distance(F, front_indices)
            pos1 = front_indices.index(idx1)
            pos2 = front_indices.index(idx2)
            if cd[pos1] >= cd[pos2]:
                return idx1
            else:
                return idx2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = min(30, max(10, budget // 6))

        # Initialize with Halton sequence for space-filling
        halton_pts = self._halton_sequence(N, dim)
        X = lo + (hi - lo) * halton_pts
        F = np.array([problem(x) for x in X])
        used = N

        # Track all evaluated points for hypervolume
        all_X = list(X)
        all_F = list(F)

        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = self._fast_non_dominated_sort(F)

            # Assign ranks
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Create new population
            new_X = []
            new_F = []

            # Elitism: keep best N/2
            half = N // 2
            sorted_idx = np.lexsort((ranks, np.arange(len(F))))
            for i in range(min(half, len(F))):
                new_X.append(X[sorted_idx[i]])
                new_F.append(F[sorted_idx[i]])

            # Generate offspring
            while len(new_X) < N:
                if time.time() > deadline:
                    break

                p1 = self._tournament_select(ranks, F)
                p2 = self._tournament_select(ranks, F)

                # Simulated binary crossover
                if np.random.rand() < 0.9:
                    child1 = np.empty(dim)
                    child2 = np.empty(dim)
                    for j in range(dim):
                        if np.random.rand() < 0.5:
                            child1[j] = X[p1, j]
                            child2[j] = X[p2, j]
                        else:
                            child1[j] = X[p2, j]
                            child2[j] = X[p1, j]
                else:
                    child1 = X[p1].copy()
                    child2 = X[p2].copy()

                # Polynomial mutation
                for child in [child1, child2]:
                    for j in range(dim):
                        if np.random.rand() < 0.8:
                            delta = 0.1 * (hi[j] - lo[j])
                            child[j] += delta * np.random.randn()
                            child[j] = np.clip(child[j], lo[j], hi[j])

                if used < budget:
                    f1 = problem(child1)
                    new_X.append(child1)
                    new_F.append(f1)
                    all_X.append(child1)
                    all_F.append(f1)
                    used += 1

                if used < budget:
                    f2 = problem(child2)
                    new_X.append(child2)
                    new_F.append(f2)
                    all_X.append(child2)
                    all_F.append(f2)
                    used += 1

            X = np.array(new_X[:N])
            F = np.array(new_F[:N])

        return X
# EVOLVE-BLOCK-END

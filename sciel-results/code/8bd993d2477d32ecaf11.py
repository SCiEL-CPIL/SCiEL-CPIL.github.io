"""Rewritten multi-objective optimizer for WFG7 (3 objectives, 12 vars)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Return list of fronts (lists of indices) and a dominance rank array."""
        n = len(F)
        S = [[] for _ in range(n)]   # S[i] = indices dominated by i
        n_d = np.zeros(n, dtype=int)  # number of solutions that dominate i
        fronts = []
        rank = np.full(n, -1, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_d[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_d[i] += 1

        # First front
        front0 = [i for i in range(n) if n_d[i] == 0]
        for i in front0:
            rank[i] = 0
        fronts.append(front0)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            if next_front:
                k += 1
                fronts.append(next_front)

        return fronts, rank

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for the given set of indices."""
        if len(indices) == 0:
            return np.array([])
        if len(indices) == 1:
            return np.array([np.inf])

        dist = np.zeros(len(indices))
        m = self.n_obj
        for m_idx in range(m):
            vals = F[indices, m_idx]
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            v_min = vals[order[0]]
            v_max = vals[order[-1]]
            rng = v_max - v_min
            if rng > 1e-12:
                for i in range(1, len(order) - 1):
                    dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / rng
        return dist

    def _select_parents(self, F, pop_size):
        """Tournament selection based on rank and crowding distance."""
        fronts, rank = self._non_dominated_sort(F)
        # Compute crowding distance for each front
        cd = np.zeros(len(F))
        for front in fronts:
            cd_front = self._crowding_distance(F, front)
            for idx, d in zip(front, cd_front):
                cd[idx] = d

        def dominates_in_selection(a, b):
            if rank[a] < rank[b]:
                return True
            if rank[a] > rank[b]:
                return False
            return cd[a] > cd[b]

        parents = []
        for _ in range(pop_size):
            a, b = np.random.randint(0, len(F), 2)
            parents.append(a if dominates_in_selection(a, b) else b)
        return parents

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(40, budget // 5)
        if pop_size < 10:
            pop_size = 10

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best hypervolume-relevant points (all recorded)
        all_F = F.copy()

        # NSGA-II loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Selection
            parents = self._select_parents(F, pop_size)

            # Crossover and mutation to generate offspring
            offspring_X = np.zeros((pop_size, dim))
            for i in range(pop_size):
                p1 = X[parents[i]]
                p2 = X[parents[(i + 1) % pop_size]]
                # Simulated binary crossover
                if np.random.random() < 0.9:
                    # SBX-like crossover
                    beta = np.random.random(dim)
                    child = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                else:
                    child = p1.copy()
                # Gaussian mutation
                child += 0.1 * (hi - lo) * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)
                offspring_X[i] = child

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined
            fronts, rank = self._non_dominated_sort(combined_F)

            # Compute crowding distance for fronts
            cd = np.zeros(len(combined_F))
            for front in fronts:
                cd_front = self._crowding_distance(combined_F, front)
                for idx, d in zip(front, cd_front):
                    cd[idx] = d

            # Select next population
            new_idx = []
            for front in fronts:
                if len(new_idx) + len(front) <= pop_size:
                    new_idx.extend(front)
                else:
                    # Sort front by crowding distance descending
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = pop_size - len(new_idx)
                    new_idx.extend(front_sorted[:needed])
                    break

            X = combined_X[new_idx]
            F = combined_F[new_idx]
            all_F = np.vstack([all_F, offspring_F])

        return X
# EVOLVE-BLOCK-END

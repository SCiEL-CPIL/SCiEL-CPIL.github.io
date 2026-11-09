"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary search with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Return list of fronts, each a list of indices."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        for m in range(self.n_obj):
            vals = F[indices, m]
            min_v, max_v = vals.min(), vals.max()
            if max_v - min_v < 1e-12:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for k in range(1, n - 1):
                dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_v - min_v)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Population size: balance exploration and exploitation
        pop_size = min(40, max(10, self.budget // 5))

        # Latin hypercube-style initial sampling for diversity
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                X[perm[i], d] = lo[d] + (hi[d] - lo[d]) * (i + np.random.random()) / pop_size

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Adaptive mutation scale
        range_ = hi - lo
        base_scale = 0.3

        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = self._non_dominated_sort(F)

            # Select parent from best front using crowding distance
            best_front = fronts[0]
            cd = self._crowding_distance(F, best_front)
            # Weighted selection: prefer high crowding distance
            probs = cd / (cd.sum() + 1e-12)
            parent_idx = best_front[np.random.choice(len(best_front), p=probs)]
            parent = X[parent_idx]

            # Adaptive mutation: larger steps when population is spread out
            spread = np.std(X, axis=0)
            scale = base_scale * (1.0 + 0.5 * (spread / (range_ + 1e-12)).mean())
            mutation = np.random.standard_normal(dim) * scale * range_
            child = np.clip(parent + mutation, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst point (highest rank, then lowest crowding)
            # Find worst: last front, then lowest crowding distance in that front
            worst_front = fronts[-1]
            cd_worst = self._crowding_distance(F, worst_front)
            worst_pos_in_front = np.argmin(cd_worst)
            worst_idx = worst_front[worst_pos_in_front]

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END

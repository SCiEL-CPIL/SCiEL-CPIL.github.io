import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for WFG4 using LHS initialization + evolutionary refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # safety margin from 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Latin Hypercube Sampling for initial population
        n_init = 80
        lhs = self._latin_hypercube(n_init, self.dim, rng)
        X = lo + (hi - lo) * lhs
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = n_init

        # Keep all evaluated points for hypervolume calculation
        all_X = [X.copy()]
        all_F = [F.copy()]

        # Evolutionary refinement
        pop_size = min(40, self.budget - used)
        # Select non-dominated points for parent pool
        n_idx = self._non_dominated_indices(F)
        parents = X[n_idx] if len(n_idx) > 0 else X
        parent_F = F[n_idx] if len(n_idx) > 0 else F

        while used < self.budget:
            if time.time() - t0 > deadline:
                break

            # Pick parents with diversity-aware selection
            if len(parents) >= 2:
                i = rng.integers(len(parents))
                j = rng.integers(len(parents))
                if i == j:
                    j = (j + 1) % len(parents)
                parent1 = parents[i]
                parent2 = parents[j]
            elif len(parents) == 1:
                parent1 = parent2 = parents[0]
            else:
                parent1 = parent2 = X[rng.integers(len(X))]

            # Crossover
            child = 0.5 * (parent1 + parent2) + 0.5 * rng.standard_normal(self.dim) * np.abs(parent1 - parent2)

            # Mutation
            mutation_scale = 0.1 * (hi - lo)
            child = child + rng.standard_normal(self.dim) * mutation_scale
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Record for hypervolume
            all_X.append(child.reshape(1, -1))
            all_F.append(fc.reshape(1, -1))

            # Update population: replace worst point
            worst_idx = self._worst_point_index(F)
            X[worst_idx] = child
            F[worst_idx] = fc

            # Periodically refresh non-dominated set
            if used % 20 == 0:
                n_idx = self._non_dominated_indices(F)
                parents = X[n_idx] if len(n_idx) > 0 else X
                parent_F = F[n_idx] if len(n_idx) > 0 else F

        return X

    def _latin_hypercube(self, n, dim, rng):
        """Generate n points in [0,1]^dim using Latin Hypercube Sampling."""
        points = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            points[:, d] = (perm + rng.random(n)) / n
        return points

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        non_dominated = np.ones(n, dtype=bool)
        for i in range(n):
            if not non_dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    non_dominated[i] = False
                    break
        return np.where(non_dominated)[0]

    def _worst_point_index(self, F):
        """Find the index of the worst point using a simple sum-based criterion."""
        # Use sum of normalized objectives as a rough proxy
        sums = np.sum(F, axis=1)
        return int(np.argmax(sums))

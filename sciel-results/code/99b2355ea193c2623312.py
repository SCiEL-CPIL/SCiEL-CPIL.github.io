"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential-evolution multi-objective optimizer with non-dominated archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Latin hypercube-style initial population for good spread
        pop_size = min(30, max(10, self.budget // 6))
        X = np.zeros((pop_size, self.dim))
        for j in range(self.dim):
            perm = rng.permutation(pop_size)
            X[:, j] = lo[j] + span[j] * (perm + 0.5) / pop_size
        # Add slight jitter
        X += span * rng.random((pop_size, self.dim)) * 0.01
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for final archive
        all_X = [X.copy()]
        all_F = [F.copy()]

        def non_dominated_sort(F):
            """Return Pareto front indices (non-dominated points)."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def crowding_distance(F, indices):
            """Compute crowding distance for points at given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            F_sub = F[indices]
            for k in range(self.n_obj):
            # Sort by objective k
                order = np.argsort(F_sub[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = F_sub[order[-1], k] - F_sub[order[0], k]
                if f_range == 0:
                    continue
                for m in range(1, n - 1):
                    cd[order[m]] += (F_sub[order[m + 1], k] - F_sub[order[m - 1], k]) / f_range
            return cd

        # Main DE loop
        while used < self.budget:
            # Get non-dominated front
            front_idx = non_dominated_sort(F)
            
            if len(front_idx) < 2:
                # Fallback: use all points
                parents = np.arange(pop_size)
            else:
                # Use non-dominated points as parents
                parents = front_idx
            
            # Select parent triplet for DE mutation
            n_par = len(parents)
            if n_par < 3:
                # Not enough diversity, use random parents from full population
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                i3 = rng.integers(pop_size)
                base = rng.integers(pop_size)
            else:
                i1 = parents[rng.integers(n_par)]
                i2 = parents[rng.integers(n_par)]
                i3 = parents[rng.integers(n_par)]
                base = parents[rng.integers(n_par)]
                # Ensure distinct indices where possible
                while i2 == i1:
                    i2 = parents[rng.integers(n_par)]
                while i3 == i1 or i3 == i2:
                    i3 = parents[rng.integers(n_par)]

            # Adaptive F parameter: start high, decrease over time
            progress = used / self.budget
            F_param = 0.8 * (1.0 - 0.5 * progress) + 0.3 * progress

            # DE/rand/1 mutation
            mutant = X[base] + F_param * (X[i1] - X[i2] + X[i3] - X[base])
            # Add small random perturbation for exploration
            mutant += 0.05 * span * rng.standard_normal(self.dim)
            mutant = np.clip(mutant, lo, hi)

            f_mutant = problem(mutant)
            used += 1

            # Selection: replace worst point in population
            # Use a combination of dominance and crowding
            # Simple approach: replace point with highest objective sum among dominated
            dominated_mask = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_mask[i] = True
                        break

            if np.any(dominated_mask):
                # Among dominated, pick the one with worst (largest) objective sum
                dominated_indices = np.where(dominated_mask)[0]
                worst_in_dominated = dominated_indices[np.argmax(np.sum(F[dominated_indices], axis=1))]
                replace_idx = worst_in_dominated
            else:
                # All non-dominated: use crowding distance to find least crowded
                cd = crowding_distance(F, np.arange(pop_size))
                replace_idx = int(np.argmin(cd))

            X[replace_idx] = mutant
            F[replace_idx] = f_mutant

            # Periodically add to archive
            all_X.append(X.copy())
            all_F.append(F.copy())

        return X
# EVOLVE-BLOCK-END

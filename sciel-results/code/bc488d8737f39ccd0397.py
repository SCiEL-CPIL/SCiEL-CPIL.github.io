"""Multi-objective optimizer for RE37: NSGA-II with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with deadline safety."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 20

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        pop_size = self.pop_size

        rng = np.random.default_rng(0)

        # Initialization
        X = lo + (hi - lo) * rng.random((pop_size, d))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main NSGA-II loop
        while used + pop_size <= self.budget:
            if time.time() >= deadline:
                break

            # Calculate ranks and crowding distance
            ranks, cd = self._calculate_rank_cd(F)

            # Create offspring
            Q = np.zeros((pop_size, d))
            FQ = np.zeros((pop_size, self.n_obj))

            for i in range(pop_size):
                p1 = self._tournament(ranks, cd, rng)
                p2 = self._tournament(ranks, cd, rng)

                x1 = X[p1]
                x2 = X[p2]

                # BLX-Alpha crossover
                alpha = 0.5
                low = np.minimum(x1, x2) - alpha * np.abs(x1 - x2)
                high = np.maximum(x1, x2) + alpha * np.abs(x1 - x2)
                child = rng.uniform(low, high, size=d)
                child = np.clip(child, lo, hi)

                # Gaussian mutation
                for k in range(d):
                    if rng.random() < 1.0 / d:
                        child[k] += 0.1 * (hi[k] - lo[k]) * rng.standard_normal()
                        child[k] = np.clip(child[k], lo[k], hi[k])

                Q[i] = child
                FQ[i] = problem(child)

            used += pop_size

            # Combine and select
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, FQ])

            X, F = self._nsga2_select(R_X, R_F)

        # Tail: use remaining budget with simple replacement
        while used < self.budget:
            if time.time() >= deadline:
                break

            # Pick a parent from non-dominated set
            dominated = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F
            ])
            parents = X[~dominated] if np.any(~dominated) else X
            parent = parents[rng.integers(len(parents))]

            # Mutate
            child = parent + 0.1 * (hi - lo) * rng.standard_normal(d)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point in population
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X

    def _calculate_rank_cd(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        cd = np.zeros(n)

        # Non-dominated sorting
        remaining = set(range(n))
        current_rank = 0

        while remaining:
            front_indices = []
            for i in list(remaining):
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front_indices.append(i)
                    remaining.remove(i)

            if not front_indices:
                break

            for i in front_indices:
                ranks[i] = current_rank

            if len(front_indices) > 1:
                for m in range(self.n_obj):
                    order = np.argsort(F[front_indices, m])
                    sorted_indices = [front_indices[k] for k in order]

                    cd[sorted_indices[0]] = np.inf
                    cd[sorted_indices[-1]] = np.inf

                    span = F[sorted_indices[-1], m] - F[sorted_indices[0], m]
                    if span > 0:
                        for k in range(1, len(sorted_indices) - 1):
                            idx_curr = sorted_indices[k]
                            idx_prev = sorted_indices[k - 1]
                            idx_next = sorted_indices[k + 1]
                            cd[idx_curr] += (F[idx_next, m] - F[idx_prev, m]) / span
            else:
                for i in front_indices:
                    cd[i] = np.inf

            current_rank += 1

        return ranks, cd

    def _tournament(self, ranks, cd, rng):
        i1 = rng.integers(len(ranks))
        i2 = rng.integers(len(ranks))

        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            if cd[i1] > cd[i2]:
                return i1
            elif cd[i2] > cd[i1]:
                return i2
            else:
                return i1

    def _nsga2_select(self, R_X, R_F):
        ranks, cd = self._calculate_rank_cd(R_F)

        # Sort by rank asc, then cd desc
        sorted_indices = np.lexsort((-cd, ranks))

        selected_idx = sorted_indices[:self.pop_size]

        return R_X[selected_idx], R_F[selected_idx]
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        rng = np.random.default_rng(42)

        # Use a smaller population to allow more generations within budget
        pop_size = 20

        # Initialize population with diverse points
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II parameters
        mu = pop_size
        eta = pop_size

        # Track best hypervolume for early stopping
        ref_point = np.array([2.5, 2.5, 2.5])

        while used + eta <= self.budget and time.time() < deadline:
            # Create offspring
            children_X = []
            children_F = []

            for i in range(eta):
                if time.time() > deadline:
                    break

                # Binary tournament selection
                idx1 = rng.integers(mu)
                idx2 = rng.integers(mu)
                parent_idx = self._tournament_select(F, idx1, idx2, rng)

                # Polynomial mutation (more appropriate for DTLZ4)
                child = X[parent_idx].copy()
                for d in range(self.dim):
                    if rng.random() < 0.7:  # Mutation probability
                        delta = rng.random()
                        if delta < 0.5:
                            u = (2.0 * delta) ** (1.0 / 21.0)
                        else:
                            u = (2.0 * (1.0 - delta)) ** (1.0 / 21.0)
                        child[d] += span[d] * (u - 1.0)

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                children_X.append(child)
                children_F.append(fc)

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parents and offspring
            combined_X = np.vstack([X, children_X])
            combined_F = np.vstack([F, children_F])

            # Non-dominated sorting
            fronts = self._non_dominated_sort(combined_F)

            # Build next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= mu:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    remaining = mu - len(new_X)
                    front_F = combined_F[front]
                    cd = self._crowding_distance(front_F)
                    sorted_indices = np.argsort(-cd)
                    for idx in sorted_indices[:remaining]:
                        orig_idx = front[idx]
                        new_X.append(combined_X[orig_idx])
                        new_F.append(combined_F[orig_idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X

    def _tournament_select(self, F, idx1, idx2, rng):
        """Binary tournament selection based on dominance."""
        f1 = F[idx1]
        f2 = F[idx2]

        if np.all(f1 <= f2) and np.any(f1 < f2):
            return idx1
        elif np.all(f2 <= f1) and np.any(f2 < f1):
            return idx2
        else:
            return idx1 if rng.random() < 0.5 else idx2

    def _non_dominated_sort(self, F):
        """Non-dominated sorting. Returns list of fronts, each a list of indices."""
        n = len(F)
        S = [[] for _ in range(n)]
        n_dominates = [0] * n
        fronts = []

        for i in range(n):
            for j in range(i + 1, n):
                fi = F[i]
                fj = F[j]
                if np.all(fi <= fj) and np.any(fi < fj):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif np.all(fj <= fi) and np.any(fj < fi):
                    S[j].append(i)
                    n_dominates[i] += 1

        front = [i for i in range(n) if n_dominates[i] == 0]
        while front:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        next_front.append(j)
            front = next_front

        return fronts

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of solutions in the same front."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)

        cd = np.zeros(n)
        m = F.shape[1]

        for k in range(m):
            order = np.argsort(F[:, k])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_range = F[order[-1], k] - F[order[0], k]
            if f_range == 0:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (F[order[i + 1], k] - F[order[i - 1], k]) / f_range

        return cd
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using NSGA-II-like approach with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Initial population
        pop_size = 40
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Crowding distance
        def crowding_distance(F):
            n = len(F)
            if n <= 1:
                return np.ones(n)
            cd = np.zeros(n)
            for m in range(F.shape[1]):
                order = np.argsort(F[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                if F[order[-1], m] - F[order[0], m] > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / (F[order[-1], m] - F[order[0], m])
            return cd

        # Non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1
                if n_dominates[i] == 0:
                    fronts[0].append(i)

            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            fronts.pop()
            return fronts

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size // 2

            for _ in range(n_offspring):
                # Binary tournament from non-dominated front
                fronts = fast_non_dominated_sort(F)
                if not fronts or not fronts[0]:
                    break
                front = fronts[0]
                i1 = front[rng.integers(len(front))]
                i2 = front[rng.integers(len(front))]

                # Select parent with better crowding distance
                cd = crowding_distance(F)
                if cd[i1] >= cd[i2]:
                    p1_idx = i1
                else:
                    p1_idx = i2

                # Mutate
                sigma = 0.1
                child = X[p1_idx] + sigma * range_ * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

                if time.time() > deadline:
                    break

            if not offspring_X:
                break

            # Combine and select
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.vstack([F, np.array(offspring_F)])
            fronts = fast_non_dominated_sort(F_comb)

            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    # Sort by crowding distance
                    cd = crowding_distance(F_comb[front])
                    order = np.argsort(-cd)
                    needed = pop_size - len(new_X)
                    for i in range(needed):
                        idx = front[order[i]]
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END

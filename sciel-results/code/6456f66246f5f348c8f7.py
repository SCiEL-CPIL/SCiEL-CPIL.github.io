"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using efficient NSGA-II with structured initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        d = self.dim
        n_obj = self.n_obj

        # For DTLZ4, the first ~9-10 variables determine distance to front,
        # last ~2-3 determine position. Initialize distance vars near 0.5.
        pop_size = 30

        # Structured initialization: place points along the expected Pareto front
        # Distance variables (first 9) centered at 0.5, position vars (last 3) spread out
        X = np.zeros((pop_size, d))
        for i in range(pop_size):
            # Distance vars: near 0.5 with small noise
            X[i, :9] = 0.5 + 0.05 * rng.standard_normal(9)
            # Position vars: spread across [0,1]
            X[i, 9] = (i + 0.5) / pop_size
            X[i, 10] = rng.random()
            X[i, 11] = rng.random()
            X[i] = np.clip(X[i], lo, hi)

        F = np.array([problem(x) for x in X])
        used = pop_size

        def crowding_distance(F):
            n = len(F)
            if n <= 1:
                return np.ones(n)
            cd = np.zeros(n)
            for m in range(F.shape[1]):
                order = np.argsort(F[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                span = F[order[-1], m] - F[order[0], m]
                if span > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / span
            return cd

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)
            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            fronts.pop()
            return fronts

        # Main loop: generate offspring and select
        while used < self.budget:
            if time.time() > deadline:
                break

            # Determine how many offspring we can actually generate
            n_off = min(pop_size, self.budget - used)
            if n_off <= 0:
                break

            # Pre-compute fronts and crowding distance for current population
            # This is much faster than doing it for every offspring
            fronts = fast_non_dominated_sort(F)
            cd = crowding_distance(F)
            
            # Get indices of the first front for selection
            if not fronts or not fronts[0]:
                break
            first_front_indices = np.array(fronts[0])
            
            offspring_X = []
            offspring_F = []

            for _ in range(n_off):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break

                # Tournament selection from the pre-computed first front
                i1 = first_front_indices[rng.integers(len(first_front_indices))]
                i2 = first_front_indices[rng.integers(len(first_front_indices))]

                # Select parent based on crowding distance
                p_idx = i1 if cd[i1] >= cd[i2] else i2

                # Mutation: blend toward 0.5 for distance vars, random for position
                child = X[p_idx].copy()
                # Distance vars: pull toward 0.5 with some noise
                child[:9] = 0.5 + 0.1 * (child[:9] - 0.5) + 0.05 * rng.standard_normal(9)
                # Position vars: small perturbation
                child[9:] += 0.1 * rng.standard_normal(3)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

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
                    cd_comb = crowding_distance(F_comb[front])
                    order = np.argsort(-cd_comb)
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

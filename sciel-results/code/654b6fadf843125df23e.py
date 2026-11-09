"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a hybrid approach:
    1. Latin Hypercube sampling for initial spread
    2. Differential Evolution with non-dominated sorting
    3. Local search on promising points
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 315.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        rng = np.random.default_rng(42)

        # Phase 1: Latin Hypercube Sampling for good initial spread
        n_lhs = min(30, budget // 5)
        X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            X[:, d] = lo[d] + span[d] * (perm + 0.5) / n_lhs
        F = np.array([problem(x) for x in X])
        used = n_lhs

        # Helper: check if point is dominated
        def is_dominated(f, F_arr):
            return np.any(np.all(F_arr <= f, axis=1) & np.any(F_arr < f, axis=1))

        # Helper: non-dominated set indices
        def non_dominated_indices(F_arr):
            n = len(F_arr)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        # Helper: crowding distance
        def crowding_distance(F_arr, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            F_sub = F_arr[indices]
            cd = np.zeros(len(indices))
            for k in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_min = F_sub[sorted_idx[0], k]
                f_max = F_sub[sorted_idx[-1], k]
                if f_max - f_min < 1e-15:
                    continue
                for i in range(1, len(indices) - 1):
                    cd[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], k] - F_sub[sorted_idx[i - 1], k]) / (f_max - f_min)
            return cd

        # Phase 2: Differential Evolution style search
        pop_size = min(20, budget // 10)
        while used < budget - 10:
            if time.time() > deadline - 5:
                break

            # Get non-dominated front
            nd_idx = non_dominated_indices(F[:used])
            if len(nd_idx) == 0:
                nd_idx = [0]

            # Select parents from non-dominated front
            if len(nd_idx) >= 3:
                parents_idx = rng.choice(nd_idx, size=3, replace=False)
                x1, x2, x3 = X[parents_idx[0]], X[parents_idx[1]], X[parents_idx[2]]
            else:
                # Fallback: random selection
                parents_idx = rng.choice(used, size=min(3, used), replace=False)
                x1, x2, x3 = X[parents_idx[0]], X[parents_idx[1]], X[parents_idx[2]]

            # Differential evolution mutation
            F_factor = 0.8
            CR = 0.5
            v = x1 + F_factor * (x2 - x3)
            trial = np.zeros(dim)
            for d in range(dim):
                if rng.random() < CR or d == rng.integers(dim):
                    trial[d] = v[d]
                else:
                    trial[d] = x1[d]
            trial = np.clip(trial, lo, hi)

            ft = problem(trial)
            used += 1

            # Replacement strategy: keep if non-dominated or better
            # Find the worst point to replace
            worst_idx = np.argmax(np.sum(F[:used], axis=1))
            X[worst_idx] = trial
            F[worst_idx] = ft

        # Phase 3: Local refinement on best points
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 10:
            # Sort by sum of objectives
            order = np.argsort(np.sum(F[:used], axis=1))
            top_k = min(5, used)
            for _ in range(remaining):
                if time.time() > deadline - 5:
                    break
                base_idx = order[rng.integers(top_k)]
                x_base = X[base_idx]
                # Small perturbation
                step = 0.05 * span
                x_trial = x_base + step * rng.standard_normal(dim)
                x_trial = np.clip(x_trial, lo, hi)
                ft = problem(x_trial)
                used += 1
                # Replace worst if trial is better
                worst_idx = np.argmax(np.sum(F[:used], axis=1))
                if np.sum(ft) < np.sum(F[worst_idx]):
                    X[worst_idx] = x_trial
                    F[worst_idx] = ft

        return X
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer using an NSGA-II-like strategy with
    non-dominated sorting, crowding distance, and adaptive mutation.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a reasonable fraction of the budget
        pop_size = min(50, max(10, budget // 4))

        # RNG
        rng = np.random.default_rng(42)

        # Helper: fast non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = [0] * n
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[j] += 1
                if n_dominates[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            return fronts

        # Helper: crowding distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n

            F_sub = F[indices]
            cd = [0.0] * n

            for m in range(n_obj):
                sorted_indices = np.argsort(F_sub[:, m])
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')

                f_range = F_sub[sorted_indices[-1], m] - F_sub[sorted_indices[0], m]
                if f_range == 0:
                    continue

                for i in range(1, n - 1):
                    cd[sorted_indices[i]] += (F_sub[sorted_indices[i + 1], m] - F_sub[sorted_indices[i - 1], m]) / f_range

            return cd

        # Helper: tournament selection
        def tournament_select(fronts, cd_map, k=2):
            all_indices = []
            for rank, front in enumerate(fronts):
                for idx in front:
                    all_indices.append((idx, rank))

            selected = []
            for _ in range(k):
                i = rng.integers(len(all_indices))
                selected.append(all_indices[i])

            if selected[0][1] < selected[1][1]:
                return selected[0][0]
            elif selected[1][1] < selected[0][1]:
                return selected[1][0]
            else:
                cd0 = cd_map.get(selected[0][0], 0)
                cd1 = cd_map.get(selected[1][0], 0)
                return selected[0][0] if cd0 > cd1 else selected[1][0]

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        evals_used = pop_size

        # Main evolutionary loop
        while evals_used < budget:
            if time.time() > deadline:
                break

            # Compute non-dominated fronts and crowding distances
            fronts = fast_non_dominated_sort(F)

            cd_map = {}
            for front in fronts:
                if len(front) > 0:
                    cds = crowding_distance(F, front)
                    for idx, cd in zip(front, cds):
                        cd_map[idx] = cd

            # Generate offspring
            X_offspring = []
            F_offspring = []

            for _ in range(pop_size):
                if evals_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select parents via tournament
                p1 = tournament_select(fronts, cd_map)
                p2 = tournament_select(fronts, cd_map)

                x1 = X[p1]
                x2 = X[p2]

                # SBX Crossover
                if rng.random() < 0.9:
                    eta_c = 20
                    x_c = np.zeros(dim)
                    for i in range(dim):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta_c + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1 / (eta_c + 1))
                        x_c[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                else:
                    x_c = x1.copy()

                # Polynomial mutation
                eta_m = 20
                for i in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u < 0.5:
                            delta1 = (2 * u) ** (1 / (eta_m + 1)) - 1
                        else:
                            delta1 = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                        x_c[i] += delta1 * (hi[i] - lo[i])

                x_c = np.clip(x_c, lo, hi)

                # Evaluate
                fc = problem(x_c)
                evals_used += 1
                X_offspring.append(x_c)
                F_offspring.append(fc)

            if not X_offspring:
                break

            X_offspring = np.array(X_offspring)
            F_offspring = np.array(F_offspring)

            # Combine parents and offspring
            X_combined = np.vstack([X, X_offspring])
            F_combined = np.vstack([F, F_offspring])

            # Non-dominated sort on combined population
            fronts_combined = fast_non_dominated_sort(F_combined)

            # Select next generation
            X_new = []
            F_new = []
            cd_map_new = {}

            for front in fronts_combined:
                if len(X_new) + len(front) <= pop_size:
                    # Take entire front
                    for idx in front:
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                    cds = crowding_distance(F_combined, front)
                    for idx, cd in zip(front, cds):
                        cd_map_new[idx] = cd
                else:
                    # Need to select subset from this front
                    remaining = pop_size - len(X_new)
                    cds = crowding_distance(F_combined, front)
                    sorted_by_cd = sorted(zip(front, cds), key=lambda x: x[1], reverse=True)
                    for idx, cd in sorted_by_cd[:remaining]:
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                    break

            if X_new:
                X = np.array(X_new)
                F = np.array(F_new)

        return X
# EVOLVE-BLOCK-END

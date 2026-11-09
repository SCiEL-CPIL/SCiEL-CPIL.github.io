"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        pop_size = min(40, max(10, self.budget // 5))
        # Ensure we don't exceed budget
        pop_size = min(pop_size, self.budget)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # NSGA-II helper functions
        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # set of solutions dominated by i
            n_dom = [0] * n  # number of solutions dominated by i
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            fronts.pop()  # remove last empty front
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i]][m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                f_min = F[front_indices[sorted_idx[0]]][m]
                f_max = F[front_indices[sorted_idx[-1]]][m]
                if f_max - f_min < 1e-30:
                    continue
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (
                        F[front_indices[sorted_idx[k + 1]]][m] - F[front_indices[sorted_idx[k - 1]]][m]
                    ) / (f_max - f_min)
            return dist

        # Main NSGA-II loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Fast non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size

            # Binary tournament selection
            for _ in range(n_offspring):
                # Select two parents via tournament
                i1, i2 = rng.integers(0, pop_size, size=2)
                # Determine ranks
                rank_i1 = next(k for k, fr in enumerate(fronts) if i1 in fr)
                rank_i2 = next(k for k, fr in enumerate(fronts) if i2 in fr)

                if rank_i1 < rank_i2:
                    p1 = i1
                elif rank_i2 < rank_i1:
                    p1 = i2
                else:
                    # Same rank, use crowding distance
                    front_idx = fronts[rank_i1]
                    cd = crowding_distance(F, front_idx)
                    pos_i1 = front_idx.index(i1)
                    pos_i2 = front_idx.index(i2)
                    p1 = i1 if cd[pos_i1] > cd[pos_i2] else i2

                # Second parent
                i3, i4 = rng.integers(0, pop_size, size=2)
                rank_i3 = next(k for k, fr in enumerate(fronts) if i3 in fr)
                rank_i4 = next(k for k, fr in enumerate(fronts) if i4 in fr)
                if rank_i3 < rank_i4:
                    p2 = i3
                elif rank_i4 < rank_i3:
                    p2 = i4
                else:
                    front_idx = fronts[rank_i3]
                    cd = crowding_distance(F, front_idx)
                    pos_i3 = front_idx.index(i3)
                    pos_i4 = front_idx.index(i4)
                    p2 = i3 if cd[pos_i3] > cd[pos_i4] else i4

                # SBX crossover
                if rng.random() < 0.9:
                    child = np.zeros(self.dim)
                    for d in range(self.dim):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (2 + 20))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (2 + 20))
                        x1, x2 = X[p1][d], X[p2][d]
                        child[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                        child[d] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2) if rng.random() < 0.5 else child[d]
                    # Polynomial mutation
                    for d in range(self.dim):
                        if rng.random() < 1.0 / self.dim:
                            u = rng.random()
                            if u < 0.5:
                                delta = (2 * u) ** (1 / (2 + 20)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1 / (2 + 20))
                            child[d] += delta * (hi[d] - lo[d])
                else:
                    child = X[p1].copy()
                    for d in range(self.dim):
                        if rng.random() < 1.0 / self.dim:
                            u = rng.random()
                            if u < 0.5:
                                delta = (2 * u) ** (1 / (2 + 20)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1 / (2 + 20))
                            child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                offspring_X.append(child)
                fc = problem(child)
                offspring_F.append(fc)
                used += 1

                if used >= self.budget:
                    break

                if time.time() > deadline - 5.0:
                    break

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F, dtype=float)

            # Combine parent and offspring
            all_X = np.vstack([X, np.array(offspring_X)])
            all_F = np.vstack([F, offspring_F])
            n_all = len(all_F)

            # Non-dominated sorting on combined
            fronts_all = fast_non_dominated_sort(all_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                else:
                    # Need crowding distance to select remaining
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(all_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in sorted_front[:remaining]:
                        idx = front[i]
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X

# EVOLVE-BLOCK-END

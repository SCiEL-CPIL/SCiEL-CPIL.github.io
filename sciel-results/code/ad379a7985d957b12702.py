"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1.1, 1.3, 1.1])

        pop_size = min(30, self.budget // 5)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_nondominated_sort(F):
            """
            Returns:
            - fronts: list of lists of indices, each list is a non-dominated front
            - ranks: array of rank for each individual
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of j dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of individuals dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            for i in range(n):
                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 1
            while len(fronts[k - 1]) > 0:
                next_front = []
                for i in fronts[k - 1]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                    k += 1
                else:
                    break

            ranks = np.zeros(n, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            return fronts, ranks

        def crowding_distance(F, indices):
            """
            Compute crowding distance for individuals in 'indices'.
            Returns dict: index -> distance
            """
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}

            m = self.n_obj
            cd = np.zeros(len(indices))
            f_sub = F[indices]

            for j in range(m):
                sorted_idx = np.argsort(f_sub[:, j])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')

                f_range = f_sub[sorted_idx[-1], j] - f_sub[sorted_idx[0], j]
                if f_range > 1e-15:
                    for k in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[k]] += (f_sub[sorted_idx[k + 1], j] - f_sub[sorted_idx[k - 1], j]) / f_range

            # Map back to original indices
            return {indices[i]: cd[i] for i in range(len(indices))}

        def tournament_select(F, ranks, pop_size):
            """
            Binary tournament selection based on rank and crowding distance.
            """
            # Compute crowding distance for the entire population
            cd_dict = crowding_distance(F, list(range(pop_size)))

            def compare(i, j):
                if ranks[i] != ranks[j]:
                    return i if ranks[i] < ranks[j] else j
                else:
                    return i if cd_dict[i] > cd_dict[j] else j

            # Select parents
            p1 = rng.integers(pop_size)
            p2 = rng.integers(pop_size)
            parent1_idx = compare(p1, p2)

            p1 = rng.integers(pop_size)
            p2 = rng.integers(pop_size)
            parent2_idx = compare(p1, p2)

            return X[parent1_idx].copy(), X[parent2_idx].copy()

        def crossover(x1, x2):
            """Simulated Binary Crossover (SBX)"""
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()

            eta = 20.0
            x1, x2 = x1.copy(), x2.copy()
            for i in range(self.dim):
                if x1[i] == x2[i]:
                    continue
                if x1[i] > x2[i]:
                    x1[i], x2[i] = x2[i], x1[i]

                beta1 = 1.0 + (2.0 * (x1[i] - lo[i]) / (hi[i] - lo[i] + 1e-15))
                beta2 = 1.0 + (2.0 * (hi[i] - x2[i]) / (hi[i] - lo[i] + 1e-15))

                u = rng.random()
                if u <= 0.5:
                    alpha = 2.0 * u + (1.0 - 2.0 * u) * (beta1 ** (eta + 1.0))
                    beta = 1.0 / alpha
                else:
                    alpha = 2.0 * (1.0 - u) + (2.0 * u - 1.0) * (beta2 ** (eta + 1.0))
                    beta = 1.0 / alpha

                child1 = 0.5 * ((x1[i] + x2[i]) - beta * (x2[i] - x1[i]))
                child2 = 0.5 * ((x1[i] + x2[i]) + beta * (x2[i] - x1[i]))

                x1[i] = child1
                x2[i] = child2

            return np.clip(x1, lo, hi), np.clip(x2, lo, hi)

        def mutate(x):
            """Polynomial mutation"""
            if rng.random() < 0.5:
                return x.copy()

            eta = 20.0
            x = x.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    continue
                delta1 = (x[i] - lo[i]) / (hi[i] - lo[i] + 1e-15)
                delta2 = (hi[i] - x[i]) / (hi[i] - lo[i] + 1e-15)
                u = rng.random()
                if u < 0.5:
                    xy = 1.0 + (1.0 + delta1) * u
                    deltaq = (xy ** (1.0 / (eta + 1.0)) - 1.0)
                else:
                    xy = 1.0 + (1.0 - delta2) * (1.0 - u)
                    deltaq = 1.0 - (xy ** (1.0 / (eta + 1.0)))

                x[i] += deltaq * (hi[i] - lo[i])

            return np.clip(x, lo, hi)

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring_X = []
            offspring_F = []
            for _ in range(pop_size):
                if time.time() > deadline:
                    break

                x1, x2 = tournament_select(F, ranks, pop_size)
                c1, c2 = crossover(x1, x2)
                c1 = mutate(c1)
                c2 = mutate(c2)

                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)

                if used >= self.budget:
                    break

                f2 = problem(c2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            # Non-dominated sort
            fronts, ranks = fast_nondominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            next_indices = []

            for front in fronts:
                if len(next_indices) + len(front) <= pop_size:
                    next_indices.extend(front)
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_indices)
                    cd_dict = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd_dict[i], reverse=True)
                    next_indices.extend(sorted_front[:remaining])
                    break

            # Reorder and assign
            X = combined_X[next_indices]
            F = combined_F[next_indices]

            # Recompute ranks for next iteration
            _, ranks = fast_nondominated_sort(F)

        return X
# EVOLVE-BLOCK-END

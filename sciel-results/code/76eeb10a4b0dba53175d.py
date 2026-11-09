"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: random init + NSGA-II-style crowded sort,
    SBX crossover, polynomial mutation, and a deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 320  # seconds, safety margin before 330s kill

        rng = np.random
        dim, budget = self.dim, self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n_eval = 0

        # Population size: use a reasonable fraction of budget
        pop_size = min(40, max(16, budget // 5))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        n_eval = pop_size

        # Keep track of all evaluated points for hypervolume
        all_F = [F[i].copy() for i in range(pop_size)]

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sorting."""
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            for i in range(n):
                if domination_count[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F_arr, front_indices):
            """Calculate crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            m = self.n_obj

            for k in range(m):
                obj_values = F_arr[front_indices, k]
                order = np.argsort(obj_values)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                f_range = obj_values[order[-1]] - obj_values[order[0]]
                if f_range > 0:
                    for i in range(1, n - 1):
                        dist[order[i]] += (obj_values[order[i + 1]] - obj_values[order[i - 1]]) / f_range

            return dist

        def tournament_select(X_pop, F_pop, fronts, crowding_dists, n_select):
            """Tournament selection based on rank and crowding distance."""
            ranks = np.zeros(len(F_pop), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            all_crowding = np.zeros(len(F_pop))
            offset = 0
            for front in fronts:
                cd = crowding_dists[offset:offset + len(front)]
                for i, idx in enumerate(front):
                    all_crowding[idx] = cd[i]
                offset += len(front)

            selected = []
            for _ in range(n_select):
                i1 = rng.integers(len(F_pop))
                i2 = rng.integers(len(F_pop))
                if ranks[i1] < ranks[i2]:
                    selected.append(i1)
                elif ranks[i2] < ranks[i1]:
                    selected.append(i2)
                else:
                    if all_crowding[i1] > all_crowding[i2]:
                        selected.append(i1)
                    else:
                        selected.append(i2)
            return selected

        def sbx_crossover(p1, p2):
            """Simulated binary crossover."""
            if rng.random() < 0.9:
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for d in range(dim):
                    if rng.random() < 0.5:
                        if p1[d] == p2[d]:
                            child1[d] = p1[d]
                            child2[d] = p2[d]
                        else:
                            x1 = p1[d]
                            x2 = p2[d]
                            if x1 > x2:
                                x1, x2 = x2, x1
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 20))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (2 + 20))
                            child1[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                            child2[d] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                    else:
                        child1[d] = p1[d]
                        child2[d] = p2[d]
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                return child1, child2
            else:
                return p1.copy(), p2.copy()

        def polynomial_mutation(child):
            """Polynomial mutation."""
            for d in range(dim):
                if rng.random() < 1.0 / dim:
                    if child[d] == lo[d]:
                        child[d] = lo[d] + (hi[d] - lo[d]) * rng.random()
                    elif child[d] == hi[d]:
                        child[d] = hi[d] - (hi[d] - lo[d]) * rng.random()
                    else:
                        delta = rng.random()
                        if delta < 0.5:
                            child[d] -= (hi[d] - lo[d]) * delta
                        else:
                            child[d] += (hi[d] - lo[d]) * (1 - delta)
                    child[d] = np.clip(child[d], lo[d], hi[d])
            return child

        # Main evolution loop
        while n_eval < budget:
            if time.time() - start_time > DEADLINE:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Crowding distance per front
            crowding_dists = []
            for front in fronts:
                if len(front) > 0:
                    F_front = F[np.array(front)]
                    cd = crowding_distance(F_front, list(range(len(F_front))))
                    crowding_dists.append(cd)
                else:
                    crowding_dists.append(np.array([]))

            # Create offspring
            n_offspring = pop_size
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < n_offspring and n_eval < budget:
                if time.time() - start_time > DEADLINE:
                    break

                # Select parents
                parent_idx = tournament_select(X, F, fronts, crowding_dists, 2)
                p1 = X[parent_idx[0]]
                p2 = X[parent_idx[1]]

                # Crossover
                child1, child2 = sbx_crossover(p1, p2)

                # Mutation
                child1 = polynomial_mutation(child1)
                child2 = polynomial_mutation(child2)

                # Evaluate
                f1 = problem(child1)
                n_eval += 1
                offspring_X.append(child1)
                offspring_F.append(f1)
                all_F.append(f1)

                if n_eval < budget and len(offspring_X) < n_offspring:
                    f2 = problem(child2)
                    n_eval += 1
                    offspring_X.append(child2)
                    offspring_F.append(f2)
                    all_F.append(f2)

            # Combine parent and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.array([F] + [np.array(offspring_F)])
            F_combined = np.vstack([F, np.array(offspring_F)])

            # Non-dominated sorting on combined
            fronts_combined = fast_non_dominated_sort(F_combined)

            # Crowding distance per front
            crowding_dists_combined = []
            for front in fronts_combined:
                if len(front) > 0:
                    F_front = F_combined[np.array(front)]
                    cd = crowding_distance(F_front, list(range(len(F_front))))
                    crowding_dists_combined.append(cd)
                else:
                    crowding_dists_combined.append(np.array([]))

            # Select next generation
            X_next = []
            F_next = []
            for front in fronts_combined:
                if len(X_next) + len(front) <= pop_size:
                    for idx in front:
                        X_next.append(X_combined[idx])
                        F_next.append(F_combined[idx])
                else:
                    # Need to use crowding distance to select from this front
                    remaining = pop_size - len(X_next)
                    if remaining > 0:
                        F_front = F_combined[np.array(front)]
                        cd = crowding_distance(F_front, list(range(len(F_front))))
                        order = np.argsort(-cd)
                        for i in range(remaining):
                            idx = front[order[i]]
                            X_next.append(X_combined[idx])
                            F_next.append(F_combined[idx])
                    break

            X = np.array(X_next)
            F = np.array(F_next)

        return X
# EVOLVE-BLOCK-END

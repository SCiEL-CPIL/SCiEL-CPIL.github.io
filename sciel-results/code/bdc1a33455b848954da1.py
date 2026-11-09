"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using NSGA-II with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: balance exploration and computation
        pop_size = min(50, max(20, budget // 4))

        # Initialize population uniformly
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track all evaluated points for hypervolume
        all_points = F.copy()

        def fast_non_dominated_sort(F):
            """NSGA-II fast non-dominated sorting. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_dom = np.zeros(n, dtype=np.int64)  # Number of dominators
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
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for individuals in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                min_val = vals.min()
                max_val = vals.max()
                if max_val - min_val < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_val - min_val)
            return cd

        def adaptive_mutation(parent, gen, total_gens):
            """Adaptive mutation with decreasing step size."""
            # Step size decreases over generations
            progress = gen / max(total_gens, 1)
            base_step = 0.3 * (1.0 - 0.7 * progress)
            noise = base_step * np.random.standard_normal(dim)
            child = parent + noise * (hi - lo)
            return np.clip(child, lo, hi)

        # Estimate total generations
        total_gens = max(1, (budget - pop_size) // (pop_size // 2 + 1))
        gen = 0

        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Select parents from best fronts
            parents = []
            for front in fronts:
                parents.extend(front)
                if len(parents) >= pop_size // 2:
                    break
            if not parents:
                parents = list(range(pop_size))

            # Generate offspring
            n_offspring = min(pop_size // 2, budget - used)
            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                # Pick two parents (tournament-like)
                p1_idx = parents[np.random.randint(len(parents))]
                p2_idx = parents[np.random.randint(len(parents))]

                # Crossover (SBX-like)
                if np.random.random() < 0.9:
                    alpha = 15.0
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (alpha + 1))
                    else:
                        beta = (2 * (1 - u)) ** (1.0 / (alpha + 1))
                    child1 = 0.5 * ((1 + beta) * X[p1_idx] + (1 - beta) * X[p2_idx])
                    child2 = 0.5 * ((1 - beta) * X[p1_idx] + (1 + beta) * X[p2_idx])
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                else:
                    child1 = X[p1_idx].copy()
                    child2 = X[p2_idx].copy()

                # Mutate
                child1 = adaptive_mutation(child1, gen, total_gens)
                child2 = adaptive_mutation(child2, gen, total_gens)

                offspring_X.append(child1)
                offspring_F.append(problem(child1))
                used += 1

                if used < budget:
                    offspring_X.append(child2)
                    offspring_F.append(problem(child2))
                    used += 1
                else:
                    break

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)
            all_points = np.vstack([all_points, offspring_F])

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])

            # Sort combined population
            fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate this front by crowding distance
                    cd = crowding_distance(combined_F, front)
                    remaining = pop_size - len(new_X)
                    # Sort by crowding distance descending
                    order = np.argsort(cd)[::-1]
                    for i in range(min(remaining, len(order))):
                        idx = front[order[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)
            gen += 1

        return X
# EVOLVE-BLOCK-END

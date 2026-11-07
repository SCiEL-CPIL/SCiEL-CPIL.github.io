"""Multi-objective optimizer on WFG4 using NSGA-II style evolutionary algorithm."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        pop_size = 80
        evals = 0

        # Initialize population with LHS for good coverage
        def lhs_sample(n, dim, lo, hi, rng):
            """Generate n points using Latin Hypercube Sampling."""
            points = np.zeros((n, dim))
            for j in range(dim):
                strata = np.linspace(lo[j], hi[j], n + 1)
                samples = np.array([strata[i] + (strata[i+1] - strata[i]) * rng.random() for i in range(n)])
                points[:, j] = samples
            return points

        # Initial population
        population = lhs_sample(pop_size, self.dim, lo, hi, rng)
        fitness = np.zeros((pop_size, self.n_obj))
        for i in range(pop_size):
            if time.time() > deadline - 5.0:
                break
            fitness[i] = problem(population[i])
            evals += 1

        if evals == 0:
            return None

        # NSGA-II helper functions
        def fast_non_dominated_sort(F):
            """Perform fast non-dominated sorting. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_dom = np.zeros(n, dtype=int)  # Number of dominators
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
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

        def crowding_distance_assignment(F, front_indices):
            """Assign crowding distances to points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = np.argsort([F[front_indices[i], m] for i in range(n)])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf

                f_min = F[front_indices[sorted_idx[0]], m]
                f_max = F[front_indices[sorted_idx[-1]], m]
                if f_max - f_min < 1e-12:
                    continue

                for i in range(1, n - 1):
                    prev_idx = sorted_idx[i - 1]
                    next_idx = sorted_idx[i + 1]
                    dist[sorted_idx[i]] += (F[front_indices[next_idx], m] - F[front_indices[prev_idx], m]) / (f_max - f_min)

            return dist

        def assign_crowding_distances(F, fronts):
            """Assign crowding distances to all points."""
            n = len(F)
            crowd = np.zeros(n)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance_assignment(F, front)
                    for i, idx in enumerate(front):
                        crowd[idx] = cd[i]
            return crowd

        def dominates(a, b):
            """Check if a dominates b (lower is better)."""
            return np.all(a <= b) and np.any(a < b)

        def binary_tournament(pop, F, ranks, crowd):
            """Select one individual using binary tournament."""
            i1 = rng.integers(0, len(pop))
            i2 = rng.integers(0, len(pop))
            if ranks[i1] < ranks[i2]:
                return i1
            elif ranks[i2] < ranks[i1]:
                return i2
            else:
                return i1 if crowd[i1] > crowd[i2] else i2

        def de_crossover(x1, x2, x3, CR=0.8):
            """DE/rand/1 crossover."""
            v = x1 + 0.7 * (x2 - x3)
            j_rand = rng.integers(0, self.dim)
            child = np.zeros(self.dim)
            for j in range(self.dim):
                if rng.random() < CR or j == j_rand:
                    child[j] = v[j]
                else:
                    child[j] = x1[j]
            return np.clip(child, lo, hi)

        # Main NSGA-II loop
        while evals < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Perform non-dominated sorting
            fronts = fast_non_dominated_sort(fitness)

            # Compute ranks
            ranks = np.zeros(pop_size, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Compute crowding distances
            crowd = assign_crowding_distances(fitness, fronts)

            # Generate offspring
            offspring = np.zeros((pop_size, self.dim))
            offspring_F = np.zeros((pop_size, self.n_obj))

            for i in range(pop_size):
                if time.time() > deadline - 5.0:
                    break

                # Select parents
                p1 = binary_tournament(population, fitness, ranks, crowd)
                p2 = binary_tournament(population, fitness, ranks, crowd)
                p3 = binary_tournament(population, fitness, ranks, crowd)

                # Create child
                child = de_crossover(population[p1], population[p2], population[p3])
                offspring[i] = child

                # Evaluate
                if evals >= self.budget:
                    break
                offspring_F[i] = problem(child)
                evals += 1

            # Combine parent and offspring
            combined_pop = np.vstack([population, offspring])
            combined_F = np.vstack([fitness, offspring_F])
            n_combined = len(combined_pop)

            # Sort combined population
            fronts_combined = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_pop = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))
            count = 0

            for front in fronts_combined:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_pop[count] = combined_pop[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - count
                    # Compute crowding distances within this front
                    front_F = combined_F[front]
                    cd = crowding_distance_assignment(combined_F, front)
                    sorted_by_cd = np.argsort(cd)[::-1]
                    for i in range(min(remaining, len(sorted_by_cd))):
                        idx = front[sorted_by_cd[i]]
                        new_pop[count] = combined_pop[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                    break

            population = new_pop[:pop_size]
            fitness = new_F[:pop_size]

        # Return the final non-dominated set
        fronts = fast_non_dominated_sort(fitness)
        archive = []
        for front in fronts:
            for idx in front:
                archive.append({'x': population[idx].copy(), 'f': fitness[idx].copy()})

        return archive
# EVOLVE-BLOCK-END

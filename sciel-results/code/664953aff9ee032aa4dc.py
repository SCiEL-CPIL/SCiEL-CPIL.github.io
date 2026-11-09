"""Multi-objective optimizer for ZDT3 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use numpy random with the seed set by the evaluator
        rng = np.random

        # Population size
        pop_size = min(budget, 100)
        # If budget is small, use a smaller population
        if budget <= 50:
            pop_size = budget
        elif budget <= 100:
            pop_size = budget // 2
        else:
            pop_size = 100

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II parameters
        crossover_rate = 0.9
        mutation_rate = 0.1
        sigma = 0.2

        def fast_non_dominated_sort(F):
            """Perform fast non-dominated sorting."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []
            current_front = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

                if domination_count[i] == 0:
                    current_front.append(i)

            idx = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(n_obj):
                fm = F[front_indices, m]
                imin = np.argmin(fm)
                imax = np.argmax(fm)
                dist[imin] = np.inf
                dist[imax] = np.inf
                if fm[imax] - fm[imin] > 0:
                    for i in range(1, n - 1):
                        sorted_idx = np.argsort(fm)
                        # Find position of front_indices[i] in sorted order
                        pos = np.where(sorted_idx == front_indices[i])[0][0]
                        if pos > 0 and pos < n - 1:
                            prev_val = fm[sorted_idx[pos - 1]]
                            next_val = fm[sorted_idx[pos + 1]]
                            dist[i] += (next_val - prev_val) / (fm[imax] - fm[imin])
            return dist

        def tournament_select(F, fronts, crowding, k=2):
            """Tournament selection."""
            n = len(F)
            selected = []
            for _ in range(k):
                idx = rng.integers(n)
                selected.append(idx)

            # Compare based on front rank and crowding distance
            best = selected[0]
            for idx in selected[1:]:
                # Find front ranks
                rank_best = -1
                rank_idx = -1
                for fi, front in enumerate(fronts):
                    if best in front:
                        rank_best = fi
                        break
                    if idx in front:
                        rank_idx = fi
                        break

                if rank_idx < rank_best:
                    best = idx
                elif rank_idx == rank_best:
                    # Same front, use crowding distance
                    if crowding[idx] > crowding[best]:
                        best = idx
            return best

        def sbx_crossover(x1, x2, lo, hi, dim):
            """Simulated binary crossover."""
            if rng.random() < crossover_rate:
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for i in range(dim):
                    if rng.random() < 0.5:
                        beta = 1.0 / (1.0 + 2.0 * rng.random())
                        alpha = 2.0 - beta ** (1.0 + 2.0 * 15.0)
                    else:
                        beta = 1.0 / (1.0 + 2.0 * rng.random())
                        alpha = 2.0 - beta ** (1.0 + 2.0 * 15.0)

                    if rng.random() < 0.5:
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child2[i] = 0.5 * ((1 + 1.0 / beta) * x1[i] + (1 - 1.0 / beta) * x2[i])
                    else:
                        child1[i] = 0.5 * ((1 + 1.0 / beta) * x1[i] + (1 - 1.0 / beta) * x2[i])
                        child2[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def polynomial_mutation(x, lo, hi, dim, rate, sigma):
            """Polynomial mutation."""
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < rate:
                    y = rng.random()
                    if y < 0.5:
                        delta = (2.0 * y) ** (1.0 / (1.0 + 2.0 * 20.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (1.0 - y)) ** (1.0 / (1.0 + 2.0 * 20.0))
                    x_mut[i] = x[i] + delta * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main NSGA-II loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Calculate crowding distances
            crowding = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowding[idx] = cd[i]

            # Create offspring
            offspring_X = []
            offspring_F = []

            # Select parents and create offspring
            while len(offspring_X) < pop_size:
                # Check deadline
                if time.time() > deadline:
                    break

                # Tournament selection
                p1 = tournament_select(F, fronts, crowding)
                p2 = tournament_select(F, fronts, crowding)

                # Crossover
                c1, c2 = sbx_crossover(X[p1], X[p2], lo, hi, dim)

                # Mutation
                c1 = polynomial_mutation(c1, lo, hi, dim, mutation_rate, sigma)
                c2 = polynomial_mutation(c2, lo, hi, dim, mutation_rate, sigma)

                # Evaluate
                f1 = problem(c1)
                used += 1
                f2 = problem(c2)
                used += 1

                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)

                # Check budget
                if used >= budget:
                    break

            if len(offspring_X) == 0:
                break

            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array(offspring_F[:pop_size])

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sorting on combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(new_X)
                    # Calculate crowding for this front
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance (descending)
                    sorted_indices = np.argsort(-cd)
                    for i in range(min(remaining, len(front))):
                        idx = front[sorted_indices[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X[:pop_size])
            F = np.array(new_F[:pop_size])

        return X

# EVOLVE-BLOCK-END

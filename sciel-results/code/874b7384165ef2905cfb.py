"""Multi-objective DTLZ2 optimizer: Differential Evolution with time-budget loop."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size: balance diversity vs. per-generation cost
        pop_size = 80
        self.dim = self.dim
        self.n_obj = self.n_obj

        # Initialize population uniformly
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts (indices) sorted by non-domination level."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            current_front = [i for i in range(n) if domination_count[i] == 0]
            fronts.append(current_front)
            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[indices, m]
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                rng_span = vals[order[-1]] - vals[order[0]]
                if rng_span < 1e-15:
                    continue
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_span
            return dist

        def de_generate_parents(X, F, n_parents):
            """Generate parent indices using DE best/1/bin mutation."""
            # Find non-dominated front for selection
            fronts = fast_non_dominated_sort(F)
            # Build rank array
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            # Select parents from top fronts with crowding distance
            # Get top 2 fronts
            top_indices = []
            for front in fronts[:2]:
                top_indices.extend(front)
            if len(top_indices) < n_parents:
                top_indices = list(range(len(F)))
            # Sample parents with replacement, weighted toward better ranks
            parent_indices = []
            for _ in range(n_parents):
                # Pick a random individual from top_indices
                parent_indices.append(rng.integers(len(top_indices)))
            return [top_indices[i] for i in parent_indices]

        # Main DE loop
        while used < self.budget and time.time() < deadline:
            # Check time more frequently
            if time.time() > deadline - 5:
                break

            # Generate offspring via DE mutation
            n_offspring = pop_size
            offspring = np.empty((n_offspring, self.dim))
            offspring_F = np.empty((n_offspring, self.n_obj))

            for i in range(n_offspring):
                # DE best/1/bin mutation
                # Pick 3 distinct random individuals
                a, b, c = rng.integers(pop_size, size=3)
                while b == a:
                    b = rng.integers(pop_size)
                while c == a or c == b:
                    c = rng.integers(pop_size)

                # Mutation: v = X[a] + F * (X[b] - X[c])
                F_factor = 0.8
                v = X[a] + F_factor * (X[b] - X[c])
                # Crossover (binomial)
                crossover_rate = 0.5
                j_rand = rng.integers(self.dim)
                mask = rng.random(self.dim) < crossover_rate
                mask[j_rand] = True
                child = np.where(mask, v, X[i])
                child = np.clip(child, lo, hi)

                offspring[i] = child
                offspring_F[i] = problem(child)
                used += 1

                if time.time() > deadline - 5:
                    break

            if time.time() > deadline - 5:
                # Partial generation: only evaluate what we have
                n_offspring = i + 1
                offspring = offspring[:n_offspring]
                offspring_F = offspring_F[:n_offspring]

            # Combine parent and offspring populations
            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F, offspring_F])
            total = len(X_combined)

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F_combined)

            # Select next population
            X_new = np.empty((0, self.dim))
            F_new = np.empty((0, self.n_obj))
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    # Need to select only some from this front using crowding distance
                    remaining = pop_size - len(selected)
                    cd = crowding_distance(F_combined, front)
                    order = np.argsort(cd)[::-1]  # sort by descending crowding distance
                    selected.extend(front[order[:remaining]])
                    break

            X_new = X_combined[selected]
            F_new = F_combined[selected]
            X = X_new
            F = F_new

            # If we hit the budget, stop
            if used >= self.budget:
                break

        return X
# EVOLVE-BLOCK-END

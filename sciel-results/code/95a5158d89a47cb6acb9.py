"""Multi-objective DTLZ2 optimizer: Structured initialization + DE + local polish."""
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
        deadline = start_time + 270  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        dim = self.dim
        n_obj = self.n_obj

        def fast_non_dominated_sort(F):
            """Return list of fronts (indices) sorted by non-domination level."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
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
            fronts = [current_front]
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
            for m in range(n_obj):
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

        def select_population(X_combined, F_combined, pop_size):
            """Select next population using non-dominated sorting + crowding distance."""
            fronts = fast_non_dominated_sort(F_combined)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    remaining = pop_size - len(selected)
                    cd = crowding_distance(F_combined, front)
                    order = np.argsort(cd)[::-1]
                    selected.extend(front[order[:remaining]])
                    break
            return X_combined[selected], F_combined[selected]

        # Phase 1: Structured initialization on DTLZ2 Pareto front
        # DTLZ2: g = sum((x_i - 0.5)^2) for i in [M..N-1]
        # f_k = (1+g) * cos(theta_1 * cos(theta_2) ... )
        # On Pareto front, g=0, so x_i = 0.5 for distance variables
        # Shape variables determine angles
        n_shape = n_obj - 1  # 2 shape variables for 3 objectives
        n_dist = dim - n_shape  # 10 distance variables

        # Create diverse angle combinations
        n_init = 60
        angles = np.zeros((n_init, n_shape))
        # Use a grid-like spread with some randomness
        for i in range(n_init):
            angles[i, 0] = (i / n_init) * (np.pi / 2) + rng.random() * (np.pi / n_init)
            angles[i, 1] = rng.random() * (np.pi / 2)

        X = np.zeros((n_init, dim))
        # Distance variables at 0.5 (on Pareto front)
        X[:, n_shape:] = 0.5
        # Shape variables: convert angles to decision variables
        # For DTLZ2, the shape variables are the angles themselves
        X[:, :n_shape] = angles
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = n_init

        # Phase 2: DE evolution
        pop_size = 60
        # Ensure we have at least pop_size individuals
        if used < pop_size:
            extra = pop_size - used
            X_extra = lo + (hi - lo) * rng.random((extra, dim))
            F_extra = np.array([problem(x) for x in X_extra])
            used += extra
            X = np.vstack([X, X_extra])
            F = np.vstack([F, F_extra])

        # DE loop
        while used < self.budget and time.time() < deadline:
            if time.time() > deadline - 10:
                break

            n_offspring = pop_size
            offspring = np.empty((n_offspring, dim))
            offspring_F = np.empty((n_offspring, n_obj))

            for i in range(n_offspring):
                # DE/rand/1/bin mutation
                a, b, c = rng.integers(pop_size, size=3)
                while b == a:
                    b = rng.integers(pop_size)
                while c == a or c == b:
                    c = rng.integers(pop_size)

                F_factor = 0.5 + 0.3 * rng.random()
                v = X[a] + F_factor * (X[b] - X[c])

                crossover_rate = 0.3
                j_rand = rng.integers(dim)
                mask = rng.random(dim) < crossover_rate
                mask[j_rand] = True
                child = np.where(mask, v, X[i])
                child = np.clip(child, lo, hi)

                offspring[i] = child
                offspring_F[i] = problem(child)
                used += 1

                if time.time() > deadline - 10:
                    break
                if used >= self.budget:
                    break

            n_actual = min(i + 1, n_offspring)
            offspring = offspring[:n_actual]
            offspring_F = offspring_F[:n_actual]

            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F, offspring_F])

            X, F = select_population(X_combined, F_combined, pop_size)

            if used >= self.budget:
                break

        # Phase 3: Local polish - use remaining budget for local search on best points
        if used < self.budget and time.time() < deadline - 5:
            # Get non-dominated front
            fronts = fast_non_dominated_sort(F)
            nd_front = fronts[0] if fronts else list(range(len(F)))

            # Local search around non-dominated points
            step_size = 0.1
            for idx in nd_front:
                if used >= self.budget or time.time() > deadline - 5:
                    break
                # Try small perturbations
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline - 5:
                        break
                    perturbation = rng.normal(0, step_size, dim)
                    candidate = X[idx] + perturbation
                    candidate = np.clip(candidate, lo, hi)
                    f_cand = problem(candidate)
                    used += 1
                    # Check if it dominates any existing point or is non-dominated
                    is_dominated = False
                    for j in range(len(F)):
                        if np.all(F[j] <= f_cand) and np.any(F[j] < f_cand):
                            is_dominated = True
                            break
                    if not is_dominated:
                        # Replace the worst dominated point or add if we have room
                        # Find points dominated by f_cand
                        dominated_by_cand = [j for j in range(len(F)) 
                                            if np.all(f_cand <= F[j]) and np.any(f_cand < F[j])]
                        if dominated_by_cand:
                            # Replace one of them
                            worst = dominated_by_cand[0]
                            X[worst] = candidate
                            F[worst] = f_cand
                        else:
                            # Add to population if we have room
                            if len(X) < pop_size + 20:
                                X = np.vstack([X, candidate])
                                F = np.vstack([F, f_cand])

        return X
# EVOLVE-BLOCK-END

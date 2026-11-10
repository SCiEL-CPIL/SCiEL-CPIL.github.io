"""Multi-objective optimizer for RE34 using NSGA-II with local polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with local polishing phase."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size
        pop_size = min(20, budget // 4)
        if pop_size < 4:
            pop_size = 4

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: fast non-dominated sort
        def fast_non_dominated_sort(F):
            """Returns list of fronts, each front is a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # Domination sets
            n_d = np.zeros(n, dtype=int)  # Number of dominators
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1

            current_front = [i for i in range(n) if n_d[i] == 0]
            rank = np.full(n, -1, dtype=int)
            front_idx = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    rank[i] = front_idx
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                current_front = next_front
                front_idx += 1
            return fronts

        # Helper: crowding distance
        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            dist = np.zeros(len(indices))
            for m in range(n_obj):
                vals = F[indices, m]
                min_val, max_val = vals.min(), vals.max()
                if max_val - min_val < 1e-12:
                    dist[:] = np.inf
                    continue
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for k in range(1, len(indices) - 1):
                    dist[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_val - min_val)
            return dist

        # Helper: tournament selection
        def tournament_select(F, parents_X, parents_F, k=2):
            """Select k individuals via tournament on non-dominated front + crowding."""
            n = len(parents_F)
            # Quick non-dominated sort for selection
            fronts = fast_non_dominated_sort(parents_F)
            # Pick from first front primarily
            if fronts:
                first_front = fronts[0]
                if len(first_front) > 1:
                    cd = crowding_distance(parents_F, first_front)
                    # Weighted random from first front based on crowding
                    weights = cd / (cd.sum() + 1e-12)
                    idx = rng.choice(first_front, p=weights)
                else:
                    idx = first_front[0]
            else:
                idx = rng.integers(n)
            return parents_X[idx].copy()

        # Helper: DE mutation
        def de_mutation(X, F, base_idx, scale=0.5):
            """Differential evolution mutation."""
            n = len(X)
            # Pick two distinct indices different from base
            other = [i for i in range(n) if i != base_idx]
            i1 = other[rng.integers(len(other))]
            other2 = [i for i in other if i != i1]
            i2 = other2[rng.integers(len(other2))]
            mutant = X[base_idx] + scale * (X[i1] - X[i2])
            return np.clip(mutant, lo, hi)

        # Helper: local polish (local search around a point)
        def local_polish(x0, f0, problem, max_iters=50, step_scale=0.05):
            """Local search using random perturbations, accepting improvements."""
            best_x = x0.copy()
            best_f = np.array(f0, dtype=float)
            for _ in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                # Random perturbation
                step = step_scale * (hi - lo) * rng.standard_normal(dim)
                new_x = np.clip(best_x + step, lo, hi)
                new_f = problem(new_x)
                # Check if new point dominates best
                if np.all(new_f <= best_f) and np.any(new_f < best_f):
                    best_x = new_x
                    best_f = new_f
                    step_scale *= 0.8  # Reduce step size
            return best_x, best_f

        # Main NSGA-II loop
        polish_start = int(budget * 0.75)  # Start polishing in last 25%

        while used < budget:
            if time.time() > deadline - 10.0:
                break

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, budget - used)

            for _ in range(n_offspring):
                if time.time() > deadline - 10.0:
                    break

                if used >= polish_start:
                    # Local polishing phase: mutate from best non-dominated points
                    fronts = fast_non_dominated_sort(F)
                    if fronts:
                        first_front = fronts[0]
                        base_idx = first_front[rng.integers(len(first_front))]
                    else:
                        base_idx = rng.integers(pop_size)

                    # Try local polish
                    if rng.random() < 0.5:
                        new_x, new_f = local_polish(X[base_idx], F[base_idx], problem,
                                                     max_iters=10, step_scale=0.02)
                        used += 10  # Approximate cost
                    else:
                        new_x = de_mutation(X, F, base_idx, scale=0.3)
                        new_f = problem(new_x)
                        used += 1
                else:
                    # Exploration phase: DE mutation
                    base_idx = rng.integers(pop_size)
                    scale = 0.5 + 0.3 * rng.random()  # Adaptive scale
                    new_x = de_mutation(X, F, base_idx, scale=scale)
                    new_f = problem(new_x)
                    used += 1

                offspring_X.append(new_x)
                offspring_F.append(new_f)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined
            fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(next_X)
                    if remaining > 0:
                        cd = crowding_distance(combined_F, front)
                        # Sort by crowding distance descending
                        order = np.argsort(-cd)
                        for k in range(remaining):
                            idx = front[order[k]]
                            next_X.append(combined_X[idx])
                            next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        # Final polish: improve the best non-dominated points
        if time.time() < deadline - 5.0:
            fronts = fast_non_dominated_sort(F)
            if fronts:
                first_front = fronts[0]
                for idx in first_front[:5]:  # Polish top 5
                    if time.time() > deadline - 5.0:
                        break
                    new_x, new_f = local_polish(X[idx], F[idx], problem,
                                                max_iters=20, step_scale=0.01)
                    # Replace if improved
                    if np.all(new_f <= F[idx]) and np.any(new_f < F[idx]):
                        X[idx] = new_x
                        F[idx] = new_f

        return X
# EVOLVE-BLOCK-END

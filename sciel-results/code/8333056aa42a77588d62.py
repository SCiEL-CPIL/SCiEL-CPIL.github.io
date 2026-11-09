"""Multi-objective optimizer on WFG7 using NSGA-II with Chebyshev scalarization seeding."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with Chebyshev scalarization seeding and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def elapsed_ok():
            return time.time() < deadline

        # --- Chebyshev scalarization seeding ---
        # Generate weight vectors on a simplex grid for n_obj objectives
        def simplex_weights(n, levels):
            """Generate weight vectors on a simplex grid."""
            def gen(k, remaining, total, current):
                if k == 1:
                    current.append(remaining / total)
                    yield list(current)
                    current.pop()
                    return
                for i in range(remaining + 1):
                    current.append(i / total)
                    yield from gen(k - 1, remaining - i, total, current)
                    current.pop()
            return list(gen(n, levels, levels, []))

        levels = 4  # gives 10 weight vectors for 3 objectives
        weights = np.array(simplex_weights(n_obj, levels), dtype=np.float64)
        # Normalize to sum to 1
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Evaluate a Chebyshev scalarization from a random start using a few gradient steps
        def chebyshev_objective(x, w):
            f = problem(x)
            f = np.asarray(f, dtype=np.float64)
            return np.max(w * f)

        # Seed population: for each weight, do a few local refinement steps
        pop_size = min(36, budget // 5)
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        used = 0

        # First, random initialization for diversity
        n_random = pop_size // 2
        X[:n_random] = lo + span * rng.random((n_random, dim))
        for i in range(n_random):
            F[i] = problem(X[i])
            used += 1
            if used >= budget or not elapsed_ok():
                break

        # Chebyshev-seeded points: for a subset of weights, start from random and do few gradient steps
        n_cheb = min(len(weights), pop_size - n_random)
        selected_weights = weights[rng.integers(0, len(weights), n_cheb)]
        for i in range(n_cheb):
            if used >= budget or not elapsed_ok():
                break
            w = selected_weights[i]
            # Start from a random point
            x = lo + span * rng.random(dim)
            fx = np.asarray(problem(x), dtype=np.float64)
            used += 1
            if used >= budget or not elapsed_ok():
                break
            # A few simple gradient-free refinement steps (Nelder-Mead-like)
            for _ in range(3):
                if used >= budget or not elapsed_ok():
                    break
                # Perturb in direction of reducing weighted sum
                direction = -w * fx
                step = 0.05 * span * rng.standard_normal(dim)
                direction = direction / (np.linalg.norm(direction) + 1e-12)
                direction = direction + 0.1 * step
                x_new = np.clip(x + direction, lo, hi)
                fx_new = np.asarray(problem(x_new), dtype=np.float64)
                used += 1
                if np.max(w * fx_new) < np.max(w * fx):
                    x, fx = x_new, fx_new
            idx = n_random + i
            X[idx] = x
            F[idx] = fx

        # Fill remaining with random
        for i in range(pop_size):
            if used >= budget or not elapsed_ok():
                break
            if X[i].sum() == 0 and F[i].sum() == 0:
                X[i] = lo + span * rng.random(dim)
                F[i] = problem(X[i])
                used += 1

        # NSGA-II evolution
        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination count
            domination_count = [0] * n
            fronts = [[]]
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        domination_count[p] += 1
                if domination_count[p] == 0:
                    fronts[0].append(p)
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        domination_count[q] -= 1
                        if domination_count[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            dist = np.zeros(n)
            F_sub = F[indices]
            for m in range(n_obj):
                min_v = F_sub[:, m].min()
                max_v = F_sub[:, m].max()
                if max_v - min_v < 1e-12:
                    continue
                order = np.argsort(F_sub[:, m])
                dist[order[0]] = float('inf')
                dist[order[-1]] = float('inf')
                for i in range(1, n - 1):
                    dist[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / (max_v - min_v)
            return dist

        def sbx_crossover(x1, x2, eta=20.0):
            """Simulated binary crossover."""
            child1 = np.empty(dim)
            child2 = np.empty(dim)
            for j in range(dim):
                if rng.random() < 0.5:
                    beta = 1.0 / (2.0 * rng.random() + 1e-12)
                    if rng.random() <= 1.0 / (beta + 1.0):
                        alpha = 2.0 - beta ** (eta + 1.0)
                    else:
                        alpha = (2.0 * beta - 1.0) ** (-(eta + 1.0))
                    beta = 0.5 * (1.0 + alpha)
                    child1[j] = 0.5 * ((1.0 + beta) * x1[j] + (1.0 - beta) * x2[j])
                    child2[j] = 0.5 * ((1.0 - beta) * x1[j] + (1.0 + beta) * x2[j])
                else:
                    child1[j] = x2[j]
                    child2[j] = x1[j]
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def polynomial_mutate(x, eta=20.0):
            """Polynomial mutation."""
            x_new = x.copy()
            for j in range(dim):
                if rng.random() < 0.8:  # mutation probability per gene
                    y = x_new[j]
                    yl = lo[j]
                    yh = hi[j]
                    delta1 = (y - yl) / (yh - yl + 1e-12)
                    delta2 = (yh - y) / (yh - yl + 1e-12)
                    rand = rng.random()
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                        deltaq = (val ** (1.0 / (eta + 1.0)) - 1.0)
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                        deltaq = 1.0 - (val ** (1.0 / (eta + 1.0)))
                    x_new[j] = y + deltaq * (yh - yl)
                    x_new[j] = np.clip(x_new[j], yl, yh)
            return x_new

        # Main NSGA-II loop
        while used < budget and elapsed_ok():
            # Combine and sort
            fronts = fast_non_dominated_sort(F)

            # Selection: tournament on rank + crowding distance
            # Assign ranks
            rank = np.zeros(pop_size, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            # Crowding distance per front
            cd = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 0:
                    cd_vals = crowding_distance(F, front)
                    for k, idx in enumerate(front):
                        cd[idx] = cd_vals[k]

            # Generate offspring
            offspring_X = np.zeros((pop_size, dim))
            offspring_F = np.zeros((pop_size, n_obj))
            for i in range(0, pop_size, 2):
                if used + 2 > budget or not elapsed_ok():
                    break
                # Tournament selection for two parents
                def tournament():
                    a = rng.integers(pop_size)
                    b = rng.integers(pop_size)
                    if rank[a] < rank[b]:
                        return a
                    elif rank[b] < rank[a]:
                        return b
                    else:
                        return a if cd[a] > cd[b] else b
                p1 = tournament()
                p2 = tournament()
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                offspring_X[i] = c1
                offspring_F[i] = problem(c1)
                used += 1
                if used >= budget or not elapsed_ok():
                    break
                if i + 1 < pop_size:
                    offspring_X[i + 1] = c2
                    offspring_F[i + 1] = problem(c2)
                    used += 1

            # Combine parent and offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])

            # Select next generation
            fronts = fast_non_dominated_sort(F_comb)
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    for k, idx in enumerate(front):
                        new_X[count + k] = X_comb[idx]
                        new_F[count + k] = F_comb[idx]
                    count += len(front)
                else:
                    # Need to select from this front by crowding distance
                    cd_vals = crowding_distance(F_comb, front)
                    order = np.argsort(cd_vals)[::-1]
                    needed = pop_size - count
                    for k in range(needed):
                        idx = front[order[k]]
                        new_X[count] = X_comb[idx]
                        new_F[count] = F_comb[idx]
                        count += 1
                    break

            X = new_X
            F = new_F

        # Return the non-dominated set
        fronts = fast_non_dominated_sort(F)
        nd_indices = fronts[0]
        return X[nd_indices]
# EVOLVE-BLOCK-END

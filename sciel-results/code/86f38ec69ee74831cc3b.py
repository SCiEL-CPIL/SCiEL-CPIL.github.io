"""NSGA-II based multi-objective optimizer for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II: Non-dominated Sorting Genetic Algorithm II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 40

        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # SBX crossover and polynomial mutation parameters
        eta_c = 20.0
        eta_m = 20.0
        p_m = 1.0 / dim

        def dominates(a, b):
            """True if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_d = np.zeros(n, dtype=int)  # n_d[i] = number of individuals dominating i
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_d[i] += 1

            # First front
            front = []
            for i in range(n):
                if n_d[i] == 0:
                    front.append(i)
            if not front:
                # All dominated? Shouldn't happen, but fallback
                front = list(range(n))
            fronts.append(front)

            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front

            return fronts

        def crowding_distance(F, indices):
            """Assign crowding distance to individuals in given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            distances = np.zeros(n)
            F_sub = F[indices]

            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = np.argsort(F_sub[:, m])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 1e-14:
                    for k in range(1, n - 1):
                        distances[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / f_range

            return distances

        def sbx_crossover(x1, x2):
            """Simulated Binary Crossover."""
            c = np.zeros(dim)
            for i in range(dim):
                if np.random.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        c[i] = x1[i]
                    else:
                        u = np.random.random()
                        if u <= 0.5:
                            beta = ((1 + 1.0 / eta_c) / (1 + (2.0 * (min(x1[i], x2[i]) - lo[i]) / abs(x1[i] - x2[i])))) ** (eta_c + 1.0)
                            c[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        else:
                            beta = ((1 + 1.0 / eta_c) / (1 + (2.0 * (hi[i] - max(x1[i], x2[i])) / abs(x1[i] - x2[i])))) ** (eta_c + 1.0)
                            c[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    c[i] = x1[i]
            return np.clip(c, lo, hi)

        def polynomial_mutation(x):
            """Polynomial Mutation."""
            m = x.copy()
            for i in range(dim):
                if np.random.random() < p_m:
                    u = np.random.random()
                    delta1 = (u + 1) * (m[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (u + 1) * (hi[i] - m[i]) / (hi[i] - lo[i])
                    if u < 0.5:
                        theta = (1 + 2.0 * (delta1 - delta1 ** (eta_m + 1.0))) / (2.0 * delta1)
                    else:
                        theta = (1 + 2.0 * ((1.0 - delta2) - (1.0 - delta2) ** (eta_m + 1.0))) / (2.0 * delta2)
                    m[i] = m[i] + (hi[i] - lo[i]) * (theta - 1.0)
            return np.clip(m, lo, hi)

        def tournament_selection(X, F, size=2):
            """Binary tournament selection using non-dominated rank and crowding distance."""
            # Compute ranks and crowding distances
            fronts = non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for front_id, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = front_id

            # Crowding distances per front
            crowd = np.zeros(len(F))
            for front in fronts:
                cd = crowding_distance(F, front)
                for k, idx in enumerate(front):
                    crowd[idx] = cd[k]

            # Tournament
            best = None
            for _ in range(size):
                idx = np.random.randint(len(F))
                if best is None:
                    best = idx
                else:
                    # Prefer lower rank, then higher crowding distance
                    if rank[idx] < rank[best]:
                        best = idx
                    elif rank[idx] == rank[best] and crowd[idx] > crowd[best]:
                        best = idx
            return X[best]

        # Main evolutionary loop
        while used < budget:
            # Generate offspring population
            offspring_X = []
            offspring_F = []

            # Determine how many offspring we can generate
            # Each offspring requires 1 evaluation
            n_offspring = min(N, budget - used)

            for _ in range(n_offspring):
                # Select two parents via tournament
                p1 = tournament_selection(X, F)
                p2 = tournament_selection(X, F)

                # Crossover and mutation
                child = sbx_crossover(p1, p2)
                child = polynomial_mutation(child)

                # Evaluate
                child_f = problem(child)
                used += 1

                offspring_X.append(child)
                offspring_F.append(child_f)

                if used >= budget:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Selection: keep best N individuals
            fronts = non_dominated_sort(combined_F)
            selected_idx = []

            for front in fronts:
                if len(selected_idx) + len(front) <= N:
                    selected_idx.extend(front)
                else:
                    # Need to select from this front using crowding distance
                    remaining = N - len(selected_idx)
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    sorted_front = [front[k] for k in np.argsort(cd)[::-1]]
                    selected_idx.extend(sorted_front[:remaining])
                    break

            X = combined_X[selected_idx]
            F = combined_F[selected_idx]

        return X
# EVOLVE-BLOCK-END

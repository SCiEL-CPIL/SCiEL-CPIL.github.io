"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 2))
        # Ensure we don't exceed budget
        if pop_size > budget:
            pop_size = budget

        # Random initialization
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Check deadline
        if time.time() > deadline:
            return X

        # NSGA-II parameters
        tournament_size = 2
        crossover_prob = 0.9
        mutation_prob = 0.1
        sigma = 0.2  # Mutation step size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_dominating = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominating[i] += 1

            # First front
            front = [i for i in range(n) if n_dominating[i] == 0]
            fronts.append(front)

            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for a front."""
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)

            distances = np.zeros(len(front_indices))
            F_front = F[front_indices]

            for m in range(n_obj):
                sorted_idx = np.argsort(F_front[:, m])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf

                f_min = F_front[sorted_idx[0], m]
                f_max = F_front[sorted_idx[-1], m]

                if f_max - f_min < 1e-15:
                    continue

                for i in range(1, len(front_indices) - 1):
                    distances[sorted_idx[i]] += (
                        F_front[sorted_idx[i + 1], m] - F_front[sorted_idx[i - 1], m]
                    ) / (f_max - f_min)

            return distances

        def tournament_select(X, F, fronts, crowding, k):
            """Tournament selection."""
            candidates = rng.integers(0, len(X), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                fi = fronts.index([i for i in fronts if i == best][0]) if best in [i for f in fronts for i in f] else len(fronts)
                fc = fronts.index([i for i in fronts if i == c][0]) if c in [i for f in fronts for i in f] else len(fronts)

                # Find front index
                fi = next((idx for idx, f in enumerate(fronts) if best in f), len(fronts))
                fc = next((idx for idx, f in enumerate(fronts) if c in f), len(fronts))

                if fi < fc:
                    best = c
                elif fi == fc:
                    # Compare crowding distance
                    ci = crowding[fronts[fi].index(best)] if best in fronts[fi] else 0
                    cc = crowding[fronts[fc].index(c)] if c in fronts[fc] else 0
                    if cc > ci:
                        best = c
            return best

        def sbx_crossover(x1, x2):
            """Simulated binary crossover."""
            if rng.random() > crossover_prob:
                return x1.copy(), x2.copy()

            eta = 20.0
            x1, x2 = x1.copy(), x2.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    x1[i], x2[i] = x2[i], x1[i]

                if rng.random() < 0.5:
                    if x1[i] == x2[i]:
                        continue
                    if x1[i] > x2[i]:
                        x1[i], x2[i] = x2[i], x1[i]

                    r = rng.random()
                    if r <= 0.5:
                        beta = (2 * r) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - r))) ** (1 / (eta + 1))

                    x1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    x2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])

            return np.clip(x1, lo, hi), np.clip(x2, lo, hi)

        def polynomial_mutation(x):
            """Polynomial mutation."""
            eta = 20.0
            x = x.copy()
            for i in range(dim):
                if rng.random() < mutation_prob:
                    r = rng.random()
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (eta + 1))
                    x[i] += delta * (hi[i] - lo[i])
            return np.clip(x, lo, hi)

        # Main NSGA-II loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Crowding distance for each front
            crowding_all = np.zeros(len(F))
            for front in fronts:
                cd = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    crowding_all[idx] = cd[i]

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, budget - used)

            for _ in range(n_offspring // 2):
                # Check deadline
                if time.time() > deadline:
                    break

                # Tournament selection for two parents
                p1 = tournament_select(X, F, fronts, crowding_all, tournament_size)
                p2 = tournament_select(X, F, fronts, crowding_all, tournament_size)

                # Crossover
                c1, c2 = sbx_crossover(X[p1], X[p2])

                # Mutation
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)

                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                used += 2

                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(f1)
                offspring_F.append(f2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sorting on combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Crowding distance for each front
            combined_crowding = np.zeros(len(combined_F))
            for front in combined_fronts:
                cd = crowding_distance(combined_F, front)
                for i, idx in enumerate(front):
                    combined_crowding[idx] = cd[i]

            # Select next generation
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Sort front by crowding distance
                    front_crowding = [(combined_crowding[idx], idx) for idx in front]
                    front_crowding.sort(reverse=True)
                    needed = pop_size - len(new_X)
                    for _, idx in front_crowding[:needed]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END

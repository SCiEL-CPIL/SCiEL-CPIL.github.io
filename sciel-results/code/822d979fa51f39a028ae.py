"""Multi-objective optimizer for WFG4 using NSGA-II-like approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-like multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim

        # Population size
        pop_size = min(100, max(50, self.budget // 4))
        # Number of evaluations used
        evals = 0

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        evals += pop_size

        # Track all evaluated points for hypervolume
        all_F = [F.copy()]

        def dominates(a, b):
            """Check if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Perform non-dominated sorting. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_count = np.zeros(n, dtype=int)  # number of individuals dominating i
            rank = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_count[i] += 1

            for i in range(n):
                if n_count[i] == 0:
                    rank[i] = 0
                    fronts.append([i])

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)

            n = len(indices)
            cd = np.zeros(n)
            m = len(F[indices[0]])

            for k in range(m):
                f_vals = F[indices, k]
                f_min = f_vals.min()
                f_max = f_vals.max()
                if f_max - f_min < 1e-12:
                    continue
                sorted_idx = np.argsort(f_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (f_vals[sorted_idx[i + 1]] - f_vals[sorted_idx[i - 1]]) / (f_max - f_min)

            return cd

        def tournament_select(fronts, rank, F, indices, k=2):
            """Tournament selection based on rank and crowding distance."""
            candidates = [indices[rng.integers(len(indices))] for _ in range(k)]
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
                elif rank[c] == rank[best]:
                    # Use crowding distance
                    cd_vals = crowding_distance(F, [c, best])
                    if cd_vals[0] > cd_vals[1]:
                        best = c
            return best

        def crossover(x1, x2):
            """SBX crossover."""
            eta = 20
            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            return c1, c2

        def mutate(x, sigma=0.1):
            """Polynomial mutation."""
            eta = 20
            x_new = x.copy()
            for i in range(len(x)):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    if u <= 0.5:
                        delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x_new[i] += sigma * delta_q
            return np.clip(x_new, lo, hi)

        # Main loop
        while evals < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)
            rank = np.zeros(pop_size, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size

            while len(offspring_X) < n_offspring:
                if time.time() > deadline:
                    break

                # Selection
                p1 = tournament_select(fronts, rank, F, list(range(pop_size)))
                p2 = tournament_select(fronts, rank, F, list(range(pop_size)))

                # Crossover and mutation
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutate(c1)
                c2 = mutate(c2)

                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                evals += 1

                if len(offspring_X) < n_offspring and evals < self.budget:
                    offspring_X.append(c2)
                    offspring_F.append(problem(c2))
                    evals += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            all_F.append(offspring_F)

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            # Non-dominated sort on combined
            fronts = non_dominated_sort(combined_F)
            rank = np.zeros(n_combined, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to trim this front
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(-cd)
                    for i in range(min(remaining, len(front))):
                        idx = front[sorted_by_cd[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        # Return all evaluated objective vectors for hypervolume calculation
        all_F_combined = np.vstack(all_F)
        return all_F_combined
# EVOLVE-BLOCK-END

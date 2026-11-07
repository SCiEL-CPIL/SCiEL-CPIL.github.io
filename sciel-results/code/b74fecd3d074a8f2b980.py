# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective NSGA-II style optimizer for WFG4."""

    def __init__(self, dim, n_obj, budget):
        self.dim = dim
        self.n_obj = n_obj
        self.budget = budget

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        mid = (lo + hi) / 2.0
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(40, max(10, budget // 10))
        # Archive size (elite)
        archive_size = pop_size

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[i].append(j)
                    elif np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts.append([i])
            k = 0
            while k < len(fronts):
                current_front = fronts[k]
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for points at given indices in F."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n, dtype=np.float64)
            F_sub = F[indices]
            for m in range(n_obj):
                vals = F_sub[:, m]
                idx_sorted = np.argsort(vals)
                cd[idx_sorted[0]] = np.inf
                cd[idx_sorted[-1]] = np.inf
                min_val = vals[idx_sorted[0]]
                max_val = vals[idx_sorted[-1]]
                if max_val - min_val < 1e-15:
                    continue
                for i in range(1, n - 1):
                    cd[idx_sorted[i]] += (vals[idx_sorted[i + 1]] - vals[idx_sorted[i - 1]]) / (max_val - min_val)
            return cd

        def tournament_select(fronts, F, X, n_select):
            """Select n_select parents using binary tournament on fronts then crowding."""
            # Assign front rank
            rank = np.zeros(len(F), dtype=int)
            for fi, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = fi

            # Compute crowding distances per front
            cd_all = np.zeros(len(F), dtype=np.float64)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd_all[idx] = cd[i]

            selected = []
            for _ in range(n_select):
                i = rng.integers(len(F))
                j = rng.integers(len(F))
                if rank[i] < rank[j]:
                    winner = i
                elif rank[j] < rank[i]:
                    winner = j
                else:
                    winner = i if cd_all[i] > cd_all[j] else j
                selected.append(winner)
            return selected

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            fronts = fast_non_dominated_sort(F)

            # Generate offspring
            n_offspring = pop_size
            parents = tournament_select(fronts, F, X, n_offspring * 2)
            X_off = np.empty((n_offspring, dim))
            for i in range(n_offspring):
                p1 = parents[2 * i]
                p2 = parents[2 * i + 1]
                # Simulated binary crossover
                if rng.random() < 0.9:
                    # SBX
                    eta = 20.0
                    child1 = np.empty(dim)
                    child2 = np.empty(dim)
                    for d in range(dim):
                        x1, x2 = X[p1, d], X[p2, d]
                        if abs(x1 - x2) < 1e-15:
                            child1[d] = x1
                            child2[d] = x2
                        else:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta + 1))
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                            child1[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                            child2[d] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                    X_off[i] = child1
                    X_off[i] = np.clip(X_off[i], lo, hi)
                else:
                    X_off[i] = X[p1].copy()

            # Mutation
            mutation_rate = 1.0 / dim
            sigma = 0.1
            for i in range(n_offspring):
                for d in range(dim):
                    if rng.random() < mutation_rate:
                        X_off[i, d] += sigma * (hi[d] - lo[d]) * rng.standard_normal()
                        X_off[i, d] = np.clip(X_off[i, d], lo[d], hi[d])

            # Evaluate offspring
            F_off = np.array([problem(x) for x in X_off], dtype=np.float64)
            used += n_offspring

            # Combine parent and offspring
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            n_all = len(X_all)

            # Non-dominated sort on combined
            fronts_all = fast_non_dominated_sort(F_all)

            # Select next population
            X_new = np.empty((0, dim))
            F_new = np.empty((0, n_obj))
            for front in fronts_all:
                if len(X_new) + len(front) <= pop_size:
                    idx = front
                    X_new = np.vstack([X_new, X_all[idx]]) if len(X_new) > 0 else X_all[idx]
                    F_new = np.vstack([F_new, F_all[idx]]) if len(F_new) > 0 else F_all[idx]
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(X_new)
                    cd = crowding_distance(F_all, front)
                    sort_idx = np.argsort(cd)[::-1]
                    selected = [front[j] for j in sort_idx[:remaining]]
                    X_new = np.vstack([X_new, X_all[selected]]) if len(X_new) > 0 else X_all[selected]
                    F_new = np.vstack([F_new, F_all[selected]]) if len(F_new) > 0 else F_all[selected]
                    break

            X = X_new
            F = F_new

        return X
# EVOLVE-BLOCK-END

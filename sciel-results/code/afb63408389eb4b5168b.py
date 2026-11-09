"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a lightweight NSGA-II variant with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(30, budget // 2))

        rng = np.random.default_rng(42)

        # Initial random population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts.append(i)
            k = 0
            while fronts:
                next_front = []
                for i in fronts:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                fronts = next_front
                k += 1
            return S, nd, fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            m = len(indices)
            cd = np.zeros(m)
            for k in range(n_obj):
                vals = F[indices, k]
                min_v, max_v = vals.min(), vals.max()
                rng_v = max_v - min_v
                if rng_v < 1e-12:
                    continue
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, m - 1):
                    cd[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / rng_v
            return cd

        # Main loop
        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Non-dominated sort
            S, nd, fronts = non_dominated_sort(F)

            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            front_idx = 0
            current_front = fronts
            while current_front:
                for i in current_front:
                    rank[i] = front_idx
                # Build next front
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                front_idx += 1
                current_front = next_front

            # Crowding distance per front
            cd = np.zeros(len(F))
            current_front = fronts
            while current_front:
                cd[current_front] = crowding_distance(F, current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        if nd[j] == 0:
                            next_front.append(j)
                current_front = next_front

            # Tournament selection
            def tournament_select():
                a = rng.integers(len(F))
                b = rng.integers(len(F))
                if rank[a] < rank[b]:
                    return a
                elif rank[b] < rank[a]:
                    return b
                else:
                    return a if cd[a] > cd[b] else b

            # Generate children
            n_children = min(pop_size, budget - used)
            children_X = np.zeros((n_children, dim))
            for i in range(n_children):
                p1 = tournament_select()
                p2 = tournament_select()
                # Simulated binary crossover
                child = X[p1].copy()
                if rng.random() < 0.9:
                    eta = 20.0
                    for d in range(dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 1 / (eta + 1):
                                beta = (2 * u) ** (1 / (eta + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                            child[d] = 0.5 * (X[p1][d] + X[p2][d] + beta * (X[p2][d] - X[p1][d]))
                        else:
                            u = rng.random()
                            if u <= 1 / (eta + 1):
                                beta = (2 * u) ** (1 / (eta + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                            child[d] = 0.5 * (X[p1][d] + X[p2][d] + beta * (X[p1][d] - X[p2][d]))
                    child = np.clip(child, lo, hi)
                # Polynomial mutation
                if rng.random() < 0.9:
                    eta_m = 20.0
                    for d in range(dim):
                        if rng.random() < 1.0 / dim:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1 / (eta_m + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                            child[d] += delta_q * (hi[d] - lo[d])
                    child = np.clip(child, lo, hi)
                children_X[i] = child

            # Evaluate children
            children_F = np.array([problem(x) for x in children_X], dtype=float)
            used += n_children

            # Combine and select
            X_all = np.vstack([X, children_X])
            F_all = np.vstack([F, children_F])

            S_all, nd_all, fronts_all = non_dominated_sort(F_all)

            rank_all = np.zeros(len(F_all), dtype=int)
            front_idx = 0
            current_front = fronts_all
            while current_front:
                for i in current_front:
                    rank_all[i] = front_idx
                next_front = []
                for i in current_front:
                    for j in S_all[i]:
                        nd_all[j] -= 1
                        if nd_all[j] == 0:
                            next_front.append(j)
                front_idx += 1
                current_front = next_front

            cd_all = np.zeros(len(F_all))
            current_front = fronts_all
            while current_front:
                cd_all[current_front] = crowding_distance(F_all, current_front)
                next_front = []
                for i in current_front:
                    for j in S_all[i]:
                        if nd_all[j] == 0:
                            next_front.append(j)
                current_front = next_front

            # Select best pop_size
            order = np.lexsort((cd_all, rank_all))
            order = order[:pop_size]
            X = X_all[order]
            F = F_all[order]

        return X
# EVOLVE-BLOCK-END

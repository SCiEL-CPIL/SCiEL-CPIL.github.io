"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer for RE37 (4 vars, 3 objs)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a reasonable fraction of budget
        # For budget=200, dim=4, n_obj=3, a pop of 30-50 works well
        pop_size = min(50, max(20, budget // 4))

        # Initial population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination sets
            n_dom = [0] * n             # number of solutions dominated by i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for k in fronts[i]:
                    for m in S[k]:
                        n_dom[m] -= 1
                        if n_dom[m] == 0:
                            next_front.append(m)
                i += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for solutions in indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                min_v, max_v = vals.min(), vals.max()
                rng_v = max_v - min_v
                if rng_v < 1e-15:
                    cd[:] = float('inf')
                    continue
                order = np.argsort(vals)
                cd[order[0]] = float('inf')
                cd[order[-1]] = float('inf')
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_v
            return cd

        def tournament_select(F, X, k=2):
            """Tournament selection based on non-domination rank + crowding."""
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            # Crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cds = crowding_distance(F, front)
                    for j, idx in enumerate(front):
                        cd[idx] = cds[j]

            a = rng.integers(0, len(F))
            b = rng.integers(0, len(F))
            if rank[a] < rank[b]:
                return a
            elif rank[b] < rank[a]:
                return b
            else:
                return a if cd[a] > cd[b] else b

        while used < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Create offspring
            offspring_X = np.zeros((pop_size, dim))
            for i in range(pop_size):
                p1 = tournament_select(F, X)
                p2 = tournament_select(F, X)
                # SBX crossover
                if rng.random() < 0.9:  # crossover probability
                    eta = 20.0
                    child = np.zeros(dim)
                    for d in range(dim):
                        if X[p1, d] != X[p2, d]:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta + 1))
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                            child[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                        else:
                            child[d] = X[p1, d]
                    # Polynomial mutation
                    eta_m = 20.0
                    for d in range(dim):
                        if rng.random() < 1.0 / dim:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            child[d] += delta_q * (hi[d] - lo[d])
                else:
                    child = X[p1].copy()
                offspring_X[i] = np.clip(child, lo, hi)

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=float)
            used += pop_size

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Selection: keep best pop_size from combined
            fronts = fast_non_dominated_sort(combined_F)
            selected_idx = []
            for front in fronts:
                if len(selected_idx) + len(front) <= pop_size:
                    selected_idx.extend(front)
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(selected_idx)
                    if remaining > 0:
                        cds = crowding_distance(combined_F, front)
                        order = np.argsort(-cds)
                        selected_idx.extend([front[i] for i in order[:remaining]])
                    break

            X = combined_X[selected_idx]
            F = combined_F[selected_idx]

        # Return the Pareto front (non-dominated points)
        fronts = fast_non_dominated_sort(F)
        if fronts:
            best_idx = fronts[0]
            return X[best_idx]
        return X

# EVOLVE-BLOCK-END

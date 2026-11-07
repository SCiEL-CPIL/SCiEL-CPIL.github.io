"""Multi-objective optimizer for WFG7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary search with NSGA-II style selection and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Initial population
        pop_size = min(30, self.budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Record all evaluations (the evaluator records every objective vector returned by problem())
        # The task says "Every returned objective vector is recorded" - so we just need to call problem()
        # and the evaluator tracks it. We don't need to store them separately for scoring.

        # NSGA-II style: crowding distance + fast non-dominated sort
        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination count
            n_dom = [0] * n  # number of points dominated
            fronts = [[]]

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    # p dominates q?
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        n_dom[p] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        S[q].append(p)
                        n_dom[q] += 1
                if n_dom[p] == 0:
                    fronts[0].append(p)

            i = 0
            while len(fronts[i]) > 0:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            return fronts[:-1]  # remove empty last

        def crowding_distance(F, front_indices):
            """Compute crowding distance for points in front_indices."""
            m = len(front_indices)
            if m <= 2:
                return {i: float('inf') for i in front_indices}
            dist = {i: 0.0 for i in front_indices}
            for k in range(self.n_obj):
                sorted_local = sorted(front_indices, key=lambda i: F[i][k])
                dist[sorted_local[0]] = float('inf')
                dist[sorted_local[-1]] = float('inf')
                f_min = F[sorted_local[0]][k]
                f_max = F[sorted_local[-1]][k]
                range_f = f_max - f_min
                if range_f == 0:
                    continue
                for i in range(1, m - 1):
                    dist[sorted_local[i]] += (F[sorted_local[i + 1]][k] - F[sorted_local[i - 1]][k]) / range_f
            return dist

        def select_parents(F, X):
            """NSGA-II selection: non-dominated sorting + crowding distance."""
            fronts = fast_non_dominated_sort(F)
            n = len(F)
            # Assign ranks
            rank = np.zeros(n, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Sort by rank then crowding distance (descending)
            # Compute crowding distance per front
            cd = np.zeros(n)
            for front in fronts:
                cd_front = crowding_distance(F, front)
                for idx in front:
                    cd[idx] = cd_front[idx]

            # Sort indices: primary key rank (ascending), secondary key cd (descending)
            order = sorted(range(n), key=lambda i: (rank[i], -cd[i]))
            # Return top 2 parents
            parents = []
            for idx in order[:2]:
                parents.append(X[idx])
            return parents

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Select parents
            parents = select_parents(F, X)
            p1, p2 = parents[0], parents[1]

            # SBX crossover
            if rng.random() < 0.9:
                # Simulated Binary Crossover
                eta_c = 20.0
                child1 = np.empty(self.dim)
                child2 = np.empty(self.dim)
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        child1[d] = p1[d]
                        child2[d] = p2[d]
                    else:
                        x1, x2 = p1[d], p2[d]
                        if abs(x2 - x1) < 1e-14:
                            child1[d] = x1
                            child2[d] = x2
                        else:
                            r = rng.random()
                            beta = 1.0
                            if r <= 0.5:
                                beta = (2 * r) ** (1 / (eta_c + 1))
                            else:
                                beta = (1 / (2 * (1 - r))) ** (1 / (eta_c + 1))
                            c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                            c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                            child1[d] = c1
                            child2[d] = c2
                child = child1
            else:
                child = p1.copy()

            # Mutation
            eta_m = 20.0
            for d in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    x = child[d]
                    r = rng.random()
                    delta = 1.0
                    if r <= 0.5:
                        delta = (2 * r) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (eta_m + 1))
                    child[d] = x * (1 + delta)
                    # Scale to bounds
                    child[d] = np.clip(child[d], lo[d], hi[d])

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point (highest sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END

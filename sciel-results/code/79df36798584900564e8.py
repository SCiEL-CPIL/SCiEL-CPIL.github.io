import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size: use a reasonable fraction of budget
        pop_size = min(40, max(10, self.budget // 5))
        remaining = self.budget

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        remaining -= pop_size

        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        def fast_non_dominated_sort(F):
            """Return list of fronts (each front is list of indices)."""
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
                        nd[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[j] += 1
                if nd[i] == 0:
                    fronts.append([i])
            k = 0
            while fronts[k]:
                Q = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            Q.append(j)
                k += 1
                if Q:
                    fronts.append(Q)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_v - min_v)
            return cd

        while remaining > 0:
            # Check deadline
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Assign crowding distances per front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd[front] = crowding_distance(F, front)

            # Select parents using tournament on (front_rank, crowding)
            front_rank = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    front_rank[idx] = rank

            def tournament_select():
                i, j = rng.integers(len(F)), rng.integers(len(F))
                if front_rank[i] < front_rank[j]:
                    return i
                elif front_rank[j] < front_rank[i]:
                    return j
                else:
                    return i if cd[i] > cd[j] else j

            # Generate offspring
            n_off = min(pop_size, remaining)
            off_X = np.empty((n_off, self.dim))
            for i in range(n_off):
                p1 = tournament_select()
                p2 = tournament_select()
                # Simulated binary crossover (SBX)
                if rng.random() < 0.9:
                    eta = 20
                    alpha = rng.random()
                    if alpha <= 0.5:
                        beta = (2 * alpha) ** (1 / (eta + 1))
                    else:
                        beta = (2 * (1 - alpha)) ** (1 / (eta + 1))
                    beta = 1.0 / beta
                    child1 = 0.5 * ((1 + beta) * X[p1] + (1 - beta) * X[p2])
                    child2 = 0.5 * ((1 - beta) * X[p1] + (1 + beta) * X[p2])
                else:
                    child1 = X[p1].copy()
                    child2 = X[p2].copy()

                # Polynomial mutation
                eta_m = 20
                for d in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        delta1 = (2 * rng.random()) ** (1 / (eta_m + 1)) - 1
                        delta2 = 1 - (2 * (1 - rng.random())) ** (1 / (eta_m + 1))
                        span = hi[d] - lo[d]
                        child1[d] += delta1 * span
                        child2[d] += delta2 * span

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                if i * 2 + 1 < n_off:
                    off_X[i * 2] = child1
                    off_X[i * 2 + 1] = child2
                else:
                    off_X[i] = child1

            # Evaluate offspring
            off_F = np.array([problem(x) for x in off_X])
            remaining -= n_off

            # Combine parent and offspring
            X_comb = np.vstack([X, off_X])
            F_comb = np.vstack([F, off_F])

            # Select next generation
            fronts_comb = fast_non_dominated_sort(F_comb)
            next_X = []
            next_F = []
            for front in fronts_comb:
                if len(next_X) + len(front) <= pop_size:
                    next_X.extend(front)
                    next_F.extend(front)
                else:
                    # Sort by crowding distance and take top
                    cd_front = crowding_distance(F_comb, front)
                    order = np.argsort(cd_front)[::-1]
                    need = pop_size - len(next_X)
                    for idx in order[:need]:
                        next_X.append(front[idx])
                    break

            X = X_comb[next_X]
            F = F_comb[next_F]

            # Record for hypervolume
            all_X.append(X.copy())
            all_F.append(F.copy())

        # Return the best set (last population) - but the evaluator records ALL evaluated points
        # Actually, the evaluator records every objective vector returned by problem()
        # So we just need to make good evaluations
        return X

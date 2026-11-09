"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size: balance diversity and per-generation cost
        N = 30
        if self.budget < 60:
            N = max(10, self.budget // 4)

        # Initial population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        # NSGA-II parameters
        eta = 0.5  # mutation strength
        crossover_prob = 0.9

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            ndom = [0] * n  # number of solutions that dominate i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        ndom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        ndom[i] += 1

            for i in range(n):
                if ndom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        ndom[j] -= 1
                        if ndom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)

            return fronts[:-1] if fronts[-1] == [] else fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            m = self.n_obj

            for k in range(m):
                sorted_idx = np.argsort(F[front_indices, k])
                cd_sorted = np.zeros(n)
                cd_sorted[0] = np.inf
                cd_sorted[-1] = np.inf
                f_min = F[front_indices[sorted_idx[0]], k]
                f_max = F[front_indices[sorted_idx[-1]], k]
                if f_max - f_min > 1e-12:
                    for i in range(1, n - 1):
                        cd_sorted[i] += (F[front_indices[sorted_idx[i + 1]], k] -
                                          F[front_indices[sorted_idx[i - 1]], k]) / (f_max - f_min)
                else:
                    cd_sorted[1:-1] = 0
                # Map back to original indices
                for i, idx in enumerate(sorted_idx):
                    cd[front_indices.index(idx)] = cd_sorted[i] if idx in front_indices else 0

            # Better approach: work with local indices
            cd = np.zeros(n)
            for k in range(m):
                vals = F[front_indices, k]
                order = np.argsort(vals)
                cd_local = np.zeros(n)
                cd_local[0] = np.inf
                cd_local[-1] = np.inf
                vmin, vmax = vals[order[0]], vals[order[-1]]
                if vmax - vmin > 1e-12:
                    for i in range(1, n - 1):
                        cd_local[i] = (vals[order[i + 1]] - vals[order[i - 1]]) / (vmax - vmin)
                # Map back
                for i, orig_idx in enumerate(order):
                    cd[orig_idx] += cd_local[i]

            return cd

        def select_parents(pop_F):
            """Binary tournament selection based on non-dominated rank and crowding distance."""
            fronts = fast_non_dominated_sort(pop_F)
            rank = np.zeros(len(pop_F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Crowding distance per front
            cd = np.zeros(len(pop_F))
            for front in fronts:
                if len(front) > 2:
                    front_cd = crowding_distance(pop_F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = front_cd[i]
                else:
                    for idx in front:
                        cd[idx] = np.inf

            def dominates(a, b):
                """Returns True if a is preferred to b (lower rank, or same rank and higher cd)."""
                if rank[a] < rank[b]:
                    return True
                if rank[a] > rank[b]:
                    return False
                return cd[a] > cd[b]

            # Binary tournament
            i = rng.integers(len(pop_F))
            j = rng.integers(len(pop_F))
            if dominates(i, j):
                return i
            else:
                return j

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring_X = []
            offspring_F = []
            for _ in range(N):
                p1 = select_parents(F)
                p2 = select_parents(F)

                # Simulated binary crossover (SBX)
                if rng.random() < crossover_prob and p1 != p2:
                    # Blend
                    beta = rng.random()
                    child = beta * X[p1] + (1 - beta) * X[p2]
                else:
                    child = X[p1].copy()

                # Polynomial mutation
                if rng.random() < 0.9:
                    mutation = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            delta = rng.random()
                            if delta < 0.5:
                                mut_val = 2 * delta
                            else:
                                mut_val = 2 * (1 - delta)
                            sign = 1 if rng.random() < 0.5 else -1
                            mutation[d] = sign * (hi[d] - lo[d]) * 0.1 * mut_val
                    child = child + mutation

                child = np.clip(child, lo, hi)
                offspring_X.append(child)
                offspring_F.append(problem(child))
                used += 1

                if time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)

            # Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            next_indices = []
            for front in fronts:
                if len(next_indices) + len(front) <= N:
                    next_indices.extend(front)
                else:
                    remaining = N - len(next_indices)
                    # Sort front by crowding distance
                    front_cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda idx: front_cd[front.index(idx)], reverse=True)
                    next_indices.extend(sorted_front[:remaining])
                    break

            X = combined_X[next_indices]
            F = combined_F[next_indices]

        return X
# EVOLVE-BLOCK-END

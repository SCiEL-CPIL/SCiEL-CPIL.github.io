# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9: small population with deep
    CMA-ES-style local search per individual per generation.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random.RandomState(42)

        pop_size = 10
        max_gen = self.budget // pop_size
        if max_gen < 1:
            pop_size = max(1, self.budget)
            max_gen = 1

        # Stratified initialization
        X = np.zeros((pop_size, self.dim))
        for j in range(self.dim):
            X[:, j] = lo[j] + span[j] * (np.arange(pop_size) + rng.uniform(size=pop_size)) / pop_size
        X = np.clip(X, lo, hi)

        used = 0
        F = np.zeros((pop_size, self.n_obj))
        for i in range(pop_size):
            F[i] = problem(X[i])
            used += 1

        all_points = [F[i].copy() for i in range(pop_size)]

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dom[i] += 1
            front = [i for i in range(n) if n_dom[i] == 0]
            for i in front:
                rank[i] = 0
            fronts = []
            cur = 0
            while front:
                fronts.append(front)
                nf = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = cur + 1
                            nf.append(j)
                front = nf
                cur += 1
            return fronts, rank

        def crowding(F, idxs):
            n = len(idxs)
            if n <= 2:
                return np.full(n, np.inf)
            d = np.zeros(n)
            Fs = F[idxs]
            for m in range(self.n_obj):
                s = np.argsort(Fs[:, m])
                d[s[0]] = np.inf
                d[s[-1]] = np.inf
                rng_m = Fs[s[-1], m] - Fs[s[0], m]
                if rng_m < 1e-15:
                    continue
                for i in range(1, n - 1):
                    d[s[i]] += (Fs[s[i+1], m] - Fs[s[i-1], m]) / rng_m
            return d

        def cma_local(x, Fx, n_evals, rng):
            """Simple CMA-like local search."""
            best_x = x.copy()
            best_F = Fx.copy()
            sigma = 0.15
            for _ in range(n_evals):
                if time.time() > deadline - 5:
                    break
                # Generate perturbation
                z = rng.randn(self.dim)
                new_x = np.clip(best_x + sigma * z, lo, hi)
                new_F = problem(new_x)
                all_points.append(new_F.copy())
                if dominates(new_F, best_F):
                    best_x = new_x
                    best_F = new_F
                    sigma *= 1.1  # expand on improvement
                else:
                    sigma *= 0.85  # shrink on failure
            return best_x, best_F

        for gen in range(max_gen):
            if used >= self.budget or time.time() > deadline - 10:
                break

            fronts, ranks = non_dominated_sort(F)

            offspring_X = np.zeros((pop_size, self.dim))
            offspring_F = np.zeros((pop_size, self.n_obj))

            for i in range(pop_size):
                if used >= self.budget or time.time() > deadline - 10:
                    break

                # Crossover for genetic diversity
                p1 = rng.randint(pop_size)
                p2 = rng.randint(pop_size)
                if ranks[p1] != ranks[p2]:
                    better = p1 if ranks[p1] < ranks[p2] else p2
                    worse = p2 if ranks[p1] < ranks[p2] else p1
                    # 70% use better as primary
                    if rng.random() < 0.7:
                        p1, p2 = better, worse
                    else:
                        p1, p2 = worse, better

                child = np.zeros(self.dim)
                for j in range(self.dim):
                    if rng.random() < 0.8:
                        x1, x2 = X[p1, j], X[p2, j]
                        if abs(x1 - x2) < 1e-15:
                            child[j] = x1
                        else:
                            u = rng.random()
                            beta = 1.0 + 2.0 * min(abs(x1), abs(x2)) / max(abs(x1 - x2), 1e-15)
                            if u <= 1.0 / (beta + 1.0):
                                alpha = (2.0 * u) ** (1.0 / (beta + 1.0))
                            else:
                                alpha = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (beta + 1.0))
                            child[j] = 0.5 * ((1.0 + alpha) * x1 + (1.0 - alpha) * x2)
                    else:
                        child[j] = X[p1, j] if rng.random() < 0.5 else X[p2, j]

                # Small mutation
                for j in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        child[j] += rng.randn() * 0.08 * span[j]

                child = np.clip(child, lo, hi)
                offspring_X[i] = child
                cf = problem(child)
                used += 1
                all_points.append(cf.copy())

                # Deep local search: 2 additional evaluations per offspring
                nx, nF = cma_local(child, cf, 2, rng)
                used += 2
                offspring_X[i] = nx
                offspring_F[i] = nF

            # Combine and select next generation
            comb_X = np.vstack([X, offspring_X])
            comb_F = np.vstack([F, offspring_F])

            c_fronts, c_ranks = non_dominated_sort(comb_F)

            next_X = np.zeros((pop_size, self.dim))
            next_F = np.zeros((pop_size, self.n_obj))
            next_idx = []

            for front in c_fronts:
                if len(next_idx) + len(front) <= pop_size:
                    next_idx.extend(front)
                else:
                    rem = pop_size - len(next_idx)
                    cd = crowding(comb_F, front)
                    srt = np.argsort(cd)[::-1]
                    for idx in srt[:rem]:
                        next_idx.append(front[idx])
                    break

            for i, idx in enumerate(next_idx):
                next_X[i] = comb_X[idx]
                next_F[i] = comb_F[idx]

            X = next_X
            F = next_F

        return np.array(all_points)
# EVOLVE-BLOCK-END

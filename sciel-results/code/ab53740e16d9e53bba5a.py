"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random.default_rng(0)

        # Population size
        pop_size = 50
        n_gen = max(1, (self.budget - pop_size) // pop_size)

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II helpers
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            return fronts[:-1] if fronts[-1] == [] else fronts

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.ones(n)
            cd = np.zeros(n)
            m = self.n_obj
            for k in range(m):
                order = np.argsort(F[front_indices, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = F[front_indices, k].max() - F[front_indices, k].min()
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    cd[order[i]] += (F[front_indices[order[i + 1]], k] - F[front_indices[order[i - 1]], k]) / f_range
            return cd

        def tournament_select(fronts, cd_cache):
            idx1 = rng.integers(len(fronts))
            idx2 = rng.integers(len(fronts))
            f1, f2 = fronts[idx1], fronts[idx2]
            if len(f1) != len(f2):
                return f1 if len(f1) < len(f2) else f2
            cd1 = cd_cache[f1]
            cd2 = cd_cache[f2]
            return f1 if cd1[rng.integers(len(f1))] > cd2[rng.integers(len(f2))] else f2

        def crossover(x1, x2):
            # Simulated binary crossover
            nchild = 2
            children = np.empty((nchild, self.dim))
            for c in range(nchild):
                child = np.empty(self.dim)
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        child[d] = x1[d]
                    else:
                        child[d] = x2[d]
                children[c] = child
            return children

        def mutate(x, sigma):
            return x + sigma * span * rng.standard_normal(self.dim)

        # Main loop
        while used < self.budget:
            if time.time() - start_time > deadline - 5:
                break

            # Evaluate all in population
            fronts = fast_non_dominated_sort(F)
            cd_cache = {}
            for front in fronts:
                cd_cache[front] = crowding_distance(F, front)

            # Generate offspring
            offspring = []
            offspring_F = []
            while len(offspring) < pop_size and used < self.budget:
                p1 = tournament_select(fronts, cd_cache)
                p2 = tournament_select(fronts, cd_cache)
                children = crossover(X[p1], X[p2])
                sigma = 0.1
                for c in children:
                    child = mutate(c, sigma)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    offspring.append(child)
                    offspring_F.append(fc)
                    if used >= self.budget:
                        break
                    if time.time() - start_time > deadline - 5:
                        break

            if not offspring:
                break

            offspring_F = np.array(offspring_F)
            offspring = np.array(offspring)

            # Combine parents and offspring
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, offspring_F])

            fronts_all = fast_non_dominated_sort(F_all)
            cd_cache_all = {}
            for front in fronts_all:
                cd_cache_all[front] = crowding_distance(F_all, front)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd = cd_cache_all[front]
                    order = np.argsort(-cd)
                    for idx in order[:remaining]:
                        new_X.append(X_all[front[idx]])
                        new_F.append(F_all[front[idx]])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END

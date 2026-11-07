# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(40, max(20, budget // 5))

        # Initial random population
        X = lo + span * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Reference point for hypervolume (not used directly in NSGA-II, but for reference)
        # We'll use non-dominated sorting + crowding distance

        def fast_non_dominated_sort(F):
            """Return list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            nd = [0] * n  # nd[i] = number of solutions dominating i
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[j] += 1
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
                if next_front:
                    fronts.append(next_front)
                else:
                    break

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for indices in F."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n

            cd = np.zeros(n)
            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = sorted(range(n), key=lambda i: F[indices[i]][m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                f_min = F[indices[sorted_idx[0]]][m]
                f_max = F[indices[sorted_idx[-1]]][m]
                if f_max - f_min > 1e-12:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F[indices[sorted_idx[i+1]]][m] - F[indices[sorted_idx[i-1]]][m]) / (f_max - f_min)
            return cd

        def select_parents(F, X):
            """Binary tournament selection based on rank and crowding distance."""
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Crowding distance
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 2:
                    cd_front = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cd_front[i]
                else:
                    for idx in front:
                        cd[idx] = float('inf')

            def dominates(a, b):
                if rank[a] < rank[b]:
                    return True
                if rank[a] > rank[b]:
                    return False
                return cd[a] > cd[b]

            # Binary tournament
            i1 = np.random.randint(len(F))
            i2 = np.random.randint(len(F))
            if dominates(i1, i2):
                return i1
            else:
                return i2

        def crossover(x1, x2):
            """SBX crossover."""
            if np.random.random() < 0.9:
                beta = 20
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for i in range(dim):
                    if np.random.random() < 0.5:
                        u = np.random.random()
                        if u <= 1.0 / (beta + 1):
                            alpha = (2 * u) ** (1.0 / (beta + 1))
                        else:
                            alpha = (1.0 / (2 * (1 - u))) ** (1.0 / (beta + 1))
                        child1[i] = 0.5 * ((1 + alpha) * x1[i] + (1 - alpha) * x2[i])
                        child2[i] = 0.5 * ((1 - alpha) * x1[i] + (1 + alpha) * x2[i])
                    else:
                        u = np.random.random()
                        if u <= 1.0 / (beta + 1):
                            alpha = (2 * u) ** (1.0 / (beta + 1))
                        else:
                            alpha = (1.0 / (2 * (1 - u))) ** (1.0 / (beta + 1))
                        child1[i] = 0.5 * ((1 - alpha) * x1[i] + (1 + alpha) * x2[i])
                        child2[i] = 0.5 * ((1 + alpha) * x1[i] + (1 - alpha) * x2[i])
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def mutate(x, rate=0.1, sigma=0.2):
            """Polynomial mutation."""
            x_new = x.copy()
            for i in range(dim):
                if np.random.random() < rate:
                    u = np.random.random()
                    if u < 0.5:
                        xy = (2 * u) ** (1.0 / (1 + 20)) - 1
                    else:
                        xy = 1 - (2 * (1 - u)) ** (1.0 / (1 + 20))
                    x_new[i] = x[i] + sigma * xy * span[i]
                    x_new[i] = np.clip(x_new[i], lo[i], hi[i])
            return x_new

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size

            for _ in range(n_offspring // 2):
                p1 = select_parents(F, X)
                p2 = select_parents(F, X)
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(problem(c1))
                offspring_F.append(problem(c2))

            used += n_offspring
            offspring_F = np.array(offspring_F)
            offspring_X = np.array(offspring_X)

            # Combine parent and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])

            # Non-dominated sort and crowding distance
            fronts = fast_non_dominated_sort(F_combined)
            rank = np.zeros(len(F_combined), dtype=int)
            cd = np.zeros(len(F_combined))
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
                if len(front) > 2:
                    cd_front = crowding_distance(F_combined, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cd_front[i]
                else:
                    for idx in front:
                        cd[idx] = float('inf')

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    remaining = pop_size - len(new_X)
                    if remaining > 0:
                        # Sort front by crowding distance
                        front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                        for idx in front_sorted[:remaining]:
                            new_X.append(X_combined[idx])
                            new_F.append(F_combined[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END

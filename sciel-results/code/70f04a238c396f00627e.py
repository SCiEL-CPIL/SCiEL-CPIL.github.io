"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280  # 300s limit, keep 20s safety margin
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: use a reasonable fraction of budget
        pop_size = min(100, max(20, self.budget // 2))
        # Ensure we can do at least a few generations
        if pop_size > self.budget:
            pop_size = max(10, self.budget // 2)

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume consideration (from task)
        ref = np.array([3.0, 5.0, 7.0])

        def dominates(a, b):
            """Check if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_count = np.zeros(n, dtype=int)  # n[i] = number of points that dominate i
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1

            front = []
            for i in range(n):
                if n_count[i] == 0:
                    front.append(i)

            front_idx = 0
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                front = next_front
                front_idx += 1

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)

            dist = np.zeros(len(indices))
            F_sub = F[indices]

            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, len(indices) - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / f_range

            return dist

        def select_parents(X, F, pop_size):
            """Binary tournament selection based on non-dominated rank and crowding distance."""
            fronts = fast_non_dominated_sort(F)

            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for front_idx, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = front_idx

            # Compute crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                cd_sub = crowding_distance(F, front)
                for k, idx in enumerate(front):
                    cd[idx] = cd_sub[k]

            # Tournament selection
            parents = []
            for _ in range(pop_size):
                i = rng.integers(len(F))
                j = rng.integers(len(F))
                if rank[i] < rank[j]:
                    parents.append(i)
                elif rank[j] < rank[i]:
                    parents.append(j)
                else:
                    if cd[i] > cd[j]:
                        parents.append(i)
                    else:
                        parents.append(j)
            return parents

        def crossover(parent1, parent2):
            """SBX crossover."""
            eta = 20.0
            child1 = np.zeros(self.dim)
            child2 = np.zeros(self.dim)
            for k in range(self.dim):
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                child1[k] = 0.5 * ((1 + beta) * parent1[k] + (1 - beta) * parent2[k])
                child2[k] = 0.5 * ((1 - beta) * parent1[k] + (1 + beta) * parent2[k])
            return child1, child2

        def mutate(x, rate=0.1, sigma=0.1):
            """Polynomial mutation."""
            eta = 20.0
            x_mut = x.copy()
            for k in range(self.dim):
                if rng.random() < rate:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x_mut[k] += delta * sigma * span[k]
            return np.clip(x_mut, lo, hi)

        # Main optimization loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Select parents
            parents = select_parents(X, F, pop_size)

            # Generate offspring
            offspring_X = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                p1 = parents[i]
                p2 = parents[(i + 1) % pop_size]
                c1, c2 = crossover(X[p1], X[p2])
                offspring_X[i] = mutate(c1)

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size

            # Combine parent and offspring populations
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])

            # Selection: keep best pop_size individuals
            fronts = fast_non_dominated_sort(F_combined)

            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_combined, front)
                    sorted_front = sorted(range(len(front)), key=lambda k: cd[k], reverse=True)
                    for k in range(remaining):
                        idx = front[sorted_front[k]]
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END

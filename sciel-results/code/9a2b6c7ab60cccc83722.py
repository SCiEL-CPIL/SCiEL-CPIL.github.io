"""Multi-objective optimizer on ZDT4 using NSGA-II with ZDT4-specific initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer tailored for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(50, max(20, budget // 4))

        # ZDT4-specific initialization
        # x0 in [-5, 5], x1..x9 in [0, 1]
        # The Pareto front of ZDT4 has f1 = x0^2, and g depends on sum of sin(10*pi*x_i)
        # Optimal g=1 when each x_i is at a specific point. For ZDT4, x_i should be
        # near 0 or specific values to make g=1.
        # Actually for ZDT4: g(x) = 1 + 10*(n-1) + sum_{i=2}^n [f_i(x_i) - 10*cos(4*pi*x_i)]
        # where f_i(x_i) = x_i^2. So g=1 when x_i^2 - 10*cos(4*pi*x_i) = 0 for all i.
        # This happens at x_i = 0 (since 0 - 10*1 = -10, not 0). Wait, let me reconsider.
        # For ZDT4: f_1 = x_1, f_2 = g(1 + f_1/g * (1 - sqrt(f_1/g)))
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^n [x_i^2 - 10*cos(4*pi*x_i)]
        # Minimum g is 1, achieved when each x_i^2 - 10*cos(4*pi*x_i) = 0
        # At x_i = 0: 0 - 10*1 = -10, so g = 1 + 90 + 9*(-10) = 1, yes!
        # Wait: 1 + 10*9 + sum of 9 terms each = -10 => 1 + 90 - 90 = 1. Correct.
        # So x_i = 0 gives g=1. Also other local minima exist.

        # Initialize with some points near x_i=0 for i>=2, and varied x_0
        X = np.zeros((pop_size, dim))
        X[:, 0] = rng.uniform(lo[0], hi[0], pop_size)
        # For x1..x9, initialize near 0 with some perturbation
        for i in range(1, dim):
            X[:, i] = rng.uniform(lo[i], hi[i], pop_size)
            # Bias towards 0
            X[:, i] = 0.5 * X[:, i] + 0.5 * 0.0

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II operators
        def fast_non_dominated_sort(F):
            """Return list of fronts, each containing indices."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            current_front = [i for i in range(n) if domination_count[i] == 0]
            rank = np.full(n, -1, dtype=int)
            front_idx = 0
            while current_front:
                fronts.append(current_front)
                for i in current_front:
                    rank[i] = front_idx
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
                front_idx += 1

            return fronts, rank

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[indices, m]
                f_min, f_max = f_vals.min(), f_vals.max()
                if f_max - f_min < 1e-30:
                    cd[:] = np.inf
                    continue
                # Sort by objective m
                sorted_idx = np.argsort(f_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (f_vals[sorted_idx[k + 1]] - f_vals[sorted_idx[k - 1]]) / (f_max - f_min)
            return cd

        def tournament_select(X, F, rank, k=2):
            """Tournament selection."""
            idx1 = rng.integers(len(X))
            idx2 = rng.integers(len(X))
            if rank[idx1] < rank[idx2]:
                return idx1
            elif rank[idx2] < rank[idx1]:
                return idx2
            else:
                # Same rank, pick randomly
                return idx1 if rng.random() < 0.5 else idx2

        def crossover(x1, x2):
            """SBX crossover for multi-objective."""
            eta = 20.0
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(dim):
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        def mutation(x, rate=0.1, sigma=0.2):
            """Gaussian mutation."""
            child = x.copy()
            mask = rng.random(dim) < rate
            child[mask] += sigma * (hi[mask] - lo[mask]) * rng.standard_normal(np.sum(mask))
            return np.clip(child, lo, hi)

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Sort into fronts
            fronts, rank = fast_non_dominated_sort(F)

            # Selection
            offspring = []
            while len(offspring) < pop_size:
                if time.time() > deadline:
                    break
                p1 = tournament_select(X, F, rank)
                p2 = tournament_select(X, F, rank)
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutation(c1)
                c2 = mutation(c2)
                offspring.append(c1)
                used += 1
                if used >= budget:
                    break
                if len(offspring) < pop_size:
                    offspring.append(c2)
                    used += 1

            if not offspring:
                break

            offspring = np.array(offspring[:pop_size])
            F_off = np.array([problem(x) for x in offspring])

            # Combine parents and offspring
            X_comb = np.vstack([X, offspring])
            F_comb = np.vstack([F, F_off])

            # Sort combined population
            fronts_comb, rank_comb = fast_non_dominated_sort(F_comb)

            # Select next generation
            X_new = []
            F_new = []
            for front in fronts_comb:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(X_new)
                    cd = crowding_distance(F_comb, front)
                    sorted_by_cd = np.argsort(-cd)
                    for k in range(remaining):
                        idx = front[sorted_by_cd[k]]
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    break

            X = np.array(X_new[:pop_size])
            F = np.array(F_new[:pop_size])

        return X
# EVOLVE-BLOCK-END

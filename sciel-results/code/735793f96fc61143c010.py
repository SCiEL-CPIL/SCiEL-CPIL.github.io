"""Multi-objective optimizer for ZDT2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA for ZDT2 with structured initialization and fast non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Population size: balance between spread and generations
        pop_size = min(50, budget // 2)
        if pop_size < 10:
            pop_size = 10

        # Structured initialization for ZDT2: spread x[0] and vary x[1:]
        # x[0] controls f1, x[1:] controls g which affects f2
        X = np.zeros((pop_size, dim))
        # Spread x[0] uniformly
        X[:, 0] = np.linspace(0, 1, pop_size, endpoint=True)
        # For x[1:], use random values but with some structure
        rng = np.random.default_rng(42)
        X[:, 1:] = rng.random((pop_size, dim - 1))

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Fast non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
            n_count = np.zeros(n, dtype=int)  # number of points dominating i
            rank = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1

            front_idx = 0
            current_front = [i for i in range(n) if n_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    rank[i] = front_idx
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
                front_idx += 1

            return fronts

        # Crowding distance
        def crowding_distance(F, front):
            if len(front) <= 2:
                return np.full(len(front), np.inf)
            m = F.shape[1]
            cd = np.zeros(len(front))
            for k in range(m):
                fk = F[front, k]
                min_val, max_val = fk.min(), fk.max()
                if max_val - min_val < 1e-12:
                    continue
                idx_order = np.argsort(fk)
                cd[idx_order[0]] = np.inf
                cd[idx_order[-1]] = np.inf
                for i in range(1, len(front) - 1):
                    cd[idx_order[i]] += (fk[idx_order[i + 1]] - fk[idx_order[i - 1]]) / (max_val - min_val)
            return cd

        # Evolutionary loop
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Build next generation
            next_X = []
            next_F = []
            next_used = 0

            # Fill with non-dominated fronts
            for front in fronts:
                if next_used + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X[idx].copy())
                        next_F.append(F[idx].copy())
                        next_used += 1
                else:
                    # Crowding distance for this front
                    cd = crowding_distance(F, front)
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    remaining = pop_size - next_used
                    for idx in front_sorted[:remaining]:
                        next_X.append(X[idx].copy())
                        next_F.append(F[idx].copy())
                        next_used += 1
                    break

            # Generate offspring
            n_offspring = budget - used
            if n_offspring <= 0:
                break

            # Select parents from first few fronts
            parent_pool = []
            for front in fronts[:2]:
                parent_pool.extend(front)
            if not parent_pool:
                parent_pool = list(range(len(X)))

            new_X = []
            new_F = []
            for _ in range(n_offspring):
                if time.time() > deadline - 5.0:
                    break

                # Tournament selection
                p1 = parent_pool[rng.integers(len(parent_pool))]
                p2 = parent_pool[rng.integers(len(parent_pool))]

                # SBX crossover
                child = X[p1].copy()
                if rng.random() < 0.9:  # crossover probability
                    for d in range(dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 10))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (2 + 10))
                            child[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])

                # Polynomial mutation
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / (2 + 10)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (2 + 10))
                        child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                new_X.append(child)
                new_F.append(fc)

            if not new_X:
                break

            # Combine and select
            combined_X = np.array(next_X + new_X)
            combined_F = np.array(next_F + new_F)

            # Truncate to pop_size using non-dominated sorting
            combined_fronts = non_dominated_sort(combined_F)
            final_idx = []
            for front in combined_fronts:
                if len(final_idx) + len(front) <= pop_size:
                    final_idx.extend(front)
                else:
                    cd = crowding_distance(combined_F, front)
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    final_idx.extend(front_sorted[:pop_size - len(final_idx)])
                    break

            X = combined_X[final_idx]
            F = combined_F[final_idx]

        return X
# EVOLVE-BLOCK-END

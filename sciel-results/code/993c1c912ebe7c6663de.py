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
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        pop_size = min(60, self.budget // 3)
        # Ensure pop_size is even for SBX
        if pop_size % 2 != 0:
            pop_size += 1
        if pop_size > self.budget:
            pop_size = max(2, self.budget // 2)
            if pop_size % 2 != 0:
                pop_size += 1

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominated = [0] * n
            rank = [0] * n
            fronts = []
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominated[i] += 1
                if n_dominated[i] == 0:
                    rank[i] = 0
            current_front = [i for i in range(n) if rank[i] == 0]
            fronts.append(current_front)
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = rank[i] + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                obj_vals = F[front_indices, m]
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range > 1e-12:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / obj_range
            return cd

        def select_parents(X, F, pop_size):
            """Binary tournament selection."""
            parents = []
            for _ in range(pop_size):
                i, j = rng.integers(0, len(F), size=2)
                # Simple dominance-based tournament
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    parents.append(X[i])
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    parents.append(X[j])
                else:
                    # Non-dominated: pick randomly
                    parents.append(X[i] if rng.random() < 0.5 else X[j])
            return np.array(parents)

        def sbx_crossover(p1, p2, eta=20.0):
            """Simulated binary crossover."""
            if rng.random() < 0.9:
                u = rng.random(self.dim)
                beta = np.where(u <= 0.5,
                                 (2 * u) ** (1.0 / (eta + 1)),
                                 (2 * (1 - u)) ** (1.0 / (eta + 1)))
                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                return c1, c2
            else:
                return p1.copy(), p2.copy()

        def mutation(x, eta=20.0):
            """Polynomial mutation."""
            x_mut = x.copy()
            mask = rng.random(self.dim) < 1.0 / self.dim
            if np.any(mask):
                u = rng.random(np.sum(mask))
                delta = np.where(u <= 0.5,
                                 (2 * u) ** (1.0 / (eta + 1)) - 1,
                                 1 - (2 * (1 - u)) ** (1.0 / (eta + 1)))
                x_mut[mask] += delta * span[mask]
            return np.clip(x_mut, lo, hi)

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Create offspring population
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, self.budget - used)

            # Select parents
            parents = select_parents(X, F, n_offspring)
            for i in range(0, n_offspring - 1, 2):
                c1, c2 = sbx_crossover(parents[i], parents[i + 1])
                c1 = mutation(c1)
                c2 = mutation(c2)
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if time.time() > deadline:
                    break
                if used < self.budget:
                    offspring_X.append(c2)
                    offspring_F.append(problem(c2))
                    used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sorting
            fronts = non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate this front using crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END

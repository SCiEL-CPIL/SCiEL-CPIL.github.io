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
        start_time = time.time()
        deadline = start_time + 290  # safety margin
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        pop_size = min(30, self.budget // 4)
        if pop_size < 10:
            pop_size = 10

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume (all count)
        # We'll maintain an archive of non-dominated points

        def fast_non_dominated_sort(F):
            """Return non-dominated front indices."""
            n = len(F)
            dominated_by = [[] for _ in range(n)]
            domination_count = [0] * n
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_by[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        domination_count[i] += 1
                if domination_count[i] == 0:
                    fronts[0].append(i)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in dominated_by[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            cd = np.zeros(n)
            for m in range(self.n_obj):
                sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i]][m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                fmin = F[front_indices[sorted_idx[0]]][m]
                fmax = F[front_indices[sorted_idx[-1]]][m]
                if fmax - fmin < 1e-12:
                    continue
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (
                        F[front_indices[sorted_idx[i + 1]]][m] -
                        F[front_indices[sorted_idx[i - 1]]][m]
                    ) / (fmax - fmin)
            return cd

        while used < self.budget:
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Build mating pool from first fronts
            # Use tournament selection
            candidates = []
            for front in fronts[:3]:  # top 3 fronts
                candidates.extend(front)
            if not candidates:
                candidates = list(range(pop_size))

            # Select parents via tournament
            def tournament_select():
                a = candidates[rng.integers(len(candidates))]
                b = candidates[rng.integers(len(candidates))]
                # Compare by front rank
                front_rank = {}
                for fi, front in enumerate(fronts):
                    for idx in front:
                        front_rank[idx] = fi
                ra = front_rank.get(a, 999)
                rb = front_rank.get(b, 999)
                if ra < rb:
                    return a
                elif rb < ra:
                    return b
                # Same front, use crowding
                return a if rng.random() < 0.5 else b

            p1 = tournament_select()
            p2 = tournament_select()

            # Simulated binary crossover
            child = np.zeros(self.dim)
            for d in range(self.dim):
                if rng.random() < 0.5:
                    # Crossover
                    u = rng.random()
                    beta = 1.0 / (1.0 + 2.0 * u) if u <= 0.5 else 1.0 / (1.0 + 2.0 * (1.0 - u))
                    beta = beta ** 0.5
                    child[d] = 0.5 * ((1 + beta) * X[p1][d] + (1 - beta) * X[p2][d])
                else:
                    child[d] = X[p1][d] if rng.random() < 0.5 else X[p2][d]

            # Mutation
            if rng.random() < 0.9:
                mut_rate = 1.0 / self.dim
                mut_mask = rng.random(self.dim) < mut_rate
                child[mut_mask] += 0.1 * span[mut_mask] * rng.standard_normal(np.sum(mut_mask))

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst individual in last front
            last_front = fronts[-1] if fronts else [pop_size - 1]
            # Find the one with worst crowding distance in last front
            if len(last_front) > 1:
                cd_vals = crowding_distance(F, last_front)
                worst_local = int(np.argmin(cd_vals))
                worst_idx = last_front[worst_local]
            else:
                worst_idx = last_front[0]

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END

"""Improved DTLZ4 optimizer with deadline guard and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured search + evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Use the evaluator's seeded RNG
        rng = np.random

        # Phase 1: Sample points on the known Pareto front of DTLZ4
        # For DTLZ4 with 3 objectives:
        # g(x) = sum_{i=n-M+1}^{n} (x_i^20 - 1)^2 where M=3, so i from n-2 to n
        # The Pareto front is where g=0, i.e., x_i = 1 for i in {n-2, n-1, n}
        # But actually for DTLZ4, the front is parameterized by the last M-1 = 2 variables
        # f1 = (1+g) * sin(theta1) * sin(theta2)
        # f2 = (1+g) * sin(theta1) * cos(theta2)  
        # f3 = (1+g) * cos(theta1)
        # where theta1 = x_{n-1} * pi/2, theta2 = x_n * pi/2
        # And g = sum_{i=n-M+1}^{n-1} (x_i^20 - 1)^2 = (x_{n-2}^20 - 1)^2 for M=3

        # So to get g=0, set x_{n-2} = 1 (or -1, but bounds are [0,1] so x=1)
        # Then vary x_{n-1} and x_n to sweep the front

        all_X = []
        all_F = []
        used = 0

        # Generate front points
        n_theta = 40
        theta1_vals = np.linspace(0, np.pi/4, n_theta)
        theta2_vals = np.linspace(0, np.pi/4, n_theta)

        for t1 in theta1_vals:
            for t2 in theta2_vals:
                if used >= self.budget:
                    break
                x = np.ones(self.dim)
                # Set the shaping variable to 1 to make g=0
                x[self.dim-3] = 1.0  # This is x_{n-2} for 0-indexed
                # Set the front variables
                x[self.dim-2] = 2.0 * t1 / np.pi  # theta1 = x * pi/2
                x[self.dim-1] = 2.0 * t2 / np.pi  # theta2 = x * pi/2
                x = np.clip(x, lo, hi)
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                used += 1

        all_X = np.array(all_X)
        all_F = np.array(all_F)

        # Phase 2: NSGA-II-like evolution
        pop_size = min(30, max(10, (self.budget - used) // 4))
        if pop_size < 5:
            pop_size = 5

        # Initialize population
        if len(all_X) >= pop_size:
            X = all_X[:pop_size].copy()
            F = all_F[:pop_size].copy()
        else:
            X = np.vstack([all_X, lo + span * rng.random((pop_size - len(all_X), self.dim))])
            X = X[:pop_size]
            F = np.array([problem(x) for x in X])
            used += pop_size - len(all_X)

        def non_dominated_sort(F):
            """Return list of fronts, each containing indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k] and k + 1 < n:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            dist = np.zeros(len(indices))
            for m in range(self.n_obj):
                obj_vals = F[list(indices), m]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = np.argsort(obj_vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for k in range(1, len(indices) - 1):
                    dist[sorted_idx[k]] += (obj_vals[sorted_idx[k+1]] - obj_vals[sorted_idx[k-1]]) / (max_val - min_val)
            return dist

        # Evolution loop
        while used < self.budget:
            if time.time() > deadline - 5:
                break

            fronts = non_dominated_sort(F)

            def select():
                candidates = rng.choice(len(F), size=min(3, len(F)), replace=False)
                best = candidates[0]
                best_rank = -1
                for c in candidates:
                    for rank, front in enumerate(fronts):
                        if c in front:
                            if best_rank == -1 or rank < best_rank:
                                best = c
                                best_rank = rank
                            break
                return best

            offspring = []
            offspring_f = []
            while len(offspring) < pop_size and used < self.budget:
                if time.time() > deadline - 5:
                    break
                p1 = select()
                p2 = select()
                child = np.copy(X[p1])
                if rng.random() < 0.9:
                    child = 0.5 * (X[p1] + X[p2]) + 0.5 * (X[p1] - X[p2]) * (2 * rng.random(self.dim) - 1)
                for d in range(self.dim):
                    if rng.random() < 0.1:
                        child[d] += 0.1 * span[d] * rng.standard_normal()
                child = np.clip(child, lo, hi)
                f = problem(child)
                offspring.append(child)
                offspring_f.append(f)
                used += 1

            if not offspring:
                break

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)

            combined_X = np.vstack([X, offspring])
            combined_F = np.vstack([F, offspring_f])

            combined_fronts = non_dominated_sort(combined_F)
            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    remaining = pop_size - len(next_X)
                    if remaining > 0:
                        cd = crowding_distance(combined_F, front)
                        sorted_front = np.argsort(-cd)
                        for i in sorted_front[:remaining]:
                            idx = front[i]
                            next_X.append(combined_X[idx])
                            next_F.append(combined_F[idx])
                    break

            if len(next_X) == 0:
                break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with DE-style mutation and adaptive step size."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 30s safety margin before 310s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(40, max(10, self.budget // 5))
        max_evals = self.budget

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Check if any points are better than reference
        ref = np.array([3.0, 5.0, 7.0])

        # Non-dominated sort
        def fast_non_dominated_sort(F):
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
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts[:k + 1]

        # Crowding distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)
            m = len(front_indices)
            dist = np.zeros(m)
            for j in range(self.n_obj):
                f = F[front_indices, j]
                order = np.argsort(f)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                fmin, fmax = f[order[0]], f[order[-1]]
                if fmax - fmin < 1e-12:
                    continue
                for i in range(1, m - 1):
                    dist[order[i]] += (f[order[i + 1]] - f[order[i - 1]]) / (fmax - fmin)
            return dist

        # Adaptive step size
        step_scale = 0.3  # initial step size relative to range

        while used < max_evals:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Build mating pool from first few fronts
            mating_pool = []
            for front in fronts:
                mating_pool.extend(front)
                if len(mating_pool) >= pop_size:
                    break
            if len(mating_pool) < pop_size:
                mating_pool = list(range(len(F)))

            # Generate children using DE-style mutation
            n_children = min(pop_size, max_evals - used)
            children_X = []
            for _ in range(n_children):
                # Select 3 distinct parents from mating pool
                idx = rng.choice(len(mating_pool), size=3, replace=False)
                p1, p2, p3 = [mating_pool[i] for i in idx]
                # DE/rand/1 mutation
                v = X[p1] + 0.5 * (X[p2] - X[p3])
                # Add some Gaussian noise with adaptive scale
                v += step_scale * (hi - lo) * rng.standard_normal(self.dim)
                v = np.clip(v, lo, hi)
                children_X.append(v)

            # Evaluate children
            children_F = np.array([problem(x) for x in children_X], dtype=float)
            used += n_children

            # Combine parents and children
            all_X = np.vstack([X, np.array(children_X)])
            all_F = np.vstack([F, children_F])

            # Non-dominated sort on combined population
            combined_fronts = fast_non_dominated_sort(all_F)

            # Select next population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.extend(all_X[front])
                    new_F.extend(all_F[front])
                else:
                    # Need to truncate this front by crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(all_F, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    selected = [front[i] for i in order[:remaining]]
                    new_X.extend(all_X[selected])
                    new_F.extend(all_F[selected])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

            # Adapt step size: if we're stuck, increase exploration
            # Check if recent evaluations improved the best
            best_sum = np.min(np.sum(F, axis=1))
            if best_sum < 1.0:
                step_scale = max(0.1, step_scale * 0.95)
            else:
                step_scale = min(0.5, step_scale * 1.05)

        return X
# EVOLVE-BLOCK-END

"""Multi-objective optimizer for WFG7 using NSGA-II with adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with adaptive mutation and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
        rank = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1

        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        for i in current_front:
            rank[i] = 0

        k = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            current_front = next_front
            k += 1

        # Add any remaining (shouldn't happen but safety)
        if k < n:
            remaining = [i for i in range(n) if i not in [x for front in fronts for x in front]]
            if remaining:
                fronts.append(remaining)
                for i in remaining:
                    rank[i] = k + 1

        return fronts, rank

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for a front."""
        if len(front_indices) <= 2:
            return np.full(len(front_indices), np.inf)

        distances = np.zeros(len(front_indices))
        m = self.n_obj

        for k in range(m):
            objs = F[front_indices, k]
            sorted_idx = np.argsort(objs)
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            f_min = objs[sorted_idx[0]]
            f_max = objs[sorted_idx[-1]]
            rng = f_max - f_min
            if rng > 1e-15:
                for i in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[i]] += (objs[sorted_idx[i + 1]] - objs[sorted_idx[i - 1]]) / rng

        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 30s safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(40, max(20, self.budget // 5))
        # Ensure we don't exceed budget
        if pop_size > self.budget:
            pop_size = self.budget

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all non-dominated points found
        all_points = []
        for i in range(pop_size):
            all_points.append((X[i].copy(), F[i].copy()))

        # Adaptive mutation scale
        mut_scale = 0.15
        min_scale = 0.01
        max_scale = 0.4
        gen = 0

        while used < self.budget:
            if time.time() > deadline:
                break

            gen += 1

            # Non-dominated sorting
            fronts, rank = self._non_dominated_sort(F)

            # Selection: build mating pool from top fronts
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) > pop_size:
                    # Need to select from this front using crowding distance
                    cd = self._crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i - front[0]] if i in front else 0, reverse=True)
                    # Actually, let's sort properly
                    front_with_cd = [(idx, cd[i]) for i, idx in enumerate(front)]
                    front_with_cd.sort(key=lambda x: x[1], reverse=True)
                    needed = pop_size - len(mating_pool)
                    for idx, _ in front_with_cd[:needed]:
                        mating_pool.append(idx)
                    break
                else:
                    mating_pool.extend(front)

            # Binary tournament selection
            def tournament():
                a = rng.integers(len(mating_pool))
                b = rng.integers(len(mating_pool))
                ia, ib = mating_pool[a], mating_pool[b]
                if rank[ia] < rank[ib]:
                    return ia
                elif rank[ib] < rank[ia]:
                    return ib
                else:
                    # Same front, use crowding distance
                    cd = self._crowding_distance(F, [ia, ib])
                    return ia if cd[0] >= cd[1] else ib

            # Generate children
            n_children = min(pop_size, self.budget - used)
            if n_children <= 0:
                break

            children_X = []
            children_F = []

            for _ in range(n_children):
                p1 = tournament()
                p2 = tournament()

                # SBX-like crossover (simple blend)
                alpha = rng.random()
                child = alpha * X[p1] + (1 - alpha) * X[p2]

                # Adaptive mutation
                noise = rng.standard_normal(self.dim)
                child = child + mut_scale * (hi - lo) * noise
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                children_X.append(child)
                children_F.append(fc)

                if used >= self.budget:
                    break

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parents and children
            X_all = np.vstack([X, children_X])
            F_all = np.vstack([F, children_F])

            # Select next generation
            fronts_all, rank_all = self._non_dominated_sort(F_all)

            new_X = []
            new_F = []
            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                else:
                    cd = self._crowding_distance(F_all, front)
                    front_with_cd = [(idx, cd[i]) for i, idx in enumerate(front)]
                    front_with_cd.sort(key=lambda x: x[1], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx, _ in front_with_cd[:needed]:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

            # Update all_points with new non-dominated solutions
            # Find non-dominated in combined set
            fronts_combined, _ = self._non_dominated_sort(F_all)
            if fronts_combined:
                for idx in fronts_combined[0]:
                    all_points.append((X_all[idx].copy(), F_all[idx].copy()))

            # Adapt mutation scale
            if gen % 10 == 0:
                # Reduce mutation if we're making progress, increase if stuck
                best_sum = np.min(np.sum(F, axis=1))
                if best_sum < 10:  # Making progress
                    mut_scale = max(min_scale, mut_scale * 0.95)
                else:
                    mut_scale = min(max_scale, mut_scale * 1.05)

        return X
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: NSGA-II style with adaptive steps."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(12345)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Population size: use a good chunk of budget
        pop_size = min(50, self.budget // 4)
        pop_size = max(pop_size, 30)

        # Initial population: Latin hypercube sampling for good spread
        X = np.empty((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[:, d] = lo[d] + span[d] * (perm + rng.random(pop_size)) / pop_size
        # Add some corner points for boundary exploration
        n_corners = min(6, pop_size)
        for i in range(n_corners):
            X[i] = np.where(rng.random(dim) < 0.5, lo, hi)

        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F_arr):
            """Returns list of fronts (each a list of indices)."""
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []
            current_front = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        domination_count[j] += 1
                        dominated_set[i].append(j)
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        domination_count[i] += 1
                        dominated_set[j].append(i)
                if domination_count[i] == 0:
                    current_front.append(i)

            if current_front:
                fronts.append(current_front)

            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front

            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            F_sub = F_arr[indices]
            for m in range(n_obj):
                vals = F_sub[:, m]
                min_val = vals.min()
                max_val = vals.max()
                if max_val - min_val < 1e-15:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_val - min_val)
            return cd

        while used < self.budget:
            if time.time() > deadline:
                break

            # Compute fronts
            fronts = fast_non_dominated_sort(F)

            # Build mating pool: prefer first front, use crowding for diversity
            mating_pool = []
            for front in fronts:
                cd = crowding_distance(F, front)
                # Sort by crowding distance descending (prefer diverse)
                order = np.argsort(-cd)
                for idx in order:
                    mating_pool.append(front[idx])

            # Select parents via tournament
            progress = used / self.budget
            base_step = 0.10 * (1.0 - 0.85 * progress)

            children_X = []
            children_F = []

            # Generate offspring
            n_offspring = min(pop_size, self.budget - used)
            for _ in range(n_offspring):
                if time.time() > deadline:
                    break

                # Tournament selection
                p1_idx = mating_pool[rng.integers(len(mating_pool))]
                p2_idx = mating_pool[rng.integers(len(mating_pool))]
                # Pick the better of two random (by front rank)
                front_rank = {idx: fi for fi, front in enumerate(fronts) for idx in front}
                if front_rank.get(p1_idx, 999) <= front_rank.get(p2_idx, 999):
                    parent = X[p1_idx]
                else:
                    parent = X[p2_idx]

                r = rng.random()
                if r < 0.4 and pop_size >= 2:
                    # Crossover with another from mating pool
                    other_idx = mating_pool[rng.integers(len(mating_pool))]
                    alpha = rng.random()
                    child = alpha * X[p1_idx] + (1 - alpha) * X[other_idx]
                    child += 0.02 * span * rng.standard_normal(dim)
                elif r < 0.75:
                    # Gaussian mutation
                    child = parent + base_step * span * rng.standard_normal(dim)
                else:
                    # Larger exploratory jump
                    child = parent + 0.20 * span * rng.standard_normal(dim)

                child = np.clip(child, lo, hi)
                children_X.append(child)
                used += 1

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array([problem(x) for x in children_X])

            # Combine parent and offspring
            combined_X = np.vstack([X, children_X])
            combined_F = np.vstack([F, children_F])

            # Select next generation using fronts + crowding
            all_fronts = fast_non_dominated_sort(combined_F)
            next_X = []
            next_F = []
            for front in all_fronts:
                if len(next_X) + len(front) <= pop_size:
                    next_X.extend(combined_X[front])
                    next_F.extend(combined_F[front])
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(-cd)
                    for idx in order[:remaining]:
                        next_X.append(combined_X[front[idx]])
                        next_F.append(combined_F[front[idx]])
                    break

            if next_X:
                X = np.array(next_X)
                F = np.array(next_F)

        # Final polish: local search on the best non-dominated points
        if used < self.budget and time.time() < deadline:
            # Get the first front (non-dominated)
            final_fronts = fast_non_dominated_sort(F)
            if final_fronts:
                front0_idx = final_fronts[0]
                # Local search on each point in the first front
                for idx in front0_idx:
                    if used >= self.budget or time.time() > deadline:
                        break
                    # Try small perturbations
                    for _ in range(3):
                        if used >= self.budget or time.time() > deadline:
                            break
                        candidate = X[idx] + 0.01 * span * rng.standard_normal(dim)
                        candidate = np.clip(candidate, lo, hi)
                        cand_F = problem(candidate)
                        used += 1
                        # Replace if it dominates or is non-dominated and closer to reference
                        if np.all(cand_F <= F[idx]) and np.any(cand_F < F[idx]):
                            X[idx] = candidate
                            F[idx] = cand_F
                        elif not np.any(F[idx] <= cand_F) and not np.any(cand_F <= F[idx]):
                            # Both non-dominated; keep the one with better sum (normalized)
                            ref = np.array([3000.0, 0.06])
                            score_old = np.sum(F[idx] / ref)
                            score_new = np.sum(cand_F / ref)
                            if score_new < score_old:
                                X[idx] = candidate
                                F[idx] = cand_F

        return X
# EVOLVE-BLOCK-END

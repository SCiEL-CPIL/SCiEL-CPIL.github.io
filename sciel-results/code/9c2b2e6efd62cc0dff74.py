"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: random init, NSGA-II style with hypervolume-aware selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Use a fixed RNG for reproducibility within a single call
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(60, max(20, self.budget // 5))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume (keep all non-dominated or a subset)
        # We'll keep a list of all evaluated F vectors that are better than reference
        ref_point = np.array([3.0, 5.0, 7.0])
        all_F = [F.copy()]  # list of arrays, or we can accumulate

        # Actually, let's just track all F values seen
        all_F_list = []
        for f in F:
            if np.all(f < ref_point):
                all_F_list.append(f.copy())

        def fast_non_dominated_sort(F_arr):
            """Return list of fronts, each front is list of indices."""
            n = len(F_arr)
            S = [[] for _ in range(n)]
            n_doms = [0] * n
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_doms[i] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        n_doms[j] += 1
                if n_doms[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for k in fronts[i]:
                    for m in S[k]:
                        n_doms[m] -= 1
                        if n_doms[m] == 0:
                            next_front.append(m)
                i += 1
                fronts.append(next_front)

            # Remove trailing empty fronts
            while fronts and not fronts[-1]:
                fronts.pop()

            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for given indices in F_arr."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F_arr[indices, m]
                min_val = np.min(vals)
                max_val = np.max(vals)
                if max_val - min_val < 1e-14:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_val - min_val)
            return cd

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Get fronts
            fronts = fast_non_dominated_sort(F)

            # Create mating pool from first few fronts
            mating_pool = []
            for front in fronts:
                mating_pool.extend(front)
                if len(mating_pool) >= pop_size:
                    break

            # If mating pool is too small, add more
            if len(mating_pool) < pop_size:
                remaining = [i for i in range(pop_size) if i not in mating_pool]
                mating_pool.extend(remaining[:pop_size - len(mating_pool)])

            # Select parents via tournament
            def tournament_select(pool, size=2):
                candidates = [pool[rng.integers(len(pool))] for _ in range(size)]
                best = candidates[0]
                for c in candidates[1:]:
                    # Compare based on front rank and crowding
                    # For simplicity, use front index
                    front_idx = {idx: i for i, f in enumerate(fronts) for idx in f}
                    if front_idx.get(c, len(fronts)) < front_idx.get(best, len(fronts)):
                        best = c
                    elif front_idx.get(c, len(fronts)) == front_idx.get(best, len(fronts)):
                        # Use crowding distance
                        pass  # Simplified: just pick random among tied
                return best

            # Generate offspring
            new_X = []
            new_F = []
            n_offspring = pop_size - len(mating_pool) if len(mating_pool) < pop_size else pop_size

            for _ in range(n_offspring):
                p1 = mating_pool[rng.integers(len(mating_pool))]
                p2 = mating_pool[rng.integers(len(mating_pool))]

                # Simulated binary crossover (SBX)
                child = X[p1].copy()
                for d in range(dim):
                    if rng.random() < 0.5:
                        x1, x2 = X[p1, d], X[p2, d]
                        if x1 == x2:
                            continue
                        if rng.random() < 0.5:
                            beta = (0.5 * (child[d] - x1) / (x2 - x1 + 1e-14)) if x1 != x2 else 0.5
                        else:
                            beta = (0.5 * (x2 - child[d]) / (x2 - x1 + 1e-14)) if x1 != x2 else 0.5
                        beta = max(0.1, min(10.0, beta))
                        child[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)

                # Polynomial mutation
                eta = 20.0
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        delta = rng.random()
                        if delta < 0.5:
                            mut = (2 * delta) ** (1 / (eta + 1)) - 1
                        else:
                            mut = 1 - (2 * (1 - delta)) ** (1 / (eta + 1))
                        child[d] += mut * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                fc = problem(child)
                new_X.append(child)
                new_F.append(fc)
                used += 1

                if time.time() > deadline:
                    break

            if not new_X:
                break

            # Combine parents and offspring
            combined_X = np.vstack([X[mating_pool], np.array(new_X)])
            combined_F = np.vstack([F[mating_pool], np.array(new_F)])

            # Track new F values
            for f in np.array(new_F):
                if np.all(f < ref_point):
                    all_F_list.append(f.copy())

            # Select next generation
            fronts = fast_non_dominated_sort(combined_F)
            next_idx = []
            for front in fronts:
                if len(next_idx) + len(front) <= pop_size:
                    next_idx.extend(front)
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(next_idx)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(cd)[::-1]
                    next_idx.extend(front[order[:remaining]])
                    break

            if len(next_idx) < pop_size:
                # Fill with random
                remaining = [i for i in range(len(combined_F)) if i not in next_idx]
                next_idx.extend(remaining[:pop_size - len(next_idx)])

            X = combined_X[next_idx[:pop_size]]
            F = combined_F[next_idx[:pop_size]]

        # Return the best points found (all_F_list is not returned, but we return X)
        # The evaluator records all objective vectors returned by problem(x) calls
        # We return X to satisfy the interface
        return X
# EVOLVE-BLOCK-END

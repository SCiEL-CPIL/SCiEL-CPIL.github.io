"""Multi-objective optimizer on WFG4 with structure-aware search and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware evolutionary multi-objective optimizer with local polish for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Track all evaluated points
        all_X = []
        all_F = []

        def evaluate(x):
            """Evaluate a single point, record it."""
            fx = problem(x)
            all_X.append(x.copy())
            all_F.append(np.asarray(fx, dtype=np.float64))
            return np.asarray(fx, dtype=np.float64)

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points in F (n x k array)."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                # Check if any other point dominates i
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # set of points dominated by i
            ndom = np.zeros(n, dtype=int)  # number of points that dominate i
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        ndom[i] += 1
                if ndom[i] == 0:
                    fronts.append([i])

            k = 0
            while k < len(fronts):
                current_front = fronts[k]
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        ndom[j] -= 1
                        if ndom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            F_sub = F[indices]
            for m in range(F.shape[1]):
            # For each objective, sort and assign distance
                order = np.argsort(F_sub[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = F_sub[order[-1], m] - F_sub[order[0], m]
                if f_range > 1e-15:
                    for i in range(1, n - 1):
                        cd[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / f_range
            return cd

        def generate_initial(x_center=None, spread=0.3):
            """Generate a point, optionally biased toward a center."""
            if x_center is not None:
                x = x_center + spread * (hi - lo) * rng.standard_normal(self.dim)
            else:
                x = lo + (hi - lo) * rng.random(self.dim)
            return np.clip(x, lo, hi)

        # WFG4 structure: x1 controls the front shape. For 3-objective WFG4,
        # the Pareto front is controlled by the first variable(s).
        # Bias initialization toward the center of the search space for x1.
        x_center = lo + 0.5 * (hi - lo)
        x_center[0] = lo[0] + 0.35 * (hi[0] - lo[0])  # bias x1 toward 0.35

        # Phase 1: Initial population with structure-aware initialization
        pop_size = 30
        X = np.zeros((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))

        # First few points biased toward the known Pareto region
        for i in range(min(10, pop_size)):
            X[i] = generate_initial(x_center=x_center, spread=0.15)
        # Rest random
        for i in range(10, pop_size):
            X[i] = generate_initial()

        for i in range(pop_size):
            F[i] = evaluate(X[i])

        # Phase 2: Evolutionary search (NSGA-II style)
        while len(all_X) < self.budget:
            if time.time() > deadline - 60:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Selection: build mating pool from best fronts
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) >= pop_size:
                    # Need crowding distance to truncate
                    cd = crowding_distance(F, front)
                    order = np.argsort(-cd)
                    need = pop_size - len(mating_pool)
                    mating_pool.extend(front[order[:need]])
                    break
                mating_pool.extend(front)

            # Generate children via mutation of selected parents
            n_children = min(pop_size, self.budget - len(all_X))
            if n_children <= 0:
                break

            new_X = np.zeros((n_children, self.dim))
            new_F = np.zeros((n_children, self.n_obj))

            for i in range(n_children):
                # Tournament selection from mating pool
                p1_idx = mating_pool[rng.integers(len(mating_pool))]
                p1 = X[p1_idx]

                # Adaptive mutation
                if rng.random() < 0.3:
                    # Small perturbation around parent
                    mutation_scale = 0.05 * (hi - lo)
                    child = p1 + mutation_scale * rng.standard_normal(self.dim)
                elif rng.random() < 0.5:
                    # Medium perturbation
                    mutation_scale = 0.15 * (hi - lo)
                    child = p1 + mutation_scale * rng.standard_normal(self.dim)
                else:
                    # Large exploration step
                    child = generate_initial(x_center=x_center if rng.random() < 0.5 else None,
                                             spread=0.25)

                child = np.clip(child, lo, hi)
                new_X[i] = child
                new_F[i] = evaluate(child)

            # Combine parent and child populations
            X_combined = np.vstack([X, new_X])
            F_combined = np.vstack([F, new_F])

            # Non-dominated sort of combined
            fronts = fast_non_dominated_sort(F_combined)

            # Select next population
            X = np.zeros((pop_size, self.dim))
            F = np.zeros((pop_size, self.n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        X[count] = X_combined[idx]
                        F[count] = F_combined[idx]
                        count += 1
                else:
                    # Truncate using crowding distance
                    cd = crowding_distance(F_combined, front)
                    order = np.argsort(-cd)
                    need = pop_size - count
                    for j in range(need):
                        idx = front[order[j]]
                        X[count] = X_combined[idx]
                        F[count] = F_combined[idx]
                        count += 1
                    break

        # Phase 3: Local polish - refine the best non-dominated points
        if time.time() < deadline - 10:
            # Get non-dominated points from all evaluations
            all_F = np.array(all_F)
            all_X_arr = np.array(all_X)
            nd_mask = non_dominated_mask(all_F)
            nd_indices = np.where(nd_mask)[0]

            # Polish the best 15 non-dominated points with local search
            n_polish = min(15, len(nd_indices))
            if n_polish > 0:
                # Sort by sum of objectives (simpler heuristic for "best")
                sums = all_F[nd_indices].sum(axis=1)
                best_nd = nd_indices[np.argsort(sums)[:n_polish]]

                for idx in best_nd:
                    if time.time() > deadline - 5:
                        break
                    base_x = all_X_arr[idx].copy()
                    base_f = all_F[idx].copy()

                    # Coordinate-wise local search
                    for dim_i in range(self.dim):
                        if time.time() > deadline - 5:
                            break
                        step = 0.02 * (hi[dim_i] - lo[dim_i])
                        for direction in [-1, 1]:
                            trial = base_x.copy()
                            trial[dim_i] += direction * step
                            trial = np.clip(trial, lo, hi)
                            if np.array_equal(trial, base_x):
                                continue
                            trial_f = evaluate(trial)
                            # Check if trial dominates base
                            if np.all(trial_f <= base_f) and np.any(trial_f < base_f):
                                base_x = trial
                                base_f = trial_f
                            elif np.all(base_f <= trial_f) and np.any(base_f < trial_f):
                                pass  # base still better
                            else:
                                # Neither dominates; keep base
                                pass
                        # Also try smaller step
                        step = 0.005 * (hi[dim_i] - lo[dim_i])
                        for direction in [-1, 1]:
                            trial = base_x.copy()
                            trial[dim_i] += direction * step
                            trial = np.clip(trial, lo, hi)
                            if np.array_equal(trial, base_x):
                                continue
                            trial_f = evaluate(trial)
                            if np.all(trial_f <= base_f) and np.any(trial_f < base_f):
                                base_x = trial
                                base_f = trial_f

        return all_X
# EVOLVE-BLOCK-END

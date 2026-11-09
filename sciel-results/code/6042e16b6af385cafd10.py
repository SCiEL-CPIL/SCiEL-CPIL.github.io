"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using NSGA-II-style population-based
    search with non-dominated sorting, crowding distance, and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)
        ref = np.array([1.1, 1.3, 1.1])

        def fast_non_dominated_sort(F_arr):
            """Returns list of fronts, each a list of indices."""
            n = len(F_arr)
            if n == 0:
                return []
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            current_front = [i for i in range(n) if domination_count[i] == 0]
            fronts = []
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def compute_rank_array(fronts, n):
            """Compute rank for each index for O(1) tournament selection."""
            rank = np.zeros(n, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            return rank

        def crowding_distance(F_arr, front_indices):
            """Compute crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            F_front = F_arr[front_indices]
            
            for m in range(n_obj):
                sorted_idx = np.argsort(F_front[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F_front[sorted_idx[i+1], m] - F_front[sorted_idx[i-1], m]) / f_range
            
            return cd

        def compute_cd_array(F_arr, fronts):
            """Compute crowding distance for all points, returns array indexed by original index."""
            n = len(F_arr)
            cd_arr = np.zeros(n)
            for front in fronts:
                if len(front) > 2:
                    cd_vals = crowding_distance(F_arr, front)
                    for i, idx in enumerate(front):
                        cd_arr[idx] = cd_vals[i]
            return cd_arr

        def tournament_select(rank, cd_arr, n, k=2):
            """Fast tournament selection using precomputed rank and cd arrays."""
            candidates = rng.integers(n, size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[best] < rank[c]:
                    continue
                elif rank[c] < rank[best]:
                    best = c
                else:
                    if cd_arr[c] > cd_arr[best]:
                        best = c
            return best

        def local_polish_hv(x, f, remaining_budget, max_evals_per_point=10):
            """Polish a point using hypervolume-guided local search.
            Returns (new_x, new_f, evals_used)"""
            best_x = x.copy()
            best_f = f.copy()
            evals_used = 0
            step = 0.03
            
            for iteration in range(max_evals_per_point):
                if evals_used >= remaining_budget or time.time() > deadline:
                    break
                improved = False
                # Try all directions with adaptive step
                for d in range(dim):
                    for direction in [-1, 1]:
                        trial_x = best_x.copy()
                        trial_x[d] += direction * step
                        trial_x = np.clip(trial_x, lo, hi)
                        trial_f = problem(trial_x)
                        evals_used += 1
                        if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                            best_x = trial_x
                            best_f = trial_f
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.7  # Reduce step if no improvement
                    if step < 0.005:
                        break
            
            return best_x, best_f, evals_used

        # Phase 1: Latin Hypercube Sampling initialization
        pop_size = min(40, budget)
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))

        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[perm, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(pop_size) + rng.random(pop_size)) / pop_size

        used = 0
        for i in range(pop_size):
            F[i] = problem(X[i])
            used += 1
            if time.time() > deadline:
                break

        # Phase 2: NSGA-II-style evolution
        gen = 0
        while used < budget:
            if time.time() > deadline:
                break
            gen += 1
            
            # Adaptive population size: start larger, shrink as we approach budget
            if gen < 5:
                current_pop = pop_size
            else:
                current_pop = max(20, pop_size - gen)
            
            # Compute fronts, rank, and crowding distances
            fronts = fast_non_dominated_sort(F)
            rank = compute_rank_array(fronts, len(F))
            cd_arr = compute_cd_array(F, fronts)

            # Generate offspring
            n_offspring = min(current_pop, budget - used)
            offspring_X = np.zeros((n_offspring, dim))
            offspring_F = np.zeros((n_offspring, n_obj))
            n_off = 0

            for _ in range(n_offspring):
                if used >= budget or time.time() > deadline:
                    break

                # Select parents via fast tournament
                p1 = tournament_select(rank, cd_arr, len(F))
                p2 = tournament_select(rank, cd_arr, len(F))

                # Crossover (BLX-alpha style)
                if rng.random() < 0.9:
                    alpha = 0.5
                    for d in range(dim):
                        if p1 != p2:
                            lo_c = min(X[p1, d], X[p2, d]) - alpha * abs(X[p1, d] - X[p2, d])
                            hi_c = max(X[p1, d], X[p2, d]) + alpha * abs(X[p1, d] - X[p2, d])
                            offspring_X[n_off, d] = rng.uniform(max(lo[d], lo_c), min(hi[d], hi_c))
                        else:
                            offspring_X[n_off, d] = X[p1, d]
                else:
                    offspring_X[n_off, :] = X[p1, :]

                # Mutation with adaptive step size
                progress = used / budget
                sigma = 0.2 * (1.0 - 0.5 * progress)
                mask = rng.random(dim) < 0.8
                if np.any(mask):
                    offspring_X[n_off, mask] += sigma * (hi - lo)[mask] * rng.standard_normal(np.sum(mask))
                offspring_X[n_off, :] = np.clip(offspring_X[n_off, :], lo, hi)

                offspring_F[n_off] = problem(offspring_X[n_off, :])
                used += 1
                n_off += 1

            if n_off == 0:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X[:len(F)], offspring_X[:n_off]])
            combined_F = np.vstack([F, offspring_F[:n_off]])
            n_combined = len(F) + n_off

            # Non-dominated sort the combined set
            combined_fronts = fast_non_dominated_sort(combined_F)
            combined_rank = compute_rank_array(combined_fronts, n_combined)
            combined_cd = compute_cd_array(combined_F, combined_fronts)

            # Select next population
            next_X = np.zeros((current_pop, dim))
            next_F = np.zeros((current_pop, n_obj))
            next_count = 0
            for fi, front in enumerate(combined_fronts):
                if next_count + len(front) <= current_pop:
                    for idx in front:
                        next_X[next_count] = combined_X[idx]
                        next_F[next_count] = combined_F[idx]
                        next_count += 1
                else:
                    remaining = current_pop - next_count
                    cd_vals = crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(-cd_vals)
                    for i in range(min(remaining, len(sorted_by_cd))):
                        idx = front[sorted_by_cd[i]]
                        next_X[next_count] = combined_X[idx]
                        next_F[next_count] = combined_F[idx]
                        next_count += 1
                    break

            X = next_X
            F = next_F

        # Phase 3: Local polish of non-dominated points with remaining budget
        if used < budget and time.time() < deadline:
            final_fronts = fast_non_dominated_sort(F)
            nd_indices = final_fronts[0] if final_fronts else []
            
            # Sort by potential hypervolume contribution (heuristic: points more central in objective space)
            if len(nd_indices) > 1:
                F_nd = F[nd_indices]
                # Score by how "central" each point is (lower is better for central)
                centers = np.mean(F_nd, axis=0)
                dists = np.sqrt(np.sum((F_nd - centers) ** 2, axis=1))
                order = np.argsort(dists)  # Most central first
                nd_indices = [nd_indices[i] for i in order]
            
            for idx in nd_indices:
                if used >= budget or time.time() > deadline:
                    break
                remaining = budget - used
                polished_x, polished_f, evals_used = local_polish_hv(X[idx], F[idx], remaining)
                if evals_used > 0:
                    used += evals_used
                    if np.all(polished_f <= F[idx]) and np.any(polished_f < F[idx]):
                        X[idx] = polished_x
                        F[idx] = polished_f

        return X
# EVOLVE-BLOCK-END

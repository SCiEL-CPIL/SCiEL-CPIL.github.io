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
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)
        ref = np.array([1.1, 1.3, 1.1])

        # Track all evaluated points for final hypervolume
        all_X = []
        all_F = []
        eval_count = 0

        def dominates(a, b):
            """Check if vector a dominates vector b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

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

        def tournament_select(F_arr, fronts, cd_map, k=2):
            """Tournament selection based on rank then crowding distance."""
            n = len(F_arr)
            candidates = rng.integers(n, size=k)
            best = candidates[0]
            for c in candidates[1:]:
                rank_b = next(r for r, f in enumerate(fronts) if best in f)
                rank_c = next(r for r, f in enumerate(fronts) if c in f)
                if rank_b < rank_c:
                    continue
                elif rank_c < rank_b:
                    best = c
                else:
                    # Same rank, compare crowding distance
                    front_idx = next(r for r, f in enumerate(fronts) if best in f)
                    cd_b = cd_map.get(front_idx, {}).get(best, 0)
                    cd_c = cd_map.get(front_idx, {}).get(c, 0)
                    if cd_c > cd_b:
                        best = c
            return best

        def local_polish(x, f, max_iter=50, step=0.02):
            """Polish a single point using gradient-free local search."""
            best_x = x.copy()
            best_f = f.copy()
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        trial_x = best_x.copy()
                        trial_x[d] += direction * step
                        trial_x = np.clip(trial_x, lo, hi)
                        trial_f = problem(trial_x)
                        if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                            best_x = trial_x
                            best_f = trial_f
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            return best_x, best_f

        # Phase 1: Latin Hypercube Sampling initialization
        pop_size = min(40, budget)
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))

        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[perm, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(pop_size) + rng.random(pop_size)) / pop_size

        for i in range(pop_size):
            F[i] = problem(X[i])
            eval_count += 1
            all_X.append(X[i].copy())
            all_F.append(F[i].copy())
            if time.time() > deadline:
                break
        used = eval_count

        # Phase 2: NSGA-II-style evolution
        while used < budget:
            if time.time() > deadline:
                break

            # Compute fronts and crowding distances
            fronts = fast_non_dominated_sort(F)
            cd_map = {}
            for fi, front in enumerate(fronts):
                cd_vals = crowding_distance(F, front)
                cd_map[fi] = {front[i]: cd_vals[i] for i in range(len(front))}

            # Generate offspring
            n_offspring = min(pop_size, budget - used)
            offspring_X = np.zeros((n_offspring, dim))
            offspring_F = np.zeros((n_offspring, n_obj))
            n_off = 0

            for _ in range(n_offspring):
                if used >= budget or time.time() > deadline:
                    break

                # Select parents via tournament
                p1 = tournament_select(F, fronts, cd_map)
                p2 = tournament_select(F, fronts, cd_map)

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
                mask = rng.random(dim) < 0.8  # mutate 80% of dimensions
                if np.any(mask):
                    offspring_X[n_off, mask] += sigma * (hi - lo)[mask] * rng.standard_normal(np.sum(mask))
                offspring_X[n_off, :] = np.clip(offspring_X[n_off, :], lo, hi)

                offspring_F[n_off] = problem(offspring_X[n_off, :])
                eval_count += 1
                all_X.append(offspring_X[n_off, :].copy())
                all_F.append(offspring_F[n_off, :].copy())
                used += 1
                n_off += 1

            if n_off == 0:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X[:pop_size], offspring_X[:n_off]])
            combined_F = np.vstack([F[:pop_size], offspring_F[:n_off]])
            n_combined = pop_size + n_off

            # Non-dominated sort the combined set
            combined_fronts = fast_non_dominated_sort(combined_F)
            combined_cd_map = {}
            for fi, front in enumerate(combined_fronts):
                cd_vals = crowding_distance(combined_F, front)
                combined_cd_map[fi] = {front[i]: cd_vals[i] for i in range(len(front))}

            # Select next population
            next_X = np.zeros((pop_size, dim))
            next_F = np.zeros((pop_size, n_obj))
            next_count = 0
            for fi, front in enumerate(combined_fronts):
                if next_count + len(front) <= pop_size:
                    for idx in front:
                        next_X[next_count] = combined_X[idx]
                        next_F[next_count] = combined_F[idx]
                        next_count += 1
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - next_count
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

        # Phase 3: Local polish of non-dominated points
        if used < budget and time.time() < deadline:
            final_fronts = fast_non_dominated_sort(F)
            nd_indices = final_fronts[0] if final_fronts else []
            
            for idx in nd_indices:
                if used >= budget or time.time() > deadline:
                    break
                polished_x, polished_f = local_polish(X[idx], F[idx])
                # Record polished point if it's new
                all_X.append(polished_x.copy())
                all_F.append(polished_f.copy())
                eval_count += 1  # Approximate
                # Check if polished point is better and replace
                if np.all(polished_f <= F[idx]) and np.any(polished_f < F[idx]):
                    X[idx] = polished_x
                    F[idx] = polished_f
                    used += 1

        return X
# EVOLVE-BLOCK-END

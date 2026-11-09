"""Multi-objective optimizer for ZDT4 combining NSGA-II with hypervolume-guided selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided NSGA-II for ZDT4 multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([11.0, 400.0])

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume for points strictly better than reference."""
            if len(points) == 0:
                return 0.0
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            hv = 0.0
            prev_y = ref[1]
            for p in pts:
                if p[1] < prev_y:
                    hv += (ref[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

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
            
            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            if not current_front:
                current_front = list(range(n))
            fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
                current_front = next_front
            
            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F_arr[indices, m]
                mn, mx = vals.min(), vals.max()
                if mx == mn:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (mx - mn)
            return cd

        # Population size: balance between exploration and exploitation
        pop_size = min(40, max(15, budget // 12))
        used = 0

        # Initialize with structured sampling of x1 (critical for f1)
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            # Spread x1 across [lo[0], hi[0]] to explore the Pareto front
            frac = i / max(pop_size - 1, 1)
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * frac
            # Other dimensions: random
            for j in range(1, dim):
                X[i, j] = lo[j] + (hi[j] - lo[j]) * rng.random()

        F = np.array([problem(x) for x in X])
        used += pop_size

        # Main evolution loop
        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)
            if not fronts:
                break

            # Build next generation via NSGA-II selection
            next_indices = []
            for front in fronts:
                if len(next_indices) + len(front) <= pop_size:
                    next_indices.extend(front)
                else:
                    need = pop_size - len(next_indices)
                    if need <= 0:
                        break
                    cd = crowding_distance(F, front)
                    order = np.argsort(-cd)
                    next_indices.extend([front[i] for i in order[:need]])
                    break
                if len(next_indices) >= pop_size:
                    break

            # Fill remaining with random selection
            if len(next_indices) < pop_size:
                remaining = pop_size - len(next_indices)
                all_idx = list(range(pop_size))
                extra = [i for i in all_idx if i not in next_indices]
                rng.shuffle(extra)
                next_indices.extend(extra[:remaining])

            next_indices = next_indices[:pop_size]

            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                if used >= budget or time.time() > deadline:
                    break

                # Binary tournament selection
                def tournament_select():
                    i = rng.integers(len(next_indices))
                    j = rng.integers(len(next_indices))
                    idx_i = next_indices[i]
                    idx_j = next_indices[j]
                    
                    # Find ranks
                    rank_i = next(k for k, fr in enumerate(fronts) if idx_i in fr)
                    rank_j = next(k for k, fr in enumerate(fronts) if idx_j in fr)
                    
                    if rank_i != rank_j:
                        return idx_i if rank_i < rank_j else idx_j
                    
                    # Same rank: use crowding distance
                    front_idx = rank_i
                    front = fronts[front_idx]
                    cd = crowding_distance(F, front)
                    pos_i = front.index(idx_i)
                    pos_j = front.index(idx_j)
                    return idx_i if cd[pos_i] > cd[pos_j] else idx_j

                p1_idx = tournament_select()
                p2_idx = tournament_select()
                x1 = X[p1_idx]
                x2 = X[p2_idx]

                # Crossover: blend with random weight
                if rng.random() < 0.8:
                    alpha = rng.random()
                    child_x = alpha * x1 + (1 - alpha) * x2
                else:
                    child_x = x1.copy()

                # Adaptive mutation: scale decreases over time
                progress = used / max(budget, 1)
                base_scale = 0.12 * (1.0 - 0.6 * progress)

                for j in range(dim):
                    if rng.random() < 0.5:  # Mutate each dim with prob 0.5
                        if j == 0:
                            # x1: finer steps for refinement
                            step = base_scale * 0.4 * (hi[0] - lo[0]) * rng.standard_normal()
                        else:
                            step = base_scale * (hi[j] - lo[j]) * rng.standard_normal()
                        child_x[j] += step

                child_x = np.clip(child_x, lo, hi)
                fc = problem(child_x)
                offspring_X.append(child_x)
                offspring_F.append(fc)
                used += 1

            if not offspring_X:
                break

            # Combine parents and offspring, select next population
            # Use hypervolume-guided replacement: keep points that contribute most to HV
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            combined_n = len(combined_F)

            # Find non-dominated points in combined set
            combined_fronts = fast_non_dominated_sort(combined_F)
            
            # Select top pop_size points using rank + crowding
            selected = []
            for front in combined_fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    need = pop_size - len(selected)
                    if need <= 0:
                        break
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(-cd)
                    selected.extend([front[i] for i in order[:need]])
                    break
                if len(selected) >= pop_size:
                    break

            if len(selected) < pop_size:
                remaining = pop_size - len(selected)
                all_combined = list(range(combined_n))
                extra = [i for i in all_combined if i not in selected]
                rng.shuffle(extra)
                selected.extend(extra[:remaining])

            selected = selected[:pop_size]

            # Update population
            X = combined_X[selected]
            F = combined_F[selected]

        return X
# EVOLVE-BLOCK-END

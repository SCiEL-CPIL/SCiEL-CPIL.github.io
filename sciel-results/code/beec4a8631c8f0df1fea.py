import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        bounds_range = hi - lo

        all_X = []
        all_F = []
        eval_count = 0

        ref_point = np.array([3000.0, 0.06])

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count += 1
            return f



        def get_pareto_front_2d(F):
            """Efficient 2D non-dominated sorting. O(N log N)."""
            if len(F) == 0:
                return np.array([], dtype=int)
            # Sort by first objective
            sorted_idx = np.argsort(F[:, 0])
            F_sorted = F[sorted_idx]
            
            # A point is non-dominated if its second objective is strictly lower than all previous points' second objectives
            min_f2 = np.inf
            nd_indices = []
            for i in range(len(F_sorted)):
                if F_sorted[i, 1] < min_f2:
                    nd_indices.append(sorted_idx[i])
                    min_f2 = F_sorted[i, 1]
            return np.array(nd_indices, dtype=int)

        def calc_hv_2d(F_list, ref):
            """Calculate exact 2D hypervolume."""
            if not F_list:
                return 0.0
            F = np.array(F_list)
            # Filter points strictly better than ref
            mask = (F[:, 0] < ref[0]) & (F[:, 1] < ref[1])
            if not np.any(mask):
                return 0.0
            F = F[mask]
            
            nd_idx = get_pareto_front_2d(F)
            F_nd = F[nd_idx]
            if len(F_nd) == 0:
                return 0.0
            
            # F_nd is already sorted by f1 from get_pareto_front_2d
            hv = 0.0
            for i in range(len(F_nd)):
                f1 = F_nd[i, 0]
                f2 = F_nd[i, 1]
                next_f1 = F_nd[i+1, 0] if i + 1 < len(F_nd) else ref[0]
                width = next_f1 - f1
                height = ref[1] - f2
                hv += width * height
            return hv

        def hv_contribution_2d(point, nd_points_sorted, ref):
            """
            Calculate the hypervolume contribution of a specific point.
            nd_points_sorted must be sorted by f1 ascending and contain only non-dominated points.
            """
            p1, p2 = point
            if p1 >= ref[0] or p2 >= ref[1]:
                return 0.0
            
            # Find the position of this point in the sorted list
            # We assume point is in nd_points_sorted
            idx = -1
            for i in range(len(nd_points_sorted)):
                if np.allclose(nd_points_sorted[i], point):
                    idx = i
                    break
            if idx == -1:
                # If not found exactly (shouldn't happen), return 0
                return 0.0
            
            next_f1 = nd_points_sorted[idx+1, 0] if idx + 1 < len(nd_points_sorted) else ref[0]
            width = next_f1 - p1
            height = ref[1] - p2
            return width * height

        # --- Phase 1: Initialization ---
        # Use a 3x3x3x3 grid (81 points) and select 40 of them for better coverage
        n_init = 40
        grid_vals = [np.linspace(lo[i], hi[i], 3) for i in range(self.dim)]
        grid_points = []
        for x0 in grid_vals[0]:
            for x1 in grid_vals[1]:
                for x2 in grid_vals[2]:
                    for x3 in grid_vals[3]:
                        grid_points.append(np.array([x0, x1, x2, x3]))
        
        # Shuffle and take first n_init
        np.random.shuffle(grid_points)
        init_points = grid_points[:n_init]
        
        if len(init_points) < n_init:
            n_rand = n_init - len(init_points)
            rand_points = lo + bounds_range * np.random.random((n_rand, self.dim))
            init_points = np.vstack([init_points, rand_points])

        for x in init_points:
            if time.time() > deadline or eval_count >= self.budget:
                break
            evaluate(x)

        if eval_count >= self.budget:
            return all_X

        # --- Phase 2: Differential Evolution with HV-guided selection ---
        
        # Maintain an archive of non-dominated points
        # We will use the full set of evaluated points for parent selection based on HV contribution
        # but limit the population size for DE to ~50 to keep computation tractable
        
        pop_size = 40
        F_mutation = 0.8
        CR = 0.9
        no_improve_count = 0
        best_hv = calc_hv_2d(all_F, ref_point)
        
        # Initialize population indices from current evaluations
        # Pick the top pop_size points by HV contribution
        def get_top_population_indices(size):
            if len(all_F) < size:
                return list(range(len(all_F)))
            
            # Get non-dominated points
            nd_idx = get_pareto_front_2d(np.array(all_F))
            
            # Filter strictly better than ref
            valid_nd = []
            for i in nd_idx:
                f = all_F[i]
                if f[0] < ref_point[0] and f[1] < ref_point[1]:
                    valid_nd.append(i)
            
            if len(valid_nd) == 0:
                # Fallback to random
                return np.random.choice(len(all_F), size, replace=False).tolist()
            
            # Calculate HV contribution for each valid ND point
            # Sort valid_nd by f1
            valid_nd_sorted = sorted(valid_nd, key=lambda i: all_F[i][0])
            
            contribs = {}
            F_sorted_nd = np.array([all_F[i] for i in valid_nd_sorted])
            
            for k in range(len(valid_nd_sorted)):
                idx = valid_nd_sorted[k]
                p1, p2 = F_sorted_nd[k]
                next_f1 = F_sorted_nd[k+1, 0] if k + 1 < len(F_sorted_nd) else ref_point[0]
                width = next_f1 - p1
                height = ref_point[1] - p2
                contribs[idx] = width * height
            
            # Sort by contribution descending
            sorted_indices = sorted(contribs.keys(), key=lambda i: contribs[i], reverse=True)
            
            # Take top size
            selected = sorted_indices[:size]
            
            # If we don't have enough, fill with random
            if len(selected) < size:
                remaining = size - len(selected)
                available = [i for i in range(len(all_F)) if i not in selected]
                if available:
                    rand_fill = np.random.choice(available, min(remaining, len(available)), replace=False)
                    selected.extend(rand_fill.tolist())
            
            return selected

        pop_indices = get_top_population_indices(pop_size)

        while eval_count < self.budget and time.time() < deadline:
            # Refresh population periodically
            if len(pop_indices) < pop_size or eval_count % 15 == 0:
                pop_indices = get_top_population_indices(pop_size)
                
            if len(pop_indices) < 3:
                break

            # Adaptive parameters
            if no_improve_count > 5:
                F_mutation *= 0.9
                no_improve_count = 0
                if F_mutation < 0.1:
                    F_mutation = 0.1
            
            # Pick parents
            parents = np.random.choice(pop_indices, 3, replace=False)
            x1 = all_X[parents[0]]
            x2 = all_X[parents[1]]
            x3 = all_X[parents[2]]

            # Mutation
            v = x1 + F_mutation * (x2 - x3)
            
            # Crossover
            u = np.random.random(self.dim) < CR
            y = np.where(u, v, x1)
            
            # Clip
            y = np.clip(y, lo, hi)

            fy = evaluate(y)
            if fy is None:
                break

            # Update best HV
            current_hv = calc_hv_2d(all_F, ref_point)
            if current_hv > best_hv + 1e-9:
                best_hv = current_hv
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Selection
            target_idx = np.random.choice(pop_indices)
            ft = all_F[target_idx]
            
            # Check if fy dominates ft
            if fy[0] <= ft[0] and fy[1] <= ft[1] and (fy[0] < ft[0] or fy[1] < ft[1]):
                pop_indices[pop_indices.index(target_idx)] = eval_count - 1
            elif not (ft[0] <= fy[0] and ft[1] <= fy[1] and (ft[0] < fy[0] or ft[1] < fy[1])):
                # Non-dominated
                if np.random.random() < 0.5:
                    pop_indices[pop_indices.index(target_idx)] = eval_count - 1

        return all_X

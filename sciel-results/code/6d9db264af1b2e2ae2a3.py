import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 305.0  # Safety margin (evaluator kills at 330s)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3000.0, 0.06], dtype=float)

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        def halton_sequence(n, dim):
            """Generate Halton sequence points in [0,1]^dim using first 'dim' primes."""
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            points = np.zeros((n, dim))
            for d in range(dim):
                base = primes[d]
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    i_val = i + 1  # Start from 1 to avoid 0
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    points[i, d] = r
            return points

        def compute_hv(F, ref_pt):
            """Compute hypervolume of a set of 2D objective vectors relative to ref_pt.
            Only points strictly better than ref in all objectives count."""
            if len(F) == 0:
                return 0.0
            # Filter: strictly better than ref in all objectives
            mask = np.all(F < ref_pt, axis=1)
            F_valid = F[mask]
            if len(F_valid) == 0:
                return 0.0
            # For 2D, sort by first objective and compute HV
            F_valid = F_valid[np.argsort(F_valid[:, 0])]
            hv = 0.0
            prev_f2 = ref_pt[1]
            for i in range(len(F_valid)):
                f1, f2 = F_valid[i]
                if f2 < prev_f2:
                    hv += (ref_pt[0] - f1) * (prev_f2 - f2)
                    prev_f2 = f2
                # If f2 >= prev_f2, this point is dominated in the HV sense
            return hv

        def hv_contribution(F, ref_pt):
            """Compute HV contribution of each point (hypervolume that disappears if point is removed).
            Returns array of same length as F."""
            n = len(F)
            if n == 0:
                return np.array([])
            # Filter to valid points (strictly better than ref)
            valid_mask = np.all(F < ref_pt, axis=1)
            contrib = np.zeros(n)
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                return contrib
            
            F_valid = F[valid_indices]
            # Sort by first objective
            order = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[order]
            
            # Compute total HV
            total_hv = 0.0
            prev_f2 = ref_pt[1]
            # Store the "width" contribution of each point
            widths = np.zeros(len(F_sorted))
            for i in range(len(F_sorted)):
                f1, f2 = F_sorted[i]
                if f2 < prev_f2:
                    widths[i] = (ref_pt[0] - f1) * (prev_f2 - f2)
                    total_hv += widths[i]
                    prev_f2 = f2
                else:
                    widths[i] = 0.0
            
            # Now compute contribution of each point
            # For 2D, the contribution of point i is the area that only it covers
            # Sort by f1 ascending; for each point, its contribution is the rectangle
            # from (f1_i, f2_i) to (f1_{i+1} or ref, f2_i) that isn't covered by others
            # Actually, simpler: contribution of point i = HV of all points - HV of all points except i
            # But that's O(n²). For 2D, we can do it more efficiently.
            
            # Efficient 2D HV contribution:
            # After sorting by f1, point i contributes the area:
            # (min(f1_{i+1}, ref) - f1_i) * (max(f2_i - f2_{i-1}, 0))  ... not quite
            # Let's use the standard approach:
            # For sorted points (by f1 asc), the contribution of point i is:
            # width_i = min(f1_{i+1}, ref_0) - f1_i  (with f1_{n} = ref_0)
            # height_i = max(f2_i - f2_{i-1}, 0)    (with f2_{-1} = ref_1)
            # But this isn't exactly right either. Let me use the "step" method.
            
            # Actually, the cleanest way for 2D:
            # Contribution of point i = area of the box [f1_i, next_f1) x [f2_i, next_f2_above)
            # where next_f1 is f1 of next point with smaller f2, or ref_0
            # This is getting complex. Let's just do O(n²) for n<=200.
            
            for idx in valid_indices:
                # Compute HV without this point
                mask_no_i = valid_mask.copy()
                mask_no_i[idx] = False
                F_no_i = F[mask_no_i]
                if len(F_no_i) == 0:
                    contrib[idx] = 0.0
                    continue
                F_no_i_sorted = F_no_i[np.argsort(F_no_i[:, 0])]
                hv_no_i = 0.0
                prev_f2 = ref_pt[1]
                for j in range(len(F_no_i_sorted)):
                    f1, f2 = F_no_i_sorted[j]
                    if f2 < prev_f2:
                        hv_no_i += (ref_pt[0] - f1) * (prev_f2 - f2)
                        prev_f2 = f2
                contrib[idx] = max(0.0, total_hv - hv_no_i)
            
            return contrib

        def non_dominated_sort(F):
            """Fast non-dominated sort for 2D objectives."""
            n = len(F)
            if n == 0:
                return []
            # For 2D, a point is dominated if there exists another point with both objectives <=
            # We can sort by first objective and use a running minimum of second objective
            order = np.argsort(F[:, 0])
            F_sorted = F[order]
            
            # Find non-dominated: scan in order of increasing f1, keep track of min f2 seen
            nondom_mask = np.zeros(n, dtype=bool)
            min_f2 = np.inf
            for i in range(n):
                if F_sorted[i, 1] < min_f2:
                    nondom_mask[order[i]] = True
                    min_f2 = F_sorted[i, 1]
            
            # Get indices of non-dominated points
            front0 = np.where(nondom_mask)[0]
            if len(front0) == 0:
                # All dominated? Shouldn't happen, but handle
                front0 = np.array([order[0]])
                nondom_mask[order[0]] = True
            
            # For the remaining, do a simple non-dominated sort
            remaining = np.where(~nondom_mask)[0]
            fronts = [list(front0)]
            
            # Simple iterative non-dominated sort for remaining
            remaining_set = set(remaining.tolist())
            while remaining_set:
                # Find non-dominated within remaining
                rem_list = list(remaining_set)
                rem_F = F[rem_list]
                # Sort by first objective
                rem_order = np.argsort(rem_F[:, 0])
                rem_F_sorted = rem_F[rem_order]
                
                front_in_rem = []
                min_f2 = np.inf
                for i in range(len(rem_list)):
                    orig_idx = rem_list[rem_order[i]]
                    if rem_F_sorted[i, 1] < min_f2:
                        front_in_rem.append(orig_idx)
                        min_f2 = rem_F_sorted[i, 1]
                
                if not front_in_rem:
                    front_in_rem = [rem_list[0]]
                
                fronts.append(front_in_rem)
                for idx in front_in_rem:
                    remaining_set.discard(idx)
            
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            if not front_indices:
                return {}
            m = len(front_indices)
            if m <= 2:
                return {i: float('inf') for i in front_indices}
            
            dist = {i: 0.0 for i in front_indices}
            for k in range(n_obj):
                sorted_front = sorted(front_indices, key=lambda i: F[i][k])
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                for i in range(1, m - 1):
                    dist[sorted_front[i]] += (F[sorted_front[i+1]][k] - F[sorted_front[i-1]][k]) / f_range
            return dist

        # --- Phase 1: Halton initialization (60% of budget) ---
        n_init = int(budget * 0.6)  # 120 points
        halton_pts = halton_sequence(n_init, dim)
        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * halton_pts[i]
            evaluate(x)

        if used == 0:
            return all_X

        X_arr = np.array(all_X)
        F_arr = np.array(all_F)
        
        # --- Phase 2: HV-guided DE for remaining budget ---
        N_pop = 24  # Population size
        CR = 0.7
        F_base = 0.75

        # Initialize DE population from current Pareto front + diverse points
        fronts = non_dominated_sort(F_arr)
        if not fronts:
            return all_X
        
        pareto_indices = fronts[0]
        
        # Select diverse points from Pareto front using HV contribution
        if len(pareto_indices) >= N_pop:
            # Compute HV contributions and select top contributors
            contrib = hv_contribution(F_arr[pareto_indices], ref)
            # Select top N_pop by contribution, but ensure diversity
            top_idx = np.argsort(contrib)[::-1][:N_pop]
            pop_indices = pareto_indices[top_idx]
        else:
            pop_indices = np.array(pareto_indices)
            # Fill with random points if needed
            n_needed = N_pop - len(pop_indices)
            for _ in range(n_needed):
                if used >= budget or time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.random(dim)
                f = evaluate(x)
                pop_indices = np.append(pop_indices, len(all_X) - 1)
        
        if len(pop_indices) < 2:
            return all_X
        
        pop_X = X_arr[pop_indices].copy()
        pop_F = F_arr[pop_indices].copy()
        N_pop = len(pop_X)
        
        # Main DE loop
        while used < budget and time.time() < deadline:
            # Adaptive F
            F_factor = F_base * (0.5 + 0.5 * np.random.rand())
            
            offspring_X = []
            offspring_F = []
            
            # Generate offspring
            for i in range(N_pop):
                if used >= budget or time.time() > deadline:
                    break
                
                if N_pop < 4:
                    trial = pop_X[i] + 0.1 * (hi - lo) * np.random.randn(dim)
                    trial = np.clip(trial, lo, hi)
                else:
                    candidates = [j for j in range(N_pop) if j != i]
                    r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                    v = pop_X[r1] + F_factor * (pop_X[r2] - pop_X[r3])
                    
                    # Binomial crossover
                    trial = np.zeros(dim)
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.rand() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop_X[i][j]
                    trial = np.clip(trial, lo, hi)
                
                f = evaluate(trial)
                offspring_X.append(trial)
                offspring_F.append(f)
            
            if not offspring_X:
                break
                
            # Combine population and offspring
            combined_X = np.vstack([pop_X, np.array(offspring_X)])
            combined_F = np.vstack([pop_F, np.array(offspring_F)])
            
            # Non-dominated sort
            fronts = non_dominated_sort(combined_F)
            if not fronts:
                break
                
            rank_map = {}
            for rank, front in enumerate(fronts):
                for idx in front:
                    rank_map[idx] = rank
            
            crowd_dists = {}
            for front in fronts:
                dists = crowding_distance(combined_F, front)
                crowd_dists.update(dists)
            
            # Select next population
            new_pop_X = []
            new_pop_F = []
            
            for rank, front in enumerate(fronts):
                if len(new_pop_X) + len(front) <= N_pop:
                    for idx in front:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                else:
                    remaining = N_pop - len(new_pop_X)
                    sorted_front = sorted(front, key=lambda i: crowd_dists.get(i, 0.0), reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                    break
            
            if not new_pop_X:
                break
            
            pop_X = np.array(new_pop_X)
            pop_F = np.array(new_pop_F)
            
            # Update population to focus on high HV-contribution regions
            if used < budget - 5 and time.time() < deadline - 5:
                # Recompute HV contributions for current population
                if len(pop_F) >= 2:
                    contrib = hv_contribution(pop_F, ref)
                    # Sort by contribution and keep top N_pop
                    if len(pop_F) > N_pop:
                        top_idx = np.argsort(contrib)[::-1][:N_pop]
                        pop_X = pop_X[top_idx]
                        pop_F = pop_F[top_idx]

        return all_X

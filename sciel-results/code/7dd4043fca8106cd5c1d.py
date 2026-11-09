"""Multi-objective optimizer on WFG4 with hypervolume-based island model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-based island model optimizer for WFG4 multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _fast_non_dominated_sort(F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    nd[i] += 1
        front0 = [i for i in range(n) if nd[i] == 0]
        fronts.append(front0)
        while front0:
            next_front = []
            for i in front0:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front0 = next_front
        return fronts

    @staticmethod
    def _crowding_distance(F, indices):
        """Compute crowding distance for points at given indices."""
        m = len(indices)
        if m <= 2:
            return np.full(m, np.inf)
        dist = np.zeros(m)
        n_obj = F.shape[1]
        for k in range(n_obj):
            vals = F[indices, k]
            v_min, v_max = vals.min(), vals.max()
            if v_max == v_min:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for idx in range(1, m - 1):
                dist[order[idx]] += (vals[order[idx + 1]] - vals[order[idx - 1]]) / (v_max - v_min)
        return dist

    @staticmethod
    def _hypervolume_3d(points, ref):
        """Compute exact hypervolume for 3D points relative to reference point.
        Points that are not strictly better than ref in all objectives are ignored."""
        # Filter points strictly better than ref
        valid = np.all(points < ref, axis=1)
        pts = points[valid]
        if len(pts) == 0:
            return 0.0
        
        # Sort by first objective
        pts = pts[np.argsort(pts[:, 0])]
        n = len(pts)
        
        # For each point, compute the hypervolume contribution
        # In 3D: for point i, its contribution is (ref[0] - pts[i,0]) * area in 2D
        # where area is the rectangle in (f1, f2) plane not covered by later points
        hv = 0.0
        for i in range(n):
            # The x-extent for this point
            dx = ref[0] - pts[i, 0]
            if i + 1 < n:
                # Next point's f0
                dx = min(dx, pts[i+1, 0] - pts[i, 0])
                # Actually we need to be more careful
                pass
        
        # Better approach: sweep line
        # Sort by f0 ascending
        # For each point, the contribution is (ref[0] - f0[i]) * (max of (ref[1]-f1, ref[2]-f2) not covered)
        # This is getting complex. Let's use a simpler but exact method for 3D.
        
        # Exact 3D hypervolume via inclusion-exclusion is complex.
        # Use the standard algorithm: sort by one objective, then for each point
        # compute the 2D hypervolume of the "skyline" of remaining points.
        
        # Sort by f0
        order = np.argsort(pts[:, 0])
        pts_sorted = pts[order]
        
        hv = 0.0
        for i in range(n):
            f0 = pts_sorted[i, 0]
            # The width in f0 direction
            if i + 1 < n:
                w0 = pts_sorted[i+1, 0] - f0
            else:
                w0 = ref[0] - f0
            
            # For the 2D part: consider all points j >= i
            # The 2D hypervolume of points (f1[j], f2[j]) for j >= i with ref (ref[1], ref[2])
            # But only the part where f0 is in [f0_i, f0_{i+1})
            # Actually, since we sorted by f0, for points j >= i, they all have f0 >= f0_i
            # The 2D hypervolume contribution is the area in (f1, f2) plane
            # of the region dominated by at least one of points i..n-1, relative to ref[1:], ref[2:]
            
            # Collect all points from i onwards
            sub_pts = pts_sorted[i:, 1:]
            sub_ref = ref[1:]
            
            # 2D hypervolume
            hv_2d = Optimizer._hypervolume_2d(sub_pts, sub_ref)
            hv += w0 * hv_2d
        
        return hv

    @staticmethod
    def _hypervolume_2d(points, ref):
        """Compute exact 2D hypervolume using O(n log n) sweep line."""
        # Filter valid points
        valid = np.all(points < ref, axis=1)
        pts = points[valid]
        if len(pts) == 0:
            return 0.0
        
        # Sort by f0
        pts = pts[np.argsort(pts[:, 0])]
        n = len(pts)
        
        # Pre-compute suffix minimum of f1 for O(1) lookup
        suffix_min_f1 = np.full(n, ref[1])
        for i in range(n - 1, -1, -1):
            if pts[i, 1] < ref[1]:
                suffix_min_f1[i] = min(pts[i, 1], suffix_min_f1[i + 1]) if i + 1 < n else pts[i, 1]
            else:
                suffix_min_f1[i] = suffix_min_f1[i + 1] if i + 1 < n else ref[1]
        
        hv = 0.0
        for i in range(n):
            f0 = pts[i, 0]
            if i + 1 < n:
                w0 = pts[i+1, 0] - f0
            else:
                w0 = ref[0] - f0
            
            min_f1 = suffix_min_f1[i]
            if min_f1 < ref[1]:
                h1 = ref[1] - min_f1
            else:
                h1 = 0.0
            
            hv += w0 * h1
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Island model: partition into sub-regions based on objective weights
        # For 3 objectives, use weight vectors pointing to different regions of the Pareto front
        n_islands = 8
        # Define weight vectors for each island (normalized to sum to 1)
        # These guide the search toward different parts of the Pareto front
        # More uniform coverage of the simplex
        weights = np.array([
            [0.8, 0.1, 0.1],  # Strongly emphasize objective 0
            [0.1, 0.8, 0.1],  # Strongly emphasize objective 1
            [0.1, 0.1, 0.8],  # Strongly emphasize objective 2
            [0.5, 0.3, 0.2],  # Favor obj 0, moderate obj 1
            [0.2, 0.5, 0.3],  # Favor obj 1, moderate obj 2
            [0.3, 0.2, 0.5],  # Favor obj 2, moderate obj 0
            [0.4, 0.4, 0.2],  # Balanced, slightly favor obj 2
            [0.2, 0.4, 0.4],  # Balanced, slightly favor obj 1
        ])

        # Population per island - smaller populations for more islands
        pop_per_island = max(8, budget // (n_islands * 6))
        total_pop = pop_per_island * n_islands

        # Initialize each island
        islands = []
        for k in range(n_islands):
            # Initialize with points biased toward the weight direction
            # Use a mixture of random and weight-directed initialization
            X_k = lo + (hi - lo) * rng.random((pop_per_island, dim))
            # Bias some points toward the weight direction
            n_biased = pop_per_island // 3
            if n_biased > 0:
                # Create points that are "good" in the weighted sense
                # By scaling toward the lower bounds in weighted proportion
                for i in range(n_biased):
                    scale = rng.random() * 0.5
                    X_k[i] = lo + scale * (hi - lo)
            
            F_k = np.array([problem(x) for x in X_k])
            islands.append({'X': X_k, 'F': F_k, 'weights': weights[k]})

        used = total_pop

        # DE parameters per island (varying mutation intensity)
        F_CR_base = 0.7
        CR = 0.85
        K_base = 0.6

        # Migration interval - more frequent migration for better information sharing
        migrate_every = 20

        generation = 0

        while used < budget:
            if time.time() > deadline:
                break

            generation += 1

            for k in range(n_islands):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                island = islands[k]
                X_k = island['X']
                F_k = island['F']
                w = island['weights']
                pop_k = len(X_k)

                # Non-dominated sorting within island
                fronts = self._fast_non_dominated_sort(F_k)
                front0 = fronts[0] if fronts else list(range(pop_k))

                # Selection: pick parents from best front with diversity consideration
                if len(front0) >= 3:
                    parent_pool = front0
                elif len(front0) >= 2:
                    parent_pool = front0 + (fronts[1] if len(fronts) > 1 else [])
                else:
                    parent_pool = list(range(pop_k))

                # Pick 3 distinct parents for DE mutation, preferring diverse points
                if len(parent_pool) >= 3:
                    # Use crowding distance to select diverse parents
                    if len(parent_pool) > 3:
                        cd = self._crowding_distance(F_k, parent_pool)
                        # Select top 3 by crowding distance for diversity
                        top_diverse = np.argsort(cd)[-3:]
                        p1, p2, p3 = rng.choice(top_diverse, size=3, replace=False)
                    else:
                        p1, p2, p3 = rng.choice(parent_pool, size=3, replace=False)
                else:
                    p1 = rng.choice(parent_pool)
                    p2 = rng.choice(parent_pool)
                    p3 = rng.choice(parent_pool)

                # DE mutation with weight-specific intensity
                # Islands emphasizing a particular objective use higher mutation rate
                # for that objective's direction
                K_k = K_base * (1.0 + 0.3 * max(w))
                v = X_k[p1] + K_k * (X_k[p2] - X_k[p3])

                # DE crossover with binomial distribution
                jrand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = X_k[p1][j]

                trial = np.clip(trial, lo, hi)
                
                # Add small random perturbation for exploration
                if rng.random() < 0.1:
                    noise = rng.normal(0, 0.05, dim)
                    trial = np.clip(trial + noise, lo, hi)

                # Evaluate
                ft = problem(trial)
                used += 1

                # Weighted sum for selection bias
                ws_trial = np.dot(ft, w)
                ws_existing = np.dot(F_k, w)

                # Decide replacement
                # Check if trial dominates any existing point
                dominated_by_trial = []
                for i in range(pop_k):
                    if np.all(ft <= F_k[i]) and np.any(ft < F_k[i]):
                        dominated_by_trial.append(i)

                if dominated_by_trial:
                    # Replace the one with worst weighted sum among dominated
                    worst_idx = dominated_by_trial[np.argmax(ws_existing[dominated_by_trial])]
                    X_k[worst_idx] = trial
                    F_k[worst_idx] = ft
                else:
                    # Check if trial is dominated by existing points
                    dominates_trial = []
                    for i in range(pop_k):
                        if np.all(F_k[i] <= ft) and np.any(F_k[i] < ft):
                            dominates_trial.append(i)

                    if dominates_trial:
                        # Trial is dominated; only replace if it has better weighted sum
                        # than the worst in the island
                        worst_idx = int(np.argmax(ws_existing))
                        if ws_trial < ws_existing[worst_idx]:
                            X_k[worst_idx] = trial
                            F_k[worst_idx] = ft
                    else:
                        # Non-dominated: replace worst by weighted sum if trial is better
                        worst_idx = int(np.argmax(ws_existing))
                        if ws_trial < ws_existing[worst_idx]:
                            X_k[worst_idx] = trial
                            F_k[worst_idx] = ft
                        elif len(fronts) > 1:
                            # Also try to maintain diversity: replace worst crowding in worst front
                            worst_front = fronts[-1]
                            if worst_front and len(worst_front) > 1:
                                cd = self._crowding_distance(F_k, worst_front)
                                worst_pos = np.argmin(cd)
                                candidate_idx = worst_front[worst_pos]
                                # Only replace if trial is non-dominated and improves spread
                                # Simple heuristic: replace if trial has lower weighted sum
                                if ws_trial < ws_existing[candidate_idx]:
                                    X_k[worst_idx] = trial
                                    F_k[worst_idx] = ft

                # Update island
                island['X'] = X_k
                island['F'] = F_k

            # Migration: every migrate_every generations, share best points between all islands
            if generation % migrate_every == 0:
                # Collect all best points from each island
                best_points = []
                for k in range(n_islands):
                    F_k = islands[k]['F']
                    w_k = islands[k]['weights']
                    ws_k = np.dot(F_k, w_k)
                    best_k = int(np.argmin(ws_k))
                    best_points.append((k, best_k, ws_k[best_k]))
                
                # Sort by weighted sum (lower is better)
                best_points.sort(key=lambda x: x[2])
                
                # Share top 3 best points with all other islands
                for i in range(min(3, len(best_points))):
                    src_island, src_idx, _ = best_points[i]
                    src_X = islands[src_island]['X'][src_idx].copy()
                    src_F = islands[src_island]['F'][src_idx].copy()
                    
                    for k in range(n_islands):
                        if k == src_island:
                            continue
                        F_k = islands[k]['F']
                        w_k = islands[k]['weights']
                        ws_k = np.dot(F_k, w_k)
                        worst_k = int(np.argmax(ws_k))
                        
                        # Replace worst if shared point is better
                        if np.all(src_F <= F_k[worst_k]) or np.dot(src_F, w_k) < ws_k[worst_k]:
                            islands[k]['X'][worst_k] = src_X
                            islands[k]['F'][worst_k] = src_F

        # Return the best set of points found
        # Combine all islands, remove dominated points, and return
        all_X = []
        all_F = []
        for island in islands:
            all_X.append(island['X'])
            all_F.append(island['F'])
        
        all_X = np.vstack(all_X)
        all_F = np.vstack(all_F)
        
        # Remove dominated points to return a diverse set
        if len(all_F) > 50:
            fronts = self._fast_non_dominated_sort(all_F)
            # Keep only first 2 fronts
            keep_indices = []
            for i in range(min(2, len(fronts))):
                keep_indices.extend(fronts[i])
            if keep_indices:
                all_X = all_X[keep_indices]
        
        return all_X
# EVOLVE-BLOCK-END

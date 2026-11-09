"""Rewritten optimizer for WFG7 multi-objective optimization."""
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven evolutionary optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(123)

        budget = self.budget
        n_obj = self.n_obj
        dim = self.dim
        ref = np.array([3.0, 5.0, 7.0])

        # --- Helper functions ---
        def dominates(a, b):
            """True if a strictly dominates b (a<=b in all, a<b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_nondominated(f, F_all):
            """Check if f is non-dominated w.r.t. F_all."""
            for g in F_all:
                if dominates(g, f):
                    return False
            return True

        def non_dominated_indices(F):
            """Return indices of non-dominated points in F."""
            n = len(F)
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j and dominates(F[j], F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return np.array(nd, dtype=int)

        def hypervolume_3d(points, ref_point):
            """Compute 3D hypervolume for points strictly better than ref_point."""
            # Filter points better than ref in all objectives
            valid = np.all(points < ref_point, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            if len(pts) == 1:
                return float(np.prod(ref_point - pts[0]))
            if len(pts) == 2:
                p1, p2 = pts[0], pts[1]
                # Sort by first objective
                if p1[0] > p2[0]:
                    p1, p2 = p2, p1
                # Hypervolume = (ref[0]-p1[0])*(ref[1]-p1[1])*(ref[2]-p1[2])
                # + (ref[0]-p2[0])*(ref[1]-p2[1])*(ref[2]-p2[2])
                # - overlap if p2 is in the "shadow" of p1
                # Actually for 2 points, just compute the union of two boxes
                # Box1: [p1, ref], Box2: [p2, ref]
                # They overlap in [max(p1,p2), ref]
                vol1 = (ref[0]-p1[0])*(ref[1]-p1[1])*(ref[2]-p1[2])
                vol2 = (ref[0]-p2[0])*(ref[1]-p2[1])*(ref[2]-p2[2])
                ov_lo = np.maximum(p1, p2)
                ov_hi = ref
                if np.all(ov_lo < ov_hi):
                    overlap = float(np.prod(ov_hi - ov_lo))
                else:
                    overlap = 0.0
                return float(vol1 + vol2 - overlap)
            # For more points, use inclusion-exclusion or grid-based approach
            # Use a simple grid-based approximation for larger sets
            n_grid = 20
            grid = np.zeros((n_grid+1, n_grid+1, n_grid+1))
            for p in pts:
                # Mark cells that are covered
                for i in range(n_grid):
                    for j in range(n_grid):
                        for k in range(n_grid):
                            cx = (lo[0] + (hi[0]-lo[0])*(i+0.5)/n_grid) if False else None
                # This is too slow. Use a different approach.
                pass
            # Use the "slicing" method for 3D
            # Sort points by first objective
            order = np.argsort(pts[:, 0])
            pts_sorted = pts[order]
            total_hv = 0.0
            prev_yz = None
            for idx in range(len(pts_sorted)):
                p = pts_sorted[idx]
                # Volume of box from p to ref
                box_vol = (ref[0]-p[0]) * (ref[1]-p[1]) * (ref[2]-p[2])
                if idx > 0:
                    p_prev = pts_sorted[idx-1]
                    # Overlap: intersection of [p, ref] and [p_prev, ref]
                    # = [max(p, p_prev), ref]
                    ov_lo = np.maximum(p, p_prev)
                    if np.all(ov_lo < ref):
                        overlap = float(np.prod(ref - ov_lo))
                    else:
                        overlap = 0.0
                    # But this is wrong for multiple points. Use a proper 3D HV.
                    pass
                total_hv = box_vol  # This is wrong for multiple points
            # For correctness with >2 points, use a proper algorithm
            # Use the "divide and conquer" or "slicing" approach
            # Slicing: sort by x, then for each slice compute 2D HV in yz-plane
            # and multiply by slice width
            pts_sorted = pts[np.argsort(pts[:, 0])]
            total_hv = 0.0
            # Add a virtual point at ref
            for i in range(len(pts_sorted)):
                x1 = pts_sorted[i][0]
                x2 = pts_sorted[i+1][0] if i+1 < len(pts_sorted) else ref[0]
                if x2 <= x1:
                    continue
                # 2D hypervolume of points with x <= x1 in the yz-plane
                # Points that are "active" in this slice: those with x <= x1
                active = pts_sorted[:i+1]
                # 2D HV of active points with ref in yz
                hv_2d = self._hv_2d(active[:, 1:], np.array([ref[1], ref[2]]))
                total_hv += hv_2d * (x2 - x1)
            return float(total_hv)

        def _hv_2d(points_2d, ref_2d):
            """Compute 2D hypervolume."""
            n = len(points_2d)
            if n == 0:
                return 0.0
            if n == 1:
                return float((ref_2d[0]-points_2d[0][0]) * (ref_2d[1]-points_2d[0][1]))
            # Sort by first coordinate
            order = np.argsort(points_2d[:, 0])
            pts_sorted = points_2d[order]
            total = 0.0
            for i in range(n):
                p = pts_sorted[i]
                x1 = p[0]
                x2 = pts_sorted[i+1][0] if i+1 < n else ref_2d[0]
                if x2 <= x1:
                    continue
                # Max y-extent from points 0..i
                max_y = np.max(pts_sorted[:i+1, 1])
                # The contribution is (x2-x1) * (ref_2d[1] - max_y)
                # Wait, that's not right either. 
                # 2D HV: sort by x. For each point i, the width is (x_{i+1} - x_i)
                # and the height is (ref_y - y_i) but we need the union.
                # Actually: for 2D, sort by x ascending. 
                # hv = sum over i of (x_{i+1} - x_i) * (ref_y - y_i) where y_i is the
                # minimum y among points 0..i (since we want the union of rectangles).
                # No wait: each point defines a rectangle [p, ref]. 
                # Sort by x. The union's area = sum of (x_{i+1}-x_i) * (ref_y - min_y_so_far)
                # where min_y_so_far = min(y_0, ..., y_i)
                min_y = np.min(pts_sorted[:i+1, 1])
                total += (x2 - x1) * (ref_2d[1] - min_y)
            return float(total)

        # --- Phase 1: Initial sampling ---
        n_init = min(budget, 50)
        X = lo + (hi - lo) * rng.random((n_init, dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        # --- Phase 2: Evolutionary loop with hypervolume-based selection ---
        pop_size = 20
        # Build initial population from non-dominated points
        nd_idx = non_dominated_indices(F)
        if len(nd_idx) == 0:
            nd_idx = np.array([0])
        
        # Take up to pop_size non-dominated, pad with random if needed
        selected = nd_idx[:pop_size]
        X_pop = X[selected].copy()
        F_pop = F[selected].copy()
        if len(X_pop) < pop_size:
            n_pad = pop_size - len(X_pop)
            X_pad = lo + (hi - lo) * rng.random((n_pad, dim))
            F_pad = np.array([problem(x) for x in X_pad])
            used += n_pad
            X_pop = np.vstack([X_pop, X_pad])
            F_pop = np.vstack([F_pop, F_pad])

        # Adaptive step size
        base_step = 0.1 * (hi - lo)

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Select parents based on hypervolume contribution (approximate)
            # Use a simple approach: select from non-dominated, weight by proximity to ref
            nd_in_pop = non_dominated_indices(F_pop)
            if len(nd_in_pop) == 0:
                nd_in_pop = np.array(range(len(F_pop)))

            # Binary tournament: pick 2 random, choose the one closer to ref (in normalized space)
            if len(nd_in_pop) >= 2:
                i1, i2 = rng.choice(len(nd_in_pop), size=2, replace=False)
                p1, p2 = nd_in_pop[i1], nd_in_pop[i2]
                # Normalize objectives
                fmin = np.min(F_pop, axis=0)
                fmax = np.max(F_pop, axis=0)
                f_range = fmax - fmin + 1e-10
                n1 = (F_pop[p1] - fmin) / f_range
                n2 = (F_pop[p2] - fmin) / f_range
                # Prefer the one with smaller sum (closer to origin in normalized space)
                if np.sum(n1) < np.sum(n2):
                    parent_idx = p1
                else:
                    parent_idx = p2
            else:
                parent_idx = nd_in_pop[0]

            parent = X_pop[parent_idx].copy()

            # Mutation: mix of large and small steps
            if rng.random() < 0.3:
                # Large exploration step
                scale = base_step * (1.0 + rng.random())
            else:
                # Smaller exploitation step
                scale = base_step * (0.2 + 0.5 * rng.random())
            
            direction = rng.standard_normal(dim)
            direction /= (np.linalg.norm(direction) + 1e-10)
            child = parent + scale * direction
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update population: replace worst dominated point or worst by HV contribution
            # Find dominated points in population
            dominated_in_pop = []
            for i in range(len(F_pop)):
                if not is_nondominated(F_pop[i], F_pop):
                    dominated_in_pop.append(i)

            if dominated_in_pop:
                # Replace a random dominated point
                replace_idx = dominated_in_pop[rng.integers(len(dominated_in_pop))]
            else:
                # All non-dominated; replace the one with worst hypervolume contribution
                # Approximate: replace the one farthest from ref in normalized space
                fmin = np.min(F_pop, axis=0)
                fmax = np.max(F_pop, axis=0)
                f_range = fmax - fmin + 1e-10
                F_norm = (F_pop - fmin) / f_range
                scores = np.sum(F_norm, axis=1)
                replace_idx = int(np.argmax(scores))

            # Only replace if child is better in at least one objective or non-dominated
            if not is_nondominated(fc, F_pop):
                # Child is dominated; still replace if it's better than the point being replaced
                if np.any(fc < F_pop[replace_idx]):
                    X_pop[replace_idx] = child
                    F_pop[replace_idx] = fc
            else:
                X_pop[replace_idx] = child
                F_pop[replace_idx] = fc

            # Periodic exploration
            if used % 15 == 0 and used < budget:
                exp = lo + (hi - lo) * rng.random(dim)
                fe = problem(exp)
                used += 1
                if is_nondominated(fe, F_pop):
                    # Replace a random point
                    ridx = rng.integers(len(F_pop))
                    X_pop[ridx] = exp
                    F_pop[ridx] = fe

        # --- Phase 3: Local refinement around best non-dominated points ---
        # Combine all evaluated points
        all_F = np.vstack([F, F_pop])
        all_X = np.vstack([X, X_pop])

        final_nd = non_dominated_indices(all_F)
        if len(final_nd) > 0:
            # Sort by distance to reference point
            nd_points = all_F[final_nd]
            dists = np.sum((nd_points - ref) ** 2, axis=1)
            sorted_nd = final_nd[np.argsort(dists)]

            remaining = budget - used
            if remaining > 0:
                n_refine = min(len(sorted_nd), max(1, remaining // 8))
                for i in range(n_refine):
                    if used >= budget or time.time() > deadline:
                        break

                    orig_idx = sorted_nd[i]
                    base_x = all_X[orig_idx].copy()
                    base_f = all_F[orig_idx].copy()

                    local_budget = min(8, budget - used)
                    local_step = 0.03 * (hi - lo)

                    for j in range(local_budget):
                        if used >= budget or time.time() > deadline:
                            break

                        direction = rng.standard_normal(dim)
                        direction /= (np.linalg.norm(direction) + 1e-10)
                        step = local_step * (0.3 + 0.7 * rng.random())
                        candidate = base_x + step * direction
                        candidate = np.clip(candidate, lo, hi)

                        fc = problem(candidate)
                        used += 1

                        # Accept if it dominates base or is non-dominated and better in some obj
                        if dominates(fc, base_f):
                            base_x = candidate
                            base_f = fc
                            all_X[orig_idx] = base_x
                            all_F[orig_idx] = base_f
                        elif is_nondominated(fc, np.vstack([base_f, fc])) and np.any(fc < base_f):
                            # Non-dominated and better in at least one objective
                            # Keep both: update base but also consider adding
                            # For simplicity, just update base if strictly better in sum
                            if np.sum(fc) < np.sum(base_f):
                                base_x = candidate
                                base_f = fc
                                all_X[orig_idx] = base_x
                                all_F[orig_idx] = base_f

        return all_X

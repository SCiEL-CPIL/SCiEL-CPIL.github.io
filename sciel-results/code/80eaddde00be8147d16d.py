"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random

        used = 0
        all_X = []
        all_F = []

        def evaluate(x):
            nonlocal used
            fc = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(fc)
            return fc

        def get_front(F):
            """Return indices of non-dominated points in F (2D minimization)."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            F = np.array(F, dtype=float)
            # Filter points strictly better than reference
            mask = np.all(F < ref, axis=1)
            F = F[mask]
            valid_indices = np.where(mask)[0]
            if len(F) == 0:
                return np.array([], dtype=int)
            
            # Sort by obj0 ascending
            order = np.argsort(F[:, 0])
            F_sorted = F[order]
            
            # Keep only non-dominated (obj1 must be strictly decreasing)
            front_idx_sorted = [0]
            min_obj1 = F_sorted[0][1]
            for i in range(1, len(F_sorted)):
                if F_sorted[i][1] < min_obj1:
                    front_idx_sorted.append(i)
                    min_obj1 = F_sorted[i][1]
            
            # Map back to original indices
            front_indices = valid_indices[order[front_idx_sorted]]
            return front_indices

        def hv_contribution(point, front, ref):
            """Compute HV contribution of a point given a front."""
            if len(front) == 0:
                # Contribution is rectangle from point to ref
                if not np.all(point < ref):
                    return 0.0
                return (ref[0] - point[0]) * (ref[1] - point[1])
            
            # Find where point would be inserted in front sorted by obj0
            front = np.array(front, dtype=float)
            order = np.argsort(front[:, 0])
            front = front[order]
            
            # Build the actual non-dominated subset
            nd_front = [front[0]]
            for p in front[1:]:
                if p[1] < nd_front[-1][1]:
                    nd_front.append(p)
            
            if len(nd_front) == 0:
                return 0.0
            
            # Compute HV with and without point
            def compute_hv(pts):
                if len(pts) == 0:
                    return 0.0
                pts = np.array(pts, dtype=float)
                mask = np.all(pts < ref, axis=1)
                pts = pts[mask]
                if len(pts) == 0:
                    return 0.0
                order = np.argsort(pts[:, 0])
                pts = pts[order]
                front = [pts[0]]
                for p in pts[1:]:
                    if p[1] < front[-1][1]:
                        front.append(p)
                hv = 0.0
                for i in range(len(front)):
                    x_next = front[i + 1][0] if i + 1 < len(front) else ref[0]
                    y_curr = front[i][1]
                    if y_curr >= ref[1]:
                        continue
                    width = x_next - front[i][0]
                    height = ref[1] - y_curr
                    hv += width * height
                return hv
            
            hv_with = compute_hv([point] + list(front))
            hv_without = compute_hv(list(front))
            return max(0.0, hv_with - hv_without)

        # Phase 1: Initial exploration with stratified sampling
        init_budget = min(24, self.budget // 8)
        
        # Create stratified samples: Latin hypercube + corners + center
        for i in range(init_budget):
            if time.time() > deadline:
                break
            if i == 0:
                # Center
                frac = np.full(self.dim, 0.5)
            elif i < 4:
                # Corners
                corners = [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [0, 0, 1, 1],
                    [1, 1, 0, 0],
                ]
                frac = np.array(corners[i - 1], dtype=float)
            elif i < 8:
                # Face centers
                face_centers = [
                    [0.5, 0, 0, 0],
                    [0.5, 1, 0, 0],
                    [0, 0.5, 0, 0],
                    [1, 0.5, 0, 0],
                ]
                frac = np.array(face_centers[i - 4], dtype=float)
            else:
                # Latin hypercube-like: stratified random
                frac = (rng.random(self.dim) + np.arange(self.dim) * 0) / 1.0
                # Better: use a simple stratified approach
                frac = rng.random(self.dim)
            x = lo + frac * (hi - lo)
            evaluate(x)

        # Phase 2: Main optimization loop with adaptive DE
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get current front
            front_idx = get_front(all_F)
            
            # Adaptive parameters based on progress
            progress = used / max(1, self.budget)
            F_mut = 0.8 - 0.5 * progress  # Decrease mutation over time
            CR = 0.3 + 0.4 * progress     # Increase crossover over time
            
            if len(front_idx) == 0:
                # Fallback: pick random points
                for _ in range(min(5, self.budget - used)):
                    if time.time() > deadline:
                        break
                    x = lo + (hi - lo) * rng.random(self.dim)
                    evaluate(x)
                continue
            
            # Compute contributions for front points (efficient 2D HV)
            front_points = np.array([all_F[i] for i in front_idx], dtype=float)
            # Filter strictly better than ref
            mask = np.all(front_points < ref, axis=1)
            front_points = front_points[mask]
            if len(front_points) == 0:
                for _ in range(min(5, self.budget - used)):
                    if time.time() > deadline:
                        break
                    x = lo + (hi - lo) * rng.random(self.dim)
                    evaluate(x)
                continue
            
            # Sort by obj0, keep non-dominated
            order = np.argsort(front_points[:, 0])
            front_points = front_points[order]
            nd_front = [front_points[0]]
            for p in front_points[1:]:
                if p[1] < nd_front[-1][1]:
                    nd_front.append(p)
            nd_front = np.array(nd_front)
            
            # Compute HV contribution of each front point
            contributions = []
            for i in range(len(nd_front)):
                if i == 0:
                    x_right = nd_front[1][0] if len(nd_front) > 1 else ref[0]
                elif i == len(nd_front) - 1:
                    x_right = ref[0]
                else:
                    x_right = nd_front[i + 1][0]
                y_val = nd_front[i][1]
                if y_val >= ref[1]:
                    contributions.append(0.0)
                else:
                    contributions.append((x_right - nd_front[i][0]) * (ref[1] - y_val))
            
            contributions = np.array(contributions)
            total_c = contributions.sum()
            if total_c > 1e-10:
                probs = contributions / total_c
            else:
                probs = np.ones(len(nd_front)) / len(nd_front)
            
            # Map back to original indices
            # We need to find which original indices correspond to nd_front
            # This is complex, so let's just use the front_idx directly
            # Simplify: pick parents from front_idx weighted by their contribution
            # Rebuild: compute contribution for each point in front_idx
            front_points_full = np.array([all_F[i] for i in front_idx], dtype=float)
            order_full = np.argsort(front_points_full[:, 0])
            front_points_full = front_points_full[order_full]
            nd_indices_in_sorted = [0]
            for j in range(1, len(front_points_full)):
                if front_points_full[j][1] < front_points_full[nd_indices_in_sorted[-1]][1]:
                    nd_indices_in_sorted.append(j)
            
            # Compute contributions for these
            contribs = []
            for k in range(len(nd_indices_in_sorted)):
                idx_in_sorted = nd_indices_in_sorted[k]
                if k == 0:
                    x_right = front_points_full[nd_indices_in_sorted[1]][0] if len(nd_indices_in_sorted) > 1 else ref[0]
                elif k == len(nd_indices_in_sorted) - 1:
                    x_right = ref[0]
                else:
                    x_right = front_points_full[nd_indices_in_sorted[k + 1]][0]
                y_val = front_points_full[idx_in_sorted][1]
                if y_val >= ref[1]:
                    contribs.append(0.0)
                else:
                    contribs.append((x_right - front_points_full[idx_in_sorted][0]) * (ref[1] - y_val))
            
            # Build probability vector for front_idx
            full_probs = np.ones(len(front_idx)) * 1e-10
            for k, idx_in_sorted in enumerate(nd_indices_in_sorted):
                orig_idx = order_full[idx_in_sorted]
                full_probs[orig_idx] = contribs[k]
            
            total_p = full_probs.sum()
            if total_p > 1e-10:
                full_probs = full_probs / total_p
            else:
                full_probs = np.ones(len(front_idx)) / len(front_idx)
            
            # Generate new candidate using DE on the full archive (not just front)
            # This helps exploration
            if len(all_X) >= 3:
                # Pick 3 random parents from archive, biased toward front
                # With some probability, use front-only DE
                if rng.random() < 0.7:
                    # Use front for DE
                    parent_indices = rng.choice(len(front_idx), size=3, replace=False, p=full_probs)
                    p1 = all_X[front_idx[parent_indices[0]]]
                    p2 = all_X[front_idx[parent_indices[1]]]
                    p3 = all_X[front_idx[parent_indices[2]]]
                else:
                    # Use archive for diversity
                    parent_indices = rng.choice(len(all_X), size=3, replace=False)
                    p1 = all_X[parent_indices[0]]
                    p2 = all_X[parent_indices[1]]
                    p3 = all_X[parent_indices[2]]
                
                v = p1 + F_mut * (p2 - p3)
                mask = rng.random(self.dim) < CR
                child = np.where(mask, v, p1)
            elif len(all_X) == 2:
                p1 = all_X[rng.choice(2)]
                p2 = all_X[rng.choice(2)]
                v = 0.5 * (p1 + p2) + 0.3 * (p1 - p2) * rng.random(self.dim)
                mask = rng.random(self.dim) < CR
                child = np.where(mask, v, p1)
            else:
                p1 = all_X[0]
                child = p1.copy()
            
            # Add Gaussian perturbation (adaptive scale)
            mut_scale = 0.15 * (1.0 - 0.7 * progress) + 0.01
            child = child + rng.standard_normal(self.dim) * mut_scale * (hi - lo)
            child = np.clip(child, lo, hi)
            
            fc = evaluate(child)
            
            # Occasionally add a purely random point for exploration
            if used % 20 == 0 and used < self.budget - 3:
                x_rand = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x_rand)
            
            # Final polish: in last 10% of budget, do local refinement
            if used > 0.9 * self.budget and used < self.budget:
                # Get current best front and do local search around it
                front_idx_local = get_front(all_F)
                if len(front_idx_local) > 0:
                    # Pick a random front point and do local refinement
                    best_idx = front_idx_local[rng.randint(len(front_idx_local))]
                    base = all_X[best_idx].copy()
                    # Try small perturbations
                    for _ in range(min(3, self.budget - used)):
                        if time.time() > deadline:
                            break
                        perturb = rng.standard_normal(self.dim) * 0.02 * (hi - lo)
                        x_trial = np.clip(base + perturb, lo, hi)
                        evaluate(x_trial)

        return all_X
# EVOLVE-BLOCK-END

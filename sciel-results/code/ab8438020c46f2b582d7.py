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

    def _hv_contribution(self, point, front, ref):
        """Compute the hypervolume contribution of a single point given the current front."""
        # For 2D, contribution is the area of the rectangle that would be lost if this point were removed
        # Sort front by first objective
        if len(front) == 0:
            return 0.0
        f = np.array(sorted(front, key=lambda x: x[0]))
        # Find position of point in sorted order
        idx = None
        for i, p in enumerate(f):
            if np.allclose(p, point):
                idx = i
                break
        if idx is None:
            # Point not in front (dominated), contribution is 0
            return 0.0
        # Contribution: width * height of the region uniquely covered
        # Width: gap between previous and next point in obj0
        prev_obj0 = f[idx - 1][0] if idx > 0 else ref[0]
        next_obj0 = f[idx + 1][0] if idx < len(f) - 1 else ref[0]
        # Actually for minimization, we need the region from point to ref
        # The contribution is: (next_obj0 - point[0]) * (point[1] - (f[idx+1][1] if idx < len-1 else ref[1]))
        # But this is complex. Let me use a simpler approach: compute HV with and without.
        # For efficiency, use the known formula for 2D.
        
        # In 2D, sorted by obj0 ascending, the HV is sum of (f[i][0] - f[i+1][0]) * (ref[1] - f[i][1]) for the front
        # Actually, let me just compute the area contribution directly.
        
        # The unique region contributed by point i is:
        # x-range: from f[i][0] to f[i+1][0] (or ref[0] if last)
        # y-range: from f[i][1] to min(f[i+1][1], ref[1]) if i < last, else from f[i][1] to ref[1]
        # But this isn't quite right either. Let me use a different approach.
        
        # Simpler: compute HV of front, then HV of front without this point, difference is contribution.
        # But that's O(n) per point. For small fronts it's fine.
        
        def hv_2d(pts, ref):
            if len(pts) == 0:
                return 0.0
            pts = np.array(pts)
            # Filter strictly better than ref
            mask = np.all(pts < ref, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            # Keep only non-dominated (Pareto front)
            # Sort by obj0 ascending
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # Remove dominated: for each point, check if any previous point has obj1 <= current obj1
            front = [pts[0]]
            for p in pts[1:]:
                if p[1] < front[-1][1]:
                    front.append(p)
            if len(front) == 0:
                return 0.0
            # Compute HV
            hv = 0.0
            for i in range(len(front)):
                x_next = front[i + 1][0] if i + 1 < len(front) else ref[0]
                y_curr = front[i][1]
                # The rectangle from (front[i][0], y_curr) to (x_next, ref[1])
                # But only if y_curr < ref[1]
                if y_curr >= ref[1]:
                    continue
                width = x_next - front[i][0]
                height = ref[1] - y_curr
                hv += width * height
            return hv

        front_with = list(front)
        front_without = [p for p in front if not np.allclose(p, point)]
        return hv_2d(front_with, ref) - hv_2d(front_without, ref)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        
        # Initial population: diverse random points
        pop_size = min(20, self.budget // 5)
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_X = list(X)
        all_F = list(F)
        
        # Adaptive mutation scale
        mut_scale = 0.2
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated front
            n = len(all_F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                        dominated[i] = True
                        break
            front_idx = np.where(~dominated)[0]
            
            if len(front_idx) == 0:
                front_idx = np.array([0])
            
            # Select parent from front using hypervolume contribution (greedy)
            # For small fronts, compute contributions
            best_parent_idx = front_idx[0]
            best_contrib = -1
            for idx in front_idx:
                # Approximate contribution: just use the point's "uniqueness"
                # Simpler: random selection from front with bias toward extreme points
                # Use tournament: pick 2 random from front, take the one with lower sum
                if len(front_idx) > 1:
                    i1, i2 = np.random.choice(front_idx, 2, replace=False)
                    if np.sum(all_F[i1]) < np.sum(all_F[i2]):
                        parent_idx = i1
                    else:
                        parent_idx = i2
                else:
                    parent_idx = front_idx[0]
                
                # Compute a simple contribution metric: distance from other front points
                dists = []
                for other_idx in front_idx:
                    if other_idx != parent_idx:
                        dists.append(np.linalg.norm(all_F[parent_idx] - all_F[other_idx]))
                contrib = np.mean(dists) if dists else 1.0
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_parent_idx = parent_idx
            
            parent = all_X[best_parent_idx].copy()
            
            # Mutate
            mutation = np.random.standard_normal(self.dim) * mut_scale * (hi - lo)
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            all_X.append(child)
            all_F.append(fc)
            
            # Adapt mutation scale: decrease if we're making progress, increase if stuck
            # Simple adaptation: reduce scale over time
            progress = used / self.budget
            mut_scale = 0.2 * (1.0 - 0.5 * progress)
            
            # Occasionally do a random restart to explore new regions
            if used % 40 == 0 and used < self.budget - 5:
                random_child = lo + (hi - lo) * np.random.random(self.dim)
                rc = problem(random_child)
                used += 1
                all_X.append(random_child)
                all_F.append(rc)
        
        return all_X
# EVOLVE-BLOCK-END

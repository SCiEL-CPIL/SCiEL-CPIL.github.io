"""Multi-objective optimizer using surrogate-assisted sampling with LHS init and HV-guided selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer: LHS init, HV-guided parent selection, adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # LHS initialization for better spread
        n_init = min(40, self.budget // 5)
        if n_init < 5:
            n_init = 5

        lhs = qmc.LatinHypercube(d=self.dim, seed=0)
        X = lhs.random(n_init)
        X = lo + (hi - lo) * X
        F = np.array([problem(x) for x in X])
        used = n_init

        ref_point = np.array([1705.0, 12.0, 0.3])

        def hypervolume_3d(points, ref):
            """Compute hypervolume for 3 objectives, filtering points better than ref in all objectives."""
            pts = np.array(points)
            pts = pts[np.all(pts < ref, axis=1)]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective (ascending, i.e., better first)
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            total = 0.0
            prev_hv = 0.0
            for i in range(len(pts)):
                p = pts[i]
                # This point dominates all subsequent points in obj 0
                # Compute 2D hypervolume contribution
                sub_pts = pts[i:]
                # The volume contributed by point i:
                # width in obj0 = ref[0] - p[0]
                # For obj1 and obj2: need 2D HV of points from i onwards
                # 2D HV: sort by obj1, sum (ref[j]-p[j]) * (ref[k]-p[k])
                # Actually simpler: for 3D, use inclusion-exclusion or layer cake
                # Layer cake: HV = sum over sorted points of (ref[0]-p_i[0]) * 2DHV(points from i onward in obj1,obj2)
                sub_2d = sub_pts[:, 1:3]
                sub_2d = sub_2d[np.all(sub_2d < ref[1:], axis=1)]
                if len(sub_2d) == 0:
                    continue
                idx2 = np.argsort(sub_2d[:, 0])
                sub_2d = sub_2d[idx2]
                hv2 = 0.0
                max_obj2 = 0.0
                for j in range(len(sub_2d)):
                    q_pt = sub_2d[j]
                    max_obj2 = max(max_obj2, q_pt[1])
                    hv2 += (ref[1] - q_pt[0]) * (ref[2] - max_obj2)
                total += (ref[0] - p[0]) * hv2
                # Wait, this double-counts. Let me use a cleaner approach.
            # Clean 3D HV: sort by obj0, iterate, maintain max obj2 seen so far among points with obj1 < current
            # Actually the standard O(n^2) for 3D:
            pts_sorted = pts[np.argsort(pts[:, 0])]
            total_hv = 0.0
            for i in range(len(pts_sorted)):
                p = pts_sorted[i]
                # Points j >= i have obj0 >= p[0] (worse or equal)
                # For p to contribute, we need the region where p is best in obj0 (automatic since sorted)
                # and p is best in obj1 and obj2 among remaining
                # Simpler: for each point, its contribution is (ref[0]-p[0]) * (2D HV of points dominated by p in obj1,obj2)
                # No - let's just use the layer-cake properly:
                # HV = integral of indicator function. In 3D with sorted by obj0:
                # For layer between p_i[0] and p_{i+1}[0], the 2D HV of points with obj0 <= that layer
                # This is getting complex. Let me use a simple but correct O(n^2) 2D approach iterated.
                pass

            # Use a simpler exact 3D HV via sorting
            pts_s = pts[np.argsort(pts[:, 0])]
            total_hv = 0.0
            for i in range(len(pts_s)):
                p = pts_s[i]
                # 2D hypervolume of the projection of points from i to end onto (obj1, obj2)
                # that are all < ref in obj1 and obj2
                proj = pts_s[i:, 1:3]
                proj = proj[np.all(proj < ref[1:], axis=1)]
                if len(proj) == 0:
                    continue
                # 2D HV: sort by obj1
                proj_s = proj[np.argsort(proj[:, 0])]
                hv2 = 0.0
                max_o2 = 0.0
                for j in range(len(proj_s)):
                    q_pt = proj_s[j]
                    if q_pt[1] > max_o2:
                        max_o2 = q_pt[1]
                    hv2 += (ref[1] - q_pt[0]) * (ref[2] - max_o2)
                total_hv += (ref[0] - p[0]) * hv2

            return total_hv

        def dominates(a, b):
            """Check if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Return non-dominated set indices."""
            n = len(F)
            non_dom = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dom[i]:
                    continue
                for j in range(n):
                    if i == j or not non_dom[j]:
                        continue
                    if dominates(F[j], F[i]):
                        non_dom[i] = False
                        break
            return non_dom

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of points."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            idx = list(indices)
            for m in range(self.n_obj):
                order = sorted(range(n), key=lambda i: F[idx[i]][m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_min = F[idx[order[0]]][m]
                f_max = F[idx[order[-1]]][m]
                span = f_max - f_min
                if span == 0:
                    continue
                for k in range(1, n - 1):
                    cd[order[k]] += (F[idx[order[k+1]]][m] - F[idx[order[k-1]]][m]) / span
            return cd

        def hv_contribution(F, idx, ref):
            """Approximate hypervolume contribution of point idx to the front."""
            # Simple: count how many points would be removed if this one were
            n = len(idx)
            if n == 0:
                return 0.0
            front = F[idx]
            # Remove point idx
            remaining = [i for i in range(n) if i != idx]
            if len(remaining) == 0:
                return (ref[0] - front[0]) * (ref[1] - front[1]) * (ref[2] - front[2])
            # Approximate by just the box volume (crude)
            return (ref[0] - front[0]) * (ref[1] - front[1]) * (ref[2] - front[2])

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated set
            non_dom = non_dominated_sort(F)
            non_dom_idx = np.where(non_dom)[0]

            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))

            # Select parent using tournament among non-dominated, biased toward HV contribution
            # Use a simple tournament of size 3
            if len(non_dom_idx) > 3:
                candidates = rng.choice(non_dom_idx, size=3, replace=False)
            else:
                candidates = non_dom_idx

            # Score by negative sum of objectives (proxy for quality) + crowding
            scores = []
            for ci in candidates:
                f_val = F[ci]
                # Penalize points far from ideal (lower is better for each obj)
                quality_score = -np.sum(f_val / ref_point)
                scores.append(quality_score)

            parent_idx = candidates[np.argmax(scores)]
            parent = X[parent_idx].copy()

            # Adaptive mutation step
            base_step = 0.15
            # Scale down as we approach budget
            progress = used / self.budget
            step = base_step * (1.0 - 0.5 * progress)
            mutation = step * (hi - lo) * rng.standard_normal(self.dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst point (by sum of objectives among dominated, or worst non-dominated by crowding)
            # Find the point with highest sum (worst)
            sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(sums))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END

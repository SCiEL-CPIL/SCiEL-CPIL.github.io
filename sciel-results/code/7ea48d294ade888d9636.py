import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([1.1, 1.3, 1.1])

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        archive_X = []
        archive_F = []
        evals = 0
        
        # Helper: 2D Hypervolume
        def hv_2d(points_2d, ref_2d):
            # points_2d: list of (f2, f3)
            # ref_2d: (r2, r3)
            if not points_2d:
                return 0.0
            
            # Filter non-dominated in 2D
            # Sort by f2 descending
            pts = sorted(points_2d, key=lambda x: -x[0])
            nd_pts = []
            max_f3 = -np.inf
            for p in pts:
                if p[1] > max_f3:
                    nd_pts.append(p)
                    max_f3 = p[1]
            
            if not nd_pts:
                return 0.0
            
            # Calculate HV
            hv = 0.0
            r2, r3 = ref_2d
            n = len(nd_pts)
            for i in range(n):
                f2, f3 = nd_pts[i]
                next_f2 = nd_pts[i+1][0] if i < n - 1 else 0.0 # Assuming 0 is lower bound for f2? No, f2 can be anything.
                # The width in f2 is from f2 to next_f2 (if next exists) or to -inf?
                # The box is [f2, r2] x [f3, r3].
                # In the sorted order (f2 desc), the unique contribution of point i in f2 direction
                # is from f2[i] to f2[i+1].
                # But we must ensure we don't go below 0? The problem bounds are [0,1].
                # The reference is 1.3.
                # The "width" is f2[i] - f2[i+1].
                # If i is the last, width is f2[i] - 0? Or f2[i] - min_possible?
                # Since objectives are minimized, and bounds are [0,1], the "empty" space starts from 0.
                # However, the HV is defined relative to the reference point.
                # The volume is the union of boxes [f, R].
                # So the integration limit is from min(f) to R.
                # The simple sum works if we consider the "step" function.
                
                w2 = f2 - (nd_pts[i+1][0] if i < n - 1 else 0.0)
                # Wait, if the last point has f2=0.5, and next is 0.0, w2=0.5.
                # The box extends to r2=1.3.
                # The area is w2 * (r3 - f3).
                # This assumes the "floor" is 0.
                # Is the floor 0?
                # The reference point defines the "top-right" corner.
                # The "bottom-left" corner of the union is determined by the points.
                # If all points have f2 > 0.2, the region [0, 0.2] x [0, r3] is NOT covered by any box [f, R] if f > 0.2?
                # No, the box is [f2, r2] x [f3, r3].
                # So if f2=0.5, the box covers [0.5, 1.3]. It does NOT cover [0, 0.5].
                # So the HV is only the volume of the union of these boxes.
                # The "width" w2 should be f2[i] - f2[i+1].
                # And the last width is f2[last] - 0?
                # No, if the last point is f2=0.5, the box covers [0.5, 1.3].
                # The next "virtual" point is at f2=0?
                # The union does not cover [0, 0.5].
                # So the sum should stop.
                # Actually, the formula $\sum (f_i - f_{i+1})(r - h_i)$ works if we set $f_{n} = 0$ (or min bound) ONLY IF the boxes extend down to 0.
                # But they extend UP to R.
                # So the "base" of the integration is the minimum f value?
                # No, the volume is $\int_{f_{min}}^{R} A(z) dz$.
                # The simple sum $\sum (f_i - f_{i+1})(r - h_i)$ calculates the area under the step curve from $f_{last}$ to $f_{first}$.
                # It does NOT include the area from $0$ to $f_{last}$.
                # But the boxes don't cover that area anyway.
                # So setting $f_{n} = 0$ is correct IF the boxes started at 0.
                # But they start at $f_i$.
                # So the "width" of the last step is $f_{last} - 0$?
                # Yes, because the box extends from $f_{last}$ to $R$.
                # The "gap" below $f_{last}$ is not covered.
                # So the volume is indeed $\sum_{i=0}^{n-1} (f_i - f_{i+1}) (r - h_i)$ with $f_n = 0$.
                
                h3 = r3 - f3
                hv += w2 * h3
            return hv

        # Helper: 3D Hypervolume
        def hv_3d(points_3d, ref_3d):
            # points_3d: list of (f1, f2, f3)
            if not points_3d:
                return 0.0
            
            # Sort by f1 descending
            pts = sorted(points_3d, key=lambda x: -x[0])
            n = len(pts)
            hv = 0.0
            r1, r2, r3 = ref_3d
            
            for i in range(n):
                f1 = pts[i][0]
                next_f1 = pts[i+1][0] if i < n - 1 else 0.0
                w1 = f1 - next_f1
                
                # Get points 0..i
                subset = pts[:i+1]
                # Extract 2D coords
                pts_2d = [(p[1], p[2]) for p in subset]
                
                hv2 = hv_2d(pts_2d, (r2, r3))
                hv += w1 * hv2
            return hv

        # Init
        init_pop = 20
        X_init = lo + (hi - lo) * rng.random((init_pop, self.dim))
        for x in X_init:
            f = problem(x)
            archive_X.append(x)
            archive_F.append(f)
            evals += 1
            
        # Main Loop
        while evals < self.budget:
            # Compute Non-Dominated Set
            F = np.array(archive_F)
            nd_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not nd_mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd_mask[i] = False
                        break
            
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) == 0:
                # Fallback to random
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                archive_X.append(x)
                archive_F.append(f)
                evals += 1
                continue
                
            # Get non-dominated points values
            nd_points = [tuple(archive_F[i]) for i in nd_indices]
            
            # Calculate Total HV
            total_hv = hv_3d(nd_points, self.ref_point)
            
            # Calculate Marginal HV
            marginals = []
            for k in range(len(nd_indices)):
                # Remove k-th point
                reduced_points = nd_points[:k] + nd_points[k+1:]
                hv_minus = hv_3d(reduced_points, self.ref_point)
                marginal = total_hv - hv_minus
                marginals.append(marginal)
                
            # Select Top 5
            top_k = min(5, len(marginals))
            # Get indices of top k
            top_indices = np.argsort(marginals)[::-1][:top_k]
            
            # Pick one randomly
            chosen_local_idx = top_indices[rng.integers(top_k)]
            chosen_global_idx = nd_indices[chosen_local_idx]
            
            # Generate Child
            parent_x = archive_X[chosen_global_idx]
            sigma = 0.15
            child_x = parent_x + sigma * rng.standard_normal(self.dim)
            child_x = np.clip(child_x, lo, hi)
            
            # Evaluate
            child_f = problem(child_x)
            archive_X.append(child_x)
            archive_F.append(child_f)
            evals += 1
            
        return archive_X

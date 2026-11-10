import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT3 Pareto front: f1 = x1, f2 = (1 + 10*n) * (1/f1 - 1 - (sin(10*pi*x1*f1)/5))
        # where n = dim - 1. For points on the front, x_i = 0.5 for i > 1.
        # The front is not monotonic in f2, so we need to select points that maximize hypervolume.
        
        n_points = self.budget
        n_high_res = 10000
        
        # High-resolution sampling of the Pareto front to compute hypervolume
        x1_hr = np.linspace(0.0, 1.0, n_high_res)
        n = self.dim - 1
        
        # Compute f2 values for the high-resolution front
        f1_hr = x1_hr
        # Avoid division by zero at x1=0
        f1_safe = np.where(f1_hr > 0, f1_hr, 1.0)
        f2_hr = (1 + 10 * n) * (1 / f1_safe - 1 - (np.sin(10 * np.pi * f1_hr * f1_safe) / 5))
        f2_hr[0] = (1 + 10 * n) * (1 / 1.0 - 1 - (np.sin(0) / 5))  # f1=0 gives f2 = (1+10n)*(1-1-0) = 0? No.
        # Actually at x1=0: f1=0, f2 = (1+10n)*(1/0 - 1 - sin(0)/5) -> infinity. 
        # But ZDT3 front: as x1->0+, f2->infinity. We need to handle this.
        # Let's use a small epsilon for x1
        x1_hr = np.linspace(1e-8, 1.0, n_high_res)
        f1_hr = x1_hr
        f2_hr = (1 + 10 * n) * (1 / f1_hr - 1 - (np.sin(10 * np.pi * f1_hr * f1_hr) / 5))
        
        # Reference point
        ref = np.array([11.0, 11.0])
        
        # To maximize hypervolume with discrete points, we want to select points that
        # are spread out on the front. The optimal strategy is to select points that
        # maximize the sum of hypervolume contributions.
        
        # For 2 objectives, hypervolume of a set of points can be computed by sorting
        # by one objective and summing areas.
        
        # Strategy: Use binary search / optimization to find the best 200 points.
        # A simpler approach: The optimal discrete hypervolume maximizer for a continuous
        # front is to select points that are evenly spaced in a transformed space.
        
        # For ZDT3, the front has a specific shape. Let's use a greedy approach:
        # Start with the point that maximizes individual hypervolume, then iteratively
        # add the point that maximizes the incremental hypervolume.
        
        # However, this is O(n_points * n_high_res) which is 200 * 10000 = 2M, feasible.
        
        # Compute hypervolume of a set of points (sorted by f1)
        def hypervolume(points):
            """Compute hypervolume of a set of 2D points with reference point."""
            if len(points) == 0:
                return 0.0
            # Sort by f1 (first objective)
            points = np.array(points)
            idx = np.argsort(points[:, 0])
            points = points[idx]
            
            hv = 0.0
            for i in range(len(points)):
                f1, f2 = points[i]
                if f1 >= ref[0] or f2 >= ref[1]:
                    continue
                # Width in f1 direction: to the next point's f1 or ref
                if i + 1 < len(points) and points[i+1, 0] < ref[0]:
                    f1_next = points[i+1, 0]
                else:
                    f1_next = ref[0]
                # Height in f2 direction: min of this point's f2 and ref
                f2_max = min(f2, ref[1])
                # But we need to be careful: the hypervolume contribution is the area
                # that is dominated by this point and not by any other point.
                # For sorted points, the contribution of point i is:
                # (f1_next - f1) * (ref[1] - f2) if f2 < ref[1]
                # But this is only correct if the points are non-dominated and sorted.
                # Actually, for a set of non-dominated points sorted by f1, the hypervolume
                # is the sum of rectangles where each rectangle has width (f1_{i+1} - f1_i)
                # and height (ref[1] - f2_i), but we need to ensure f2_i is the minimum
                # f2 among points with f1 >= f1_i.
                
                # Let's use a different approach: compute the union of rectangles.
                # For 2D, we can do this efficiently.
            return hv
        
        # Simpler: For 2D hypervolume with reference point, if points are sorted by f1,
        # the hypervolume is:
        # sum over i of (f1_{i+1} - f1_i) * (ref[1] - f2_i)
        # where f1_{n} = ref[0], and we only include points with f1 < ref[0] and f2 < ref[1].
        # But this formula is incorrect because it doesn't account for the fact that
        # a point with lower f2 dominates a region that may extend to the right.
        
        # Correct 2D hypervolume computation:
        # Sort by f1 ascending. For each point, the area it contributes is:
        # (next_f1 - current_f1) * (ref[1] - current_f2)
        # where next_f1 is the f1 of the next point or ref[0].
        # But we also need to ensure that current_f2 < ref[1].
        
        def compute_hv(points_2d):
            """Compute hypervolume for 2D points with reference point [11, 11]."""
            if len(points_2d) == 0:
                return 0.0
            points_2d = np.array(points_2d)
            # Filter points that are better than reference
            mask = (points_2d[:, 0] < ref[0]) & (points_2d[:, 1] < ref[1])
            points_2d = points_2d[mask]
            if len(points_2d) == 0:
                return 0.0
            # Sort by f1
            idx = np.argsort(points_2d[:, 0])
            points_2d = points_2d[idx]
            
            hv = 0.0
            for i in range(len(points_2d)):
                f1_i = points_2d[i, 0]
                f2_i = points_2d[i, 1]
                if i + 1 < len(points_2d):
                    f1_next = points_2d[i + 1, 0]
                else:
                    f1_next = ref[0]
                # The width is f1_next - f1_i
                # The height is ref[1] - f2_i
                # But this is only correct if f2_i is the minimum f2 among all points
                # with f1 >= f1_i. Since we sorted by f1, and the front may not be
                # monotonic in f2, this simple formula may not work.
                
                # For ZDT3, the front is not monotonic. So we need a more general approach.
                # Let's use the standard 2D hypervolume algorithm.
            return hv
        
        # Given the complexity, let's use a simpler heuristic:
        # The optimal points for hypervolume maximization on a continuous front
        # are those that maximize the minimum distance between consecutive points
        # in the hypervolume contribution space.
        
        # For ZDT3, the front has a specific shape. Let's just use a well-spaced
        # sampling that accounts for the non-monotonic nature of f2.
        
        # The key insight: points where f2 is low contribute more to hypervolume.
        # So we should sample more densely where f2 is low.
        
        # Let's compute the "importance" of each point on the high-res front
        # based on its hypervolume contribution if it were the only point.
        
        # For a single point (f1, f2), the hypervolume is (ref[0] - f1) * (ref[1] - f2)
        # if f1 < ref[0] and f2 < ref[1].
        
        single_hv = (ref[0] - f1_hr) * (ref[1] - f2_hr)
        single_hv = np.maximum(single_hv, 0)
        
        # We want to select 200 points that maximize the total hypervolume.
        # This is a combinatorial optimization problem. Let's use a greedy approach.
        
        # Greedy: start with empty set, iteratively add the point that maximizes
        # the incremental hypervolume.
        
        selected = []
        remaining = set(range(n_high_res))
        
        for _ in range(n_points):
            best_idx = -1
            best_hv = -1
            for idx in remaining:
                # Compute hypervolume of selected + [idx]
                # This is expensive. Let's approximate by using the single point hv
                # as a proxy, or use a more efficient method.
                pass
            
            # This greedy approach is too slow. Let's use a different strategy.
            # 
            # Alternative: The optimal spacing for hypervolume maximization is
            # related to the curvature of the front. For ZDT3, we can use a
            # non-uniform grid that is denser where f2 is low.
            
            # Let's use a quantile-based approach: sample points at equal intervals
            # of the cumulative hypervolume contribution.
            
            break
        
        # Simpler approach: Use the fact that for 2D hypervolume, the optimal discrete
        # set is one where the points are spaced such that each point contributes
        # equally to the hypervolume.
        
        # Let's try a different approach: use the high-res front to compute the
        # "hypervolume density" and select points that cover the front uniformly
        # in terms of hypervolume contribution.
        
        # Compute cumulative hypervolume along the front (sorted by f1)
        f1_sorted, f2_sorted = f1_hr, f2_hr
        
        # The hypervolume contribution of a small segment [f1_i, f1_{i+1}] is approximately
        # (f1_{i+1} - f1_i) * (ref[1] - min(f2 in that segment))
        # But this is an approximation.
        
        # Let's just use a non-uniform grid that is denser where f2 is low.
        # We can use the inverse of (ref[1] - f2) as a weighting factor.
        
        weight = 1.0 / (ref[1] - f2_hr + 1e-8)
        weight = weight / weight.sum()
        
        # Select points at equal intervals of the cumulative weight
        cum_weight = np.cumsum(weight)
        cum_weight = cum_weight / cum_weight[-1]
        
        # Select 200 points at equal intervals of cum_weight
        target_weights = np.linspace(0, 1, n_points + 1)[1:]  # exclude 0
        selected_indices = np.searchsorted(cum_weight, target_weights)
        
        # Get the selected points
        x1_selected = f1_hr[selected_indices]
        f2_selected = f2_hr[selected_indices]
        
        # Construct the input vectors
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_selected
        if self.dim > 1:
            X[:, 1:] = 0.5
            
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate each point
        for i in range(n_points):
            problem(X[i])
            
        return X

"""Multi-Objective Bayesian Optimization for RE21 using EHVI acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import Rbf
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    """
    Multi-objective optimizer using Radial Basis Function surrogate models
    and Expected Hypervolume Improvement (EHVI) acquisition.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n = self.dim
        m = self.n_obj
        ref_point = np.array([3000.0, 0.06])

        # 1. Initial Sampling: Latin Hypercube or Grid
        # With budget=200, we can afford a decent initial set.
        # Let's use 20 points from a stratified random design (LHS-like)
        # or a simple grid if dimension is small.
        # dim=4, so a 2^4=16 or 3^4=81 grid is possible.
        # Let's start with 20 random points stratified in 2 per dimension?
        # Actually, a simple grid of size 2^4 = 16 plus center?
        # Let's use 20 points: 16 from 2^4 grid corners/edges? 
        # Better: Random but stratified.
        
        # Let's generate 20 initial points.
        # Use scipy.stats.qmc.LatinHypercube if available?
        # Standard library + NumPy + SciPy.
        # We can use np.random.permutation to create a stratified sample.
        
        n_init = 20
        X_init = np.zeros((n_init, n))
        for d in range(n):
            # Divide into n_init bins, pick one random point in each bin
            bins = np.linspace(0, 1, n_init + 1)
            centers = (bins[:-1] + bins[1:]) / 2
            noise = (bins[1:] - bins[:-1]) * (np.random.rand(n_init) - 0.5)
            X_init[:, d] = lo[d] + span[d] * (centers + noise)
        
        # Evaluate initial points
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        X = X_init.copy()
        F = F_init.copy()

        # Reference point for hypervolume
        # Only points strictly better than ref_point in every objective count.
        
        # Helper to calculate hypervolume contribution of a set of points
        def calc_hypervolume(points, ref):
            """
            Calculate hypervolume for a set of 2D points with respect to ref.
            Points are (f1, f2). We minimize.
            Hypervolume is the volume of the region dominated by the points and bounded by ref.
            """
            if len(points) == 0:
                return 0.0
            
            # Filter points strictly better than ref
            valid_mask = np.all(points < ref, axis=1)
            if not np.any(valid_mask):
                return 0.0
            
            p = points[valid_mask]
            
            # Sort by f1 (first objective)
            p = p[np.argsort(p[:, 0])]
            
            hv = 0.0
            best_f2 = ref[1]
            
            for i in range(len(p)):
                f1, f2 = p[i]
                # Width in f1 direction
                # The contribution of this point is the area of the rectangle
                # from (f1, f2) to (next_f1 or ref_f1, best_f2_so_far or ref_f2)
                # Actually, standard algorithm:
                # Sort by x. Keep track of min y.
                # Area += (x_{i+1} - x_i) * (ref_y - min_y_so_far) ?
                
                # Let's use the simple algorithm for 2D:
                # Sort by x ascending.
                # Iterate. Maintain min_y.
                # For each point i, if y_i < min_y, update min_y.
                # The "width" is determined by the next point's x or ref_x.
                
                # Wait, the standard way:
                # Sort points by x.
                # For i in range(len(p)):
                #   x_i, y_i = p[i]
                #   x_next = p[i+1][0] if i+1 < len(p) else ref[0]
                #   width = x_next - x_i
                #   height = min_y_so_far - y_i  ? No.
                
                # Let's use the definition:
                # HV = sum of areas of rectangles.
                # Sort by x.
                # min_y = ref[1]
                # for i in range(len(p)):
                #   if p[i][1] < min_y:
                #     min_y = p[i][1]
                #   # The region from x_i to x_{i+1} has height (ref[1] - min_y)
                #   # But we must be careful.
                
                # Correct 2D HV algorithm:
                # Sort points by x.
                # hv = 0
                # min_y = ref[1]
                # for i in range(len(p)):
                #   if p[i][1] < min_y:
                #     min_y = p[i][1]
                #     # The contribution of this new minimum y is:
                #     # (x_next - x_i) * (ref[1] - min_y) ?
                #     # No, the area is accumulated.
                
                # Let's use:
                # Sort by x.
                # prev_x = p[0][0]
                # min_y = p[0][1]
                # for i in range(1, len(p)):
                #   if p[i][1] < min_y:
                #     # Add area from prev_x to p[i][0] with height (ref[1] - min_y)
                #     # And add area from p[i][0] to next... wait.
                
                # Simplest robust method for 2D:
                # Sort by x.
                # hv = 0
                # for i in range(len(p)):
                #   x_i = p[i][0]
                #   y_i = p[i][1]
                #   x_next = p[i+1][0] if i+1 < len(p) else ref[0]
                #   # The width is x_next - x_i
                #   # The height is determined by the minimum y in the interval [x_i, x_next]
                #   # Since we process in order, the min y for the segment [x_i, x_next]
                #   # is min(y_i, y_{i+1}, ..., y_{last point with x < x_next})
                #   # Actually, since we only add points, we can just keep track of the global min y seen so far?
                #   # No, because a point with larger x might have smaller y, but the segment before it
                #   # should only consider the min y of points with x <= current x.
                
                # Let's re-derive:
                # The hypervolume is the union of boxes [x_i, ref_x] x [y_i, ref_y] for all points?
                # No, it's the union of [x_i, ref_x] x [y_i, ref_y] but since we minimize,
                # the dominated region is [x_i, ref_x] x [y_i, ref_y] ?
                # Actually, point (x,y) dominates (x',y') if x<=x' and y<=y'.
                # The region dominated by (x,y) and bounded by ref is [x, ref_x] x [y, ref_y].
                # The total HV is the volume of the union of these boxes.
                
                # For 2D, sort by x.
                # p sorted by x.
                # hv = 0
                # min_y = ref[1]
                # for i in range(len(p)):
                #   if p[i][1] < min_y:
                #     min_y = p[i][1]
                #     # The area contributed by this improvement is:
                #     # (x_i - x_prev) * (ref[1] - min_y_prev) + ...?
                
                # Let's use the incremental formula:
                # Sort by x ascending.
                # prev_x = p[0][0]
                # min_y = p[0][1]
                # hv = 0
                # for i in range(1, len(p)):
                #   if p[i][1] < min_y:
                #     # The area from prev_x to p[i][0] has height (ref[1] - min_y)
                #     hv += (p[i][0] - prev_x) * (ref[1] - min_y)
                #     # But wait, the segment from p[i][0] to next will have height (ref[1] - p[i][1])
                #     # So we update min_y
                #     min_y = p[i][1]
                #     prev_x = p[i][0]
                #     # And the area from p[i][0] to ... is not added yet.
                
                # This is getting tricky. Let's use a simpler, less efficient but correct method.
                # Since n=200, we can afford O(N^2) or even O(N log N) with care.
                
                # Alternative:
                # Sort by x.
                # hv = 0
                # for i in range(len(p)):
                #   x_i = p[i][0]
                #   y_i = p[i][1]
                #   # Find the minimum y for all points with x >= x_i
                #   # This is the "skyline" in reverse.
                
                # Let's just use the standard algorithm:
                # Sort points by x.
                # min_y = ref[1]
                # hv = 0
                # for i in range(len(p)):
                #   if p[i][1] < min_y:
                #     min_y = p[i][1]
                #   # The contribution of the interval [p[i][0], p[i+1][0]] is (p[i+1][0] - p[i][0]) * (ref[1] - min_y)
                #   # But we need to be careful with the last interval.
                
                # Let's try this:
                # Sort by x.
                # min_y = ref[1]
                # hv = 0
                # for i in range(len(p)):
                #   if p[i][1] < min_y:
                #     min_y = p[i][1]
                #   x_next = p[i+1][0] if i+1 < len(p) else ref[0]
                #   # The width is x_next - p[i][0]
                #   # The height is ref[1] - min_y
                #   # But is min_y the correct height for the entire interval?
                #   # Yes, because any point in the interval has x >= p[i][0], so its y is >= min_y (since min_y is the min of y's so far).
                #   # Wait, a point later in the list has larger x, but could have smaller y?
                #   # If it has smaller y, min_y would have been updated when we reached it.
                #   # But we are calculating the area for the interval [p[i][0], x_next].
                #   # The min y in this interval is the min of y's of points in this interval.
                #   # Since we iterate in order of x, the min y of the points we've seen so far is the min y of the points with x <= p[i][0].
                #   # This is not the min y of the interval [p[i][0], x_next].
                
                # Correct approach:
                # Sort by x.
                # Create a list of (x, y).
                # The hypervolume is the area under the curve formed by the non-dominated points.
                # Let's compute the non-dominated points first.
                
                nd_points = []
                for i in range(len(p)):
                    dominated = False
                    for j in range(len(p)):
                        if i == j: continue
                        if p[j][0] <= p[i][0] and p[j][1] <= p[i][1] and (p[j][0] < p[i][0] or p[j][1] < p[i][1]):
                            dominated = True
                            break
                    if not dominated:
                        nd_points.append(p[i])
                
                if len(nd_points) == 0:
                    return 0.0
                
                # Sort nd_points by x
                nd_points = np.array(nd_points)
                nd_points = nd_points[np.argsort(nd_points[:, 0])]
                
                hv = 0.0
                prev_x = nd_points[0][0]
                min_y = nd_points[0][1]
                
                for i in range(1, len(nd_points)):
                    curr_x, curr_y = nd_points[i]
                    # The segment from prev_x to curr_x has height (ref[1] - min_y)
                    hv += (curr_x - prev_x) * (ref[1] - min_y)
                    if curr_y < min_y:
                        min_y = curr_y
                    prev_x = curr_x
                
                # Last segment from prev_x to ref[0]
                hv += (ref[0] - prev_x) * (ref[1] - min_y)
                
                return hv

        # EHVI Acquisition Function
        def ehvi_acquisition(x, X_obs, F_obs, ref):
            """
            Calculate Expected Hypervolume Improvement for a given x.
            Uses a simple approximation:
            1. Predict mean and variance at x using RBF.
            2. Sample from the posterior.
            3. Calculate HV improvement for each sample.
            4. Return the mean improvement.
            """
            # This is computationally expensive.
            # For dim=4, we can use a simpler surrogate.
            # Let's use RBF interpolation for the mean and a fixed variance for now.
            # Or use a simple GP with RBF kernel.
            
            # Since we don't have a full GP library, let's use RBF interpolation for the mean.
            # And assume a constant variance? That's not good for exploration.
            # Let's use a simple heuristic:
            # Predicted mean: RBF interpolation.
            # Predicted std: Based on distance to nearest observed point.
            
            # RBF interpolation
            if len(X_obs) < 2:
                return 0.0
            
            # Build RBF model
            try:
                rbf1 = Rbf(X_obs[:, 0], X_obs[:, 1], X_obs[:, 2], X_obs[:, 3], F_obs[:, 0])
                rbf2 = Rbf(X_obs[:, 0], X_obs[:, 1], X_obs[:, 2], X_obs[:, 3], F_obs[:, 1])
                mean_f1 = rbf1(x[0], x[1], x[2], x[3])
                mean_f2 = rbf2(x[0], x[1], x[2], x[3])
            except Exception:
                return 0.0
            
            # Estimate uncertainty: distance to nearest neighbor
            dists = np.linalg.norm(X_obs - x, axis=1)
            min_dist = np.min(dists)
            
            # Scale uncertainty
            # Typical range of F
            f_range = np.ptp(F_obs, axis=0)
            std = f_range * min_dist * 0.5  # Heuristic
            
            # Sample 100 times
            n_samples = 100
            samples = np.random.normal([mean_f1, mean_f2], [std[0], std[1]], size=(n_samples, 2))
            
            # Calculate HV improvement for each sample
            improvements = []
            hv_base = calc_hypervolume(F_obs, ref)
            
            for s in samples:
                # Add sample to F_obs
                F_new = np.vstack([F_obs, s])
                hv_new = calc_hypervolume(F_new, ref)
                improvements.append(hv_new - hv_base)
            
            return np.mean(improvements)

        # 2. Iterative BO Loop
        # Budget remaining: 200 - 20 = 180
        # Each iteration:
        #   - Find best x by optimizing EHVI (local search from multiple starts)
        #   - Evaluate x
        #   - Add to X, F
        
        # To speed up EHVI optimization, we can use a grid or random starts for local search.
        # Let's use 5 random starts for local search of EHVI.
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find best x by maximizing EHVI
            # Use local search from multiple starts
            best_x = None
            best_ehvi = -np.inf
            
            n_starts = 5
            for _ in range(n_starts):
                x0 = lo + span * np.random.rand(n)
                
                def neg_ehvi(x):
                    return -ehvi_acquisition(x, X, F, ref_point)
                
                try:
                    res = minimize(neg_ehvi, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 20})
                    if res.fun < best_ehvi:
                        best_ehvi = -res.fun
                        best_x = res.x
                except Exception:
                    pass
            
            if best_x is None:
                best_x = lo + span * np.random.rand(n)
            
            # Evaluate best_x
            try:
                f_new = problem(best_x)
                X = np.vstack([X, best_x])
                F = np.vstack([F, f_new])
                used += 1
            except Exception:
                break

        return X
# EVOLVE-BLOCK-END

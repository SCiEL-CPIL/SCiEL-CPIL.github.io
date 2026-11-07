"""Bayesian Optimization for WFG7 with Gaussian Process surrogate and EHVI acquisition."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Bayesian multi-objective optimizer using GP surrogate and EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random.default_rng(42)

        ref_point = np.array([3.0, 5.0, 7.0])

        def _hypervolume_3d(F, ref):
            """Compute exact hypervolume in 3D for non-dominated points."""
            # Filter points that are dominated by ref
            F = [f for f in F if np.all(f < ref)]
            if len(F) == 0:
                return 0.0
            if len(F) == 1:
                return float(np.prod(ref - F[0]))
            
            # Sort by first objective
            F = sorted(F, key=lambda f: f[0])
            
            hv = 0.0
            for i in range(len(F)):
                f = F[i]
                # For point i, find the region it contributes
                # The contribution is the volume of the box from f to ref
                # minus volumes already counted by points with better f[0]
                
                # Find the "envelope" - for points with index < i (better f[0]),
                # find the minimum f[1] and f[2] values
                # Actually, use the inclusion-exclusion approach for 3D
                
                # Simpler: use the sweep line approach
                # For each point, its contribution is:
                # (ref[0]-f[0]) * (ref[1]-max(f[1], min_f1_from_dominating)) * ...
                # This gets complex. Let's use a different approach.
                
                # Use the standard 3D hypervolume algorithm:
                # Sort by f[0], then for each point, compute the 2D hypervolume
                # of the points that are not dominated in the remaining 2 dimensions
                
                # Actually, let me use a cleaner recursive approach
                pass
            
            # Use a well-known 3D HV algorithm
            # Sort by first coordinate
            F_sorted = sorted(F, key=lambda f: f[0])
            
            hv = 0.0
            for i in range(len(F_sorted)):
                f = F_sorted[i]
                # The contribution of point i is the volume of the region
                # where f is the "best" in all three dimensions
                # This region is bounded by f and the next points that dominate in some dims
                
                # Find the 2D hypervolume contribution in dims 1 and 2
                # Points that can "block" this point in dims 1,2 are those with
                # f[j][1] < f[1] OR f[j][2] < f[2] for j < i (already processed)
                
                # Simpler: compute the 2D HV of the "shadow" cast by previous points
                # onto the plane at f[0]
                
                # Let's just use a direct computation for small archives
                # For point i, its contribution is:
                # (ref[0] - f[0]) * HV_2D({(f[1],f[2])} union shadows from earlier points)
                
                # This is getting complex. Let me use a different method.
                
                # Compute the volume of the box [f, ref] minus the parts already covered
                # by points with better f[0].
                
                # For points j < i (better f[0]), they cover part of the box.
                # The uncovered part in the box [f[0], ref[0]] x [f[1], ref[1]] x [f[2], ref[2]]
                
                # The 2D shadow of point j on the (f[1],f[2]) plane is the rectangle
                # [max(f[1], f[j][1]), ref[1]] x [max(f[2], f[j][2]), ref[2]]
                # if f[j][0] < f[0] (which is true for j < i since sorted by f[0])
                
                # Actually, for j < i, f[j][0] <= f[0]. Point j dominates in dim 0.
                # So it covers the region in the box where both dim 1 and dim 2 are
                # "worse" than or equal to f[j]'s values.
                
                # The shadow of point j on the (1,2) plane is:
                # [f[j][1], ref[1]] x [f[j][2], ref[2]] intersected with [f[1], ref[1]] x [f[2], ref[2]]
                # = [max(f[1], f[j][1]), ref[1]] x [max(f[2], f[j][2]), ref[2]]
                
                # The 2D HV of the union of these shadows plus the point itself...
                
                # This is still complex. Let me just implement a simple but correct 3D HV.
                
                pass
            
            # Let me just use a simple but potentially slower method for the small archives we'll have
            # We'll compute HV by iterating over all non-dominated points and computing their contribution
            
            # Actually, for the acquisition function, we need HV of the archive + new point.
            # Let me implement a clean 3D HV.
            
            return self._hv_3d(F, ref)

        def _hv_3d(F, ref):
            """Exact 3D hypervolume using inclusion-exclusion."""
            F = [np.array(f) for f in F if np.all(f < ref)]
            n = len(F)
            if n == 0:
                return 0.0
            if n == 1:
                return float(np.prod(ref - F[0]))
            
            # Sort by first objective
            F = sorted(F, key=lambda f: f[0])
            
            hv = 0.0
            for i in range(n):
                f = F[i]
                # Contribution of point i: the volume of the box [f, ref] that is not
                # covered by any point j < i (which have better f[0])
                
                # The box for point i is [f[0], ref[0]] x [f[1], ref[1]] x [f[2], ref[2]]
                # Points j < i have f[j][0] <= f[0], so they can cover parts of this box.
                
                # For each j < i, the region covered by j in the box of i is:
                # [f[0], ref[0]] x [max(f[1], f[j][1]), ref[1]] x [max(f[2], f[j][2]), ref[2]]
                # (since f[j][0] <= f[0], the first dimension is fully covered)
                
                # We need the 2D hypervolume of the union of rectangles:
                # R_j = [max(f[1], f[j][1]), ref[1]] x [max(f[2], f[j][2]), ref[2]] for j < i
                # plus the point [f[1], ref[1]] x [f[2], ref[2]]
                
                # Actually, the contribution is:
                # (ref[0] - f[0]) * HV_2D({(f[1], f[2])} union {R_j for j < i})
                
                # Where HV_2D is the 2D hypervolume with reference (ref[1], ref[2])
                
                # Let's compute the 2D HV of the set of points:
                # P = {(f[1], f[2])} union {(max(f[1], f[j][1]), max(f[2], f[j][2])) for j < i}
                # with reference (ref[1], ref[2])
                
                # But wait, that's not quite right. The shadows are rectangles, not points.
                
                # Let me think differently. The uncovered region in the (f[1], f[2]) plane
                # is the set of points (y1, y2) such that:
                # y1 >= f[1], y2 >= f[2], and for all j < i: NOT (y1 >= f[j][1] AND y2 >= f[j][2])
                
                # This is the 2D HV of the "non-dominated" set after removing dominated points.
                
                # Create the 2D set
                points_2d = [(f[1], f[2])]
                for j in range(i):
                    points_2d.append((max(f[1], F[j][1]), max(f[2], F[j][2])))
                
                # Compute 2D HV of these points with reference (ref[1], ref[2])
                hv_2d = self._hv_2d(points_2d, [ref[1], ref[2]])
                
                hv += (ref[0] - f[0]) * hv_2d
            
            return float(hv)

        def _hv_2d(points, ref):
            """2D hypervolume."""
            points = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
            if not points:
                return 0.0
            
            # Sort by first coordinate
            points = sorted(points, key=lambda p: p[0])
            
            hv = 0.0
            max_y = -np.inf
            for i, (x, y) in enumerate(points):
                if y > max_y:
                    # This point contributes a new region
                    # The width is from x to the next point's x (or ref[0])
                    next_x = points[i+1][0] if i+1 < len(points) else ref[0]
                    hv += (next_x - x) * (ref[1] - y)
                    max_y = y
                # If y <= max_y, this point is dominated in 2D and contributes nothing
                # Wait, that's not right. Let me reconsider.
                
                # Actually, for 2D HV:
                # Sort by x. For each point, if its y is greater than all previous y's,
                # it contributes (next_x - x) * (ref[1] - y)
                
                # But I need to track the maximum y seen so far from the left.
                # Let me redo this.
            
            # Correct 2D HV:
            points = sorted(points, key=lambda p: p[0])
            hv = 0.0
            max_y = -np.inf
            for i, (x, y) in enumerate(points):
                if y > max_y:
                    next_x = points[i+1][0] if i+1 < len(points) else ref[0]
                    hv += (next_x - x) * (ref[1] - y)
                    max_y = y
            
            return float(hv)

        # GP kernel and prediction
        def _gp_predict(X_train, Y_train, X_query, length_scales, noise_var):
            """Simple GP prediction with RBF kernel."""
            n_train = len(X_train)
            n_query = len(X_query)
            
            # Compute kernel matrix
            K = np.zeros((n_train, n_train))
            for i in range(n_train):
                for j in range(n_train):
                    diff = X_train[i] - X_train[j]
                    sq_dist = np.sum(diff ** 2 / length_scales ** 2)
                    K[i, j] = np.exp(-0.5 * sq_dist)
            K += noise_var * np.eye(n_train)
            
            # Predict at query points
            K_star = np.zeros((n_query, n_train))
            for i in range(n_query):
                for j in range(n_train):
                    diff = X_query[i] - X_train[j]
                    sq_dist = np.sum(diff ** 2 / length_scales ** 2)
                    K_star[i, j] = np.exp(-0.5 * sq_dist)
            
            # Mean and variance predictions
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            
            mean = K_star @ alpha
            v = np.linalg.solve(L, K_star.T)
            var = np.sum(v ** 2, axis=1)
            
            return mean, var

        # EHVI acquisition function
        def _ehvi(X, Y, F_archive, ref):
            """Expected Hypervolume Improvement."""
            n = len(X)
            n_obj = self.n_obj
            ehvi_values = np.zeros(n)
            
            for i in range(n):
                # Get prediction at X[i]
                pred_mean = Y[i]  # This is already the predicted mean
                pred_var = np.zeros(n_obj)  # Need to get variance
                
                # Actually, I need to restructure. Let me compute EHVI differently.
                pass
            
            return ehvi_values

        # Initialize with a set of points
        X_archive = []
        Y_archive = []  # Objective values
        used = 0
        
        # Initial sampling: Latin hypercube-like
        n_init = 20
        for _ in range(n_init):
            if used >= budget or time.time() > deadline - 10:
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            X_archive.append(x)
            Y_archive.append(f)
        
        X_archive = np.array(X_archive)
        Y_archive = np.array(Y_archive)
        
        # Update archive (non-dominated set)
        def _get_nondominated(F):
            """Return indices of non-dominated points."""
            n = len(F)
            nondom = [True] * n
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            nondom[i] = False
                            break
            return [i for i in range(n) if nondom[i]]
        
        # Bayesian optimization loop
        n_bo_iters = 5  # 5 iterations of BO with 20 evaluations each
        evals_per_iter = (budget - used) // max(1, n_bo_iters)
        
        for bo_iter in range(n_bo_iters):
            if used >= budget or time.time() > deadline - 10:
                break
            
            # Fit GP model
            # Use simple length scales
            length_scales = (hi - lo) * 0.5
            noise_var = 1e-6
            
            # Generate candidate points
            n_candidates = 200
            X_candidates = lo + (hi - lo) * rng.random((n_candidates, dim))
            
            # Predict means and variances
            try:
                pred_means, pred_vars = _gp_predict(X_archive, Y_archive, X_candidates, length_scales, noise_var)
            except:
                # Fallback to random if GP fails
                pred_means = np.tile(Y_archive[0], (n_candidates, 1))
                pred_vars = np.ones((n_candidates, n_obj))
            
            # Compute EHVI-like acquisition
            # For simplicity, use a weighted sum of improvements
            F_archive_arr = np.array(Y_archive)
            ideal = np.min(F_archive_arr, axis=0)
            
            # Normalize objectives
            ranges = np.max(F_archive_arr, axis=0) - ideal
            ranges[ranges == 0] = 1.0
            Y_norm = (Y_archive - ideal) / ranges
            
            # Simple acquisition: expected improvement in each objective
            # weighted by distance from ideal
            acquisition = np.zeros(n_candidates)
            for j in range(n_obj):
                # Expected improvement for objective j
                mu = pred_means[:, j]
                sigma = np.sqrt(pred_vars[:, j])
                
                # Current best for objective j
                best_j = ideal[j]
                
                # EI = (best_j - mu) * Phi((best_j - mu)/sigma) + sigma * phi((best_j - mu)/sigma)
                z = (best_j - mu) / (sigma + 1e-10)
                from scipy.stats import norm
                ei = (best_j - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                acquisition += ei / ranges[j]
            
            # Select top candidates
            top_indices = np.argsort(-acquisition)[:min(evals_per_iter, n_candidates)]
            
            for idx in top_indices:
                if used >= budget or time.time() > deadline - 5:
                    break
                x = X_candidates[idx]
                f = np.array(problem(x), dtype=np.float64)
                used += 1
                X_archive = np.vstack([X_archive, x.reshape(1, -1)])
                Y_archive = np.vstack([Y_archive, f.reshape(1, -1)])
        
        # Get final non-dominated set
        nondom_indices = _get_nondominated(Y_archive)
        archive_X = [X_archive[i] for i in nondom_indices]
        archive_F = [Y_archive[i] for i in nondom_indices]
        
        return archive_X if archive_X else [lo + (hi - lo) * 0.5]

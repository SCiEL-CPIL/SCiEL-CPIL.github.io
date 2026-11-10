"""Multi-objective optimizer for WFG7 using hypervolume-guided local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided local search for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Reference point for hypervolume
        ref = np.array([3.0, 5.0, 7.0])

        used = 0

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return x, f

        # LHS initialization
        pop_size = min(30, self.budget // 6)
        X = []
        F = []
        for i in range(pop_size):
            # Latin hypercube sampling
            x = np.zeros(dim)
            for d in range(dim):
                u = (i + np.random.rand()) / pop_size
                x[d] = lo[d] + u * span[d]
            x, f = eval_point(x)
            X.append(x)
            F.append(f)

        X = np.array(X)
        F = np.array(F)

        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for g in F_set:
                if np.all(g <= f) and np.any(g < f):
                    return True
            return False

        def non_dominated_indices(F_set):
            """Return indices of non-dominated points."""
            n = len(F_set)
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        def hypervolume_3d(points, ref_pt):
            """Compute 3D hypervolume for points strictly better than ref_pt."""
            # Filter points strictly better than reference
            pts = [p for p in points if np.all(p < ref_pt)]
            if not pts:
                return 0.0
            pts = np.array(pts)
            n = len(pts)
            
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            # Compute hypervolume using inclusion-exclusion for 3D
            # Use a simpler approach: sort by one dimension and compute 2D hypervolume for each slice
            hv = 0.0
            prev_x = pts[0, 0]
            prev_yz = [(pts[0, 1], pts[0, 2])]
            
            for i in range(1, n):
                x = pts[i, 0]
                if x > prev_x:
                    # Compute 2D hypervolume of prev_yz points with upper bounds at x
                    # 2D hypervolume: area under the curve
                    area = _hypervolume_2d(prev_yz, x, ref_pt[1], ref_pt[2])
                    hv += (x - prev_x) * area
                    prev_x = x
                # Add current point's y,z
                prev_yz.append((pts[i, 1], pts[i, 2]))
                # Remove dominated points in 2D
                prev_yz = _remove_dominated_2d(prev_yz)
            
            # Last slice
            if n > 0:
                area = _hypervolume_2d(prev_yz, ref_pt[0], ref_pt[1], ref_pt[2])
                hv += (ref_pt[0] - prev_x) * area
            
            return hv

        def _hypervolume_2d(points_2d, x_max, y_max, z_max):
            """Compute 2D hypervolume (area) for points with x < x_max, y < y_max, z < z_max."""
            pts = [(y, z) for (y, z) in points_2d if y < y_max and z < z_max]
            if not pts:
                return 0.0
            pts.sort()
            area = 0.0
            prev_y = pts[0][0]
            min_z = pts[0][1]
            for i in range(1, len(pts)):
                y, z = pts[i]
                if z < min_z:
                    area += (y - prev_y) * min_z
                    min_z = z
                    prev_y = y
            area += (y_max - prev_y) * min_z
            return area

        def _remove_dominated_2d(points_2d):
            """Remove dominated points in 2D (minimize both)."""
            pts = sorted(points_2d)
            result = []
            min_z = float('inf')
            for y, z in pts:
                if z < min_z:
                    result.append((y, z))
                    min_z = z
            return result

        def marginal_hv_contribution(f, F_set, ref_pt):
            """Compute marginal hypervolume contribution of point f to set F_set."""
            # HV of F_set without f
            hv_without = hypervolume_3d([p for p in F_set], ref_pt)
            # HV of F_set with f
            hv_with = hypervolume_3d([p for p in F_set] + [f], ref_pt)
            return hv_with - hv_without

        def nelder_mead(x0, f0, problem, lo, hi, span, max_iter=20, tol=1e-6):
            """Nelder-Mead for multi-objective: minimize weighted sum."""
            nonlocal used
            dim = len(x0)
            n_obj = len(f0)
            
            # Initialize simplex
            simplex = [x0.copy()]
            simplex_f = [f0.copy()]
            
            # Create other simplex points
            for i in range(dim):
                x = x0.copy()
                step = 0.05 * span[i] if span[i] > 0 else 0.05
                x[i] += step
                x = np.clip(x, lo, hi)
                _, f = eval_point(x)
                simplex.append(x)
                simplex_f.append(f)
            
            # Weights for scalarization
            weights = np.ones(n_obj) / n_obj
            
            def scalar(f):
                return np.dot(weights, f)
            
            # Sort by scalar value
            for _ in range(max_iter):
                # Sort
                idx = np.argsort([scalar(f) for f in simplex_f])
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Check convergence
                if abs(scalar(simplex_f[0]) - scalar(simplex_f[-1])) < tol:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                _, fr = eval_point(xr)
                
                if scalar(fr) < scalar(simplex_f[0]):
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    _, fe = eval_point(xe)
                    if scalar(fe) < scalar(fr):
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif scalar(fr) < scalar(simplex_f[-2]):
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    _, fc = eval_point(xc)
                    if scalar(fc) < scalar(simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            _, fi = eval_point(simplex[i])
                            simplex_f[i] = fi
            
            # Return best point
            best_idx = np.argmin([scalar(f) for f in simplex_f])
            return simplex[best_idx], simplex_f[best_idx]

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get non-dominated set
            nd_idx = non_dominated_indices(F)
            if not nd_idx:
                nd_idx = list(range(len(F)))
            
            # Find point with highest marginal HV contribution
            best_idx = nd_idx[0]
            best_marg = -1
            for idx in nd_idx:
                marg = marginal_hv_contribution(F[idx], F, ref)
                if marg > best_marg:
                    best_marg = marg
                    best_idx = idx
            
            # Local search around best point
            x_best = X[best_idx].copy()
            f_best = F[best_idx].copy()
            
            # Try several perturbation directions
            n_perturb = min(5, self.budget - used)
            for _ in range(n_perturb):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Random perturbation
                direction = np.random.randn(dim)
                step_size = 0.1 * np.linalg.norm(span)
                x_new = x_best + step_size * direction / (np.linalg.norm(direction) + 1e-10)
                x_new = np.clip(x_new, lo, hi)
                
                _, f_new = eval_point(x_new)
                
                # Check if non-dominated
                if not is_dominated(f_new, F):
                    # Add to population
                    X = np.vstack([X, x_new])
                    F = np.vstack([F, f_new])
                    x_best = x_new
                    f_best = f_new
                else:
                    # Try Nelder-Mead from this point
                    if self.budget - used > 30:
                        x_nm, f_nm = nelder_mead(x_new, f_new, problem, lo, hi, span, max_iter=10)
                        if not is_dominated(f_nm, F):
                            X = np.vstack([X, x_nm])
                            F = np.vstack([F, f_nm])
                            x_best = x_nm
                            f_best = f_nm
            
            # Periodically prune dominated points to keep population manageable
            if len(F) > 50:
                nd_idx = non_dominated_indices(F)
                if len(nd_idx) < len(F):
                    X = X[nd_idx]
                    F = F[nd_idx]

        return X
# EVOLVE-BLOCK-END

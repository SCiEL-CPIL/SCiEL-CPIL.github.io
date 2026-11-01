"""Program for covering the sphere with 130 equal caps using repulsion-based optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute grids for the smooth objective
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(1200)
GRID_MEDIUM = _make_fib_grid(3000)
GRID_FINE = _make_fib_grid(5000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_exact(pts):
        """Compute covering radius exactly as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return float(np.max(radii))
        except Exception:
            # Fallback: dense grid-based estimate
            dot = np.dot(GRID_FINE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def compute_covering_radius_fast(pts):
        """Fast grid-based covering radius estimate for local search."""
        dot = np.dot(GRID_MEDIUM, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)
        min_dists = np.min(ang_dists, axis=1)
        return float(np.max(min_dists))

    def get_largest_facet_indices(pts):
        """Return indices of points involved in the largest facets."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            # Get top 5 largest facets
            top_k = min(5, len(radii))
            top_indices = np.argsort(radii)[-top_k:]
            involved_points = set()
            for idx in top_indices:
                involved_points.add(int(faces[idx, 0]))
                involved_points.add(int(faces[idx, 1]))
                involved_points.add(int(faces[idx, 2]))
            return list(involved_points)
        except Exception:
            return list(range(N))

    def repulsion_energy_grad(pts, exponent=4.0):
        """Compute repulsion energy and gradient for uniform distribution."""
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        
        w_grad = exponent * (dist_sq ** (-exponent/2 - 1))
        np.fill_diagonal(w_grad, 0.0)
        
        grad = w_grad @ pts
        
        return float(np.sum(w)), grad

    def gradient_descent_repulsion(pts, exponent, alpha_init, decay, max_iter):
        """Optimize points by gradient descent on repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            energy, grad = repulsion_energy_grad(pts, exponent)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            pts = pts + alpha * grad
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            alpha *= decay
        
        return pts

    def grid_objective_and_grad(x_flat, grid_pts):
        """
        Smooth objective: soft-max of angular distances from grid points to nearest configuration point.
        Uses a vectorized, differentiable approximation.
        x_flat: flattened (N,3) array of points.
        grid_pts: precomputed grid of sphere points.
        Returns (value, gradient).
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        M = len(grid_pts)

        # Compute angular distances: M x N
        dot = np.dot(grid_pts, pts_norm.T)  # M x N
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # M x N

        # 1. Soft-min over points for each grid point
        # s_j = -1/T_min * log( sum_k exp(-T_min * ang_dists[j, k]) )
        T_min = 40.0
        exp_neg_T_min_d = np.exp(-T_min * ang_dists) # M x N
        sum_exp_neg = np.sum(exp_neg_T_min_d, axis=1) # M
        soft_min_dists = -1.0 / T_min * np.log(sum_exp_neg) # M
        
        # Weights for soft-min: w_jk = exp(-T_min * d_jk) / sum_k exp(-T_min * d_jk)
        w_jk = exp_neg_T_min_d / sum_exp_neg[:, np.newaxis] # M x N

        # 2. Soft-max over grid points
        # val = max_d + 1/T_max * log(sum_j exp(T_max * (s_j - max_d)))
        T_max = 40.0
        max_s = np.max(soft_min_dists)
        exp_T_max_s = np.exp(T_max * (soft_min_dists - max_s)) # M
        sum_exp_pos = np.sum(exp_T_max_s)
        val = max_s + 1.0 / T_max * np.log(sum_exp_pos)
        
        # Weights for soft-max: v_j = exp(T_max * (s_j - max_s)) / sum_exp_pos
        v_j = exp_T_max_s / sum_exp_pos # M

        # Gradient of val w.r.t pts_norm[k]
        # d(val)/d(pts_norm[k]) = sum_j v_j * (d(s_j)/d(pts_norm[k]))
        # = sum_j v_j * sum_k' w_jk' * (d(d_jk')/d(pts_norm[k']))
        # d(d_jk)/d(pts_norm[k]) = - grid_pts[j] / sqrt(1 - dot^2)
        
        # Let's compute this vectorized.
        denom = np.sqrt(np.maximum(1.0 - dot**2, 1e-10)) # M x N
        # factor = v_j[:, np.newaxis] * w_jk / denom # M x N
        # grad_pts_norm[k] = sum_j factor[j,k] * (-grid_pts[j])
        
        factor = (v_j[:, np.newaxis] * w_jk) / denom # M x N
        grad_pts_norm = -(factor.T @ grid_pts) # N x 3
        
        # Chain rule: d(pts_norm)/d(pts) = (I - pts_norm pts_norm^T) / norm
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            # d(pts_norm[i])/d(pts[i]) = (I - p p^T)/n
            grad_i = (grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p) / n
            grad[i*3:(i+1)*3] = grad_i
        
        return float(val), grad

    def optimize_with_lbfgsb(points, grid_pts, maxiter=200):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        def objective(x):
            val, _ = grid_objective_and_grad(x, grid_pts)
            return val
        
        def gradient(x):
            _, g = grid_objective_and_grad(x, grid_pts)
            return g
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 400, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 4
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100 + restart)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.08 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        points = gradient_descent_repulsion(points, 4.0, 0.05, 0.98, 150)
        
        # Then optimize with L-BFGS-B on the grid-based objective (coarse grid)
        time_left = deadline - time.time() - 15.0
        if time_left > 15.0:
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=250)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with L-BFGS-B using medium grid
    if best_points is not None and time.time() < deadline - 30.0:
        for trial in range(4):
            if time.time() > deadline - 15.0:
                break
            rng = np.random.RandomState(500 + trial)
            candidate = best_points + 0.03 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 10.0
            if time_left > 8.0:
                candidate, _ = optimize_with_lbfgsb(candidate, GRID_MEDIUM, maxiter=150)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Fast coordinate-wise local search using grid-based estimate
    if best_points is not None and time.time() < deadline - 15.0:
        points = best_points.copy()
        improved = True
        max_rounds = 2
        round_count = 0
        
        while improved and round_count < max_rounds and time.time() < deadline - 10.0:
            improved = False
            round_count += 1
            
            for i in range(N):
                if time.time() > deadline - 10.0:
                    break
                
                old_p = points[i].copy()
                
                # Try perturbing point i in random directions
                rng = np.random.RandomState(i * 1000 + round_count)
                for trial in range(4):
                    direction = rng.randn(3)
                    direction -= np.dot(direction, old_p) * old_p  # Project to tangent
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        continue
                    direction /= dir_norm
                    
                    for step in [0.015, 0.008, 0.004, 0.002]:
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        points[i] = new_p
                        
                        rad = compute_covering_radius_fast(points)
                        if rad < best_rad - 1e-8:
                            # Verify with exact computation
                            rad_exact = compute_covering_radius_exact(points)
                            if rad_exact < best_rad - 1e-10:
                                best_rad = rad_exact
                                best_points = points.copy()
                                improved = True
                                break
                            else:
                                points[i] = old_p
                        else:
                            points[i] = old_p
                    else:
                        continue
                    break
                
                # If we improved, keep the new position
                # Otherwise, restore
                if not improved:
                    points[i] = old_p

    # Phase 4: Targeted polish on largest facet points
    if best_points is not None and time.time() < deadline - 10.0:
        involved = get_largest_facet_indices(best_points)
        points = best_points.copy()
        
        for i in involved:
            if time.time() > deadline - 8.0:
                break
            
            old_p = points[i].copy()
            rng = np.random.RandomState(9000 + i)
            
            for trial in range(6):
                direction = rng.randn(3)
                direction -= np.dot(direction, old_p) * old_p
                dir_norm = np.linalg.norm(direction)
                if dir_norm < 1e-10:
                    continue
                direction /= dir_norm
                
                for step in [0.01, 0.005, 0.002, 0.001]:
                    new_p = old_p + step * direction
                    new_p /= np.linalg.norm(new_p)
                    points[i] = new_p
                    
                    rad = compute_covering_radius_exact(points)
                    if rad < best_rad - 1e-10:
                        best_rad = rad
                        best_points = points.copy()
                        break
                    else:
                        points[i] = old_p
                else:
                    continue
                break
            
            if not (rad < best_rad + 1e-10):
                points[i] = old_p

    # Phase 5: Final polish with L-BFGS-B using fine grid
    if best_points is not None and time.time() < deadline - 5.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 3.0
        if time_left > 3.0:
            final, _ = optimize_with_lbfgsb(final, GRID_FINE, maxiter=40)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END

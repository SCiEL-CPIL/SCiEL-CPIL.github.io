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

GRID_COARSE = _make_fib_grid(2000)
GRID_MEDIUM = _make_fib_grid(5000)
GRID_FINE = _make_fib_grid(8000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety

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

    def smooth_max_circumradius_and_grad(pts, T=50.0):
        """
        Smooth surrogate for max circumradius over convex hull faces.
        Uses log-sum-exp over face circumradii for differentiability.
        Returns (value, gradient w.r.t. pts).
        """
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
        except Exception:
            # Fallback: use grid-based
            return _grid_smooth_obj(pts, GRID_MEDIUM, T)
        
        F = len(faces)
        a = pts[faces[:, 0]]
        b = pts[faces[:, 1]]
        c = pts[faces[:, 2]]
        
        # Compute circumcenter direction and circumradius for each face
        ab = np.cross(a, b)
        bc = np.cross(b, c)
        ca = np.cross(c, a)
        c_dir = ab + bc + ca
        norms_c = np.linalg.norm(c_dir, axis=1)
        
        radii = np.zeros(F)
        c_dirs_norm = np.zeros((F, 3))
        
        valid = norms_c > 1e-10
        if np.any(valid):
            c_dirs_valid = c_dir[valid] / norms_c[valid, np.newaxis]
            c_dirs_norm[valid] = c_dirs_valid
            cos_r = np.clip(np.sum(c_dirs_valid * a[valid], axis=1), -1.0, 1.0)
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
        
        # Log-sum-exp for smooth max
        max_r = np.max(radii)
        exp_T_r = np.exp(T * (radii - max_r))
        sum_exp = np.sum(exp_T_r)
        val = max_r + (1.0 / T) * np.log(sum_exp)
        
        # Weights for each face
        w = exp_T_r / sum_exp  # F
        
        # Gradient: d(val)/d(pts)
        # For valid faces: r = arccos(c_dir · a), where c_dir = (ab+bc+ca)/|ab+bc+ca|
        # dr/da = -c_dir / sqrt(1 - (c_dir·a)^2)  (simplified, ignoring derivative of c_dir)
        # For a more accurate gradient, we need to account for c_dir dependence on a, b, c
        # But for a first-order approximation, treat c_dir as fixed
        
        grad = np.zeros_like(pts)
        
        if np.any(valid):
            valid_idx = np.where(valid)[0]
            cos_r_valid = np.clip(np.sum(c_dirs_norm[valid] * a[valid], axis=1), -1.0, 1.0)
            sin_r_valid = np.sqrt(np.maximum(1.0 - cos_r_valid**2, 1e-10))
            
            # dr/da for each vertex of valid faces
            # r = arccos(c_dir · a), so dr/da = -c_dir / sin(r)
            # But c_dir also depends on a, b, c. For simplicity, use the simplified gradient.
            dr_da = -c_dirs_norm[valid] / sin_r_valid[:, np.newaxis]  # (n_valid, 3)
            
            # Accumulate gradient contributions
            face_w = w[valid]  # (n_valid,)
            for k in range(len(valid_idx)):
                fi = valid_idx[k]
                w_k = face_w[k]
                # Gradient contribution to vertices a, b, c of face fi
                # d(val)/d(pts[a]) += w_k * T * dr_da
                # But we also need to account for the fact that moving a changes c_dir
                # For now, use the simplified gradient
                grad[faces[fi, 0]] += w_k * T * dr_da[k]
                # For b and c, the gradient is more complex; approximate as zero for now
                # A better approach would compute full derivatives
        
        return float(val), grad
    
    def _grid_smooth_obj(pts, grid_pts, T=50.0):
        """Fallback grid-based smooth objective."""
        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)
        min_dists = np.min(ang_dists, axis=1)
        
        max_d = np.max(min_dists)
        exp_T = np.exp(T * (min_dists - max_d))
        sum_exp = np.sum(exp_T)
        val = max_d + (1.0 / T) * np.log(sum_exp)
        
        # Gradient
        w = exp_T / sum_exp
        # For each grid point, find nearest point
        nearest_idx = np.argmin(ang_dists, axis=1)
        grad = np.zeros_like(pts)
        for j in range(len(grid_pts)):
            i = nearest_idx[j]
            p = pts[i]
            gp = grid_pts[j]
            denom = np.sqrt(max(1.0 - np.dot(p, gp)**2, 1e-10))
            grad[i] -= (w[j] * T / denom) * gp
        
        return float(val), grad

    def optimize_with_lbfgsb(points, grid_pts, maxiter=200, use_hull=False):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        def objective(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1e-10
            pts_norm = pts / norms
            if use_hull:
                val, _ = smooth_max_circumradius_and_grad(pts_norm)
            else:
                val, _ = _grid_smooth_obj(pts_norm, grid_pts)
            return val
        
        def gradient(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1e-10
            pts_norm = pts / norms
            if use_hull:
                _, g = smooth_max_circumradius_and_grad(pts_norm)
            else:
                _, g = _grid_smooth_obj(pts_norm, grid_pts)
            return g.flatten()
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 3, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 45.0:
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
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            # Apply strong repulsion to spread out
            points = gradient_descent_repulsion(points, 6.0, 0.1, 0.95, 200)
        else:
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # Optimize with repulsion to get a good starting configuration
        points = gradient_descent_repulsion(points, 4.0, 0.05, 0.98, 100)
        
        # Optimize with L-BFGS-B on the grid-based objective (coarse grid)
        time_left = deadline - time.time() - 20.0
        if time_left > 20.0:
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=150, use_hull=False)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with L-BFGS-B using medium grid and hull-based objective
    if best_points is not None and time.time() < deadline - 35.0:
        for trial in range(4):
            if time.time() > deadline - 20.0:
                break
            rng = np.random.RandomState(500 + trial)
            candidate = best_points + 0.02 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            # First pass: grid-based for speed
            candidate, _ = optimize_with_lbfgsb(candidate, GRID_MEDIUM, maxiter=100, use_hull=False)
            
            # Second pass: hull-based for precision
            if time.time() < deadline - 15.0:
                candidate, _ = optimize_with_lbfgsb(candidate, GRID_MEDIUM, maxiter=80, use_hull=True)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Targeted local search on worst facets
    if best_points is not None and time.time() < deadline - 15.0:
        points = best_points.copy()
        for round_num in range(3):
            if time.time() > deadline - 10.0:
                break
            
            try:
                hull = ConvexHull(points)
                faces = hull.simplices
                a = points[faces[:, 0]]
                b = points[faces[:, 1]]
                c = points[faces[:, 2]]
                
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                c_dir = ab + bc + ca
                norms_c = np.linalg.norm(c_dir, axis=1)
                valid = norms_c > 1e-10
                
                radii = np.zeros(len(faces))
                if np.any(valid):
                    c_dir_valid = c_dir[valid] / norms_c[valid, np.newaxis]
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
                
                # Find worst facets
                worst_indices = np.argsort(radii)[-5:]
                worst_face_ids = set()
                for wi in worst_indices:
                    for vi in faces[wi]:
                        worst_face_ids.add(vi)
                
                # Try improving vertices of worst facets
                for vi in sorted(worst_face_ids):
                    if time.time() > deadline - 10.0:
                        break
                    
                    old_p = points[vi].copy()
                    rng = np.random.RandomState(7000 + vi + round_num * 100)
                    for trial in range(6):
                        direction = rng.randn(3)
                        direction -= np.dot(direction, old_p) * old_p
                        dir_norm = np.linalg.norm(direction)
                        if dir_norm < 1e-10:
                            continue
                        direction /= dir_norm
                        
                        for step in [0.005, 0.002, 0.001]:
                            new_p = old_p + step * direction
                            new_p /= np.linalg.norm(new_p)
                            points[vi] = new_p
                            
                            rad = compute_covering_radius_exact(points)
                            if rad < best_rad - 1e-10:
                                best_rad = rad
                                best_points = points.copy()
                                old_p = new_p.copy()
                                break
                            else:
                                points[vi] = old_p
                        else:
                            continue
                        break
                    points[vi] = old_p
            except Exception:
                break

    # Phase 4: Final high-precision polish with fine grid and hull-based L-BFGS-B
    if best_points is not None and time.time() < deadline - 10.0:
        final = best_points.copy()
        if time.time() < deadline - 8.0:
            final, _ = optimize_with_lbfgsb(final, GRID_FINE, maxiter=60, use_hull=False)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final.copy()
        
        # Hull-based polish
        if best_points is not None and time.time() < deadline - 5.0:
            final = best_points.copy()
            final, _ = optimize_with_lbfgsb(final, GRID_MEDIUM, maxiter=50, use_hull=True)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final.copy()

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END

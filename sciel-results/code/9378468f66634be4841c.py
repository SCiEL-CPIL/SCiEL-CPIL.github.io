"""Program for covering the sphere with 130 equal caps using repulsion-based optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

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
            lat = np.linspace(-np.pi/2, np.pi/2, 100)
            lon = np.linspace(0, 2*np.pi, 200, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def repulsion_energy_grad(pts, exponent=2.0):
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

    def grid_objective_and_grad(x_flat, grid_pts=None):
        """
        Smooth objective: soft-max of angular distances from grid points to nearest configuration point.
        x_flat: flattened (N,3) array of points (not necessarily normalized).
        grid_pts: optional precomputed grid of sphere points.
        Returns (value, gradient).
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        if grid_pts is None:
            lat = np.linspace(-np.pi/2, np.pi/2, 40)
            lon = np.linspace(0, 2*np.pi, 80, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        M = len(grid_pts)

        # Compute angular distances: M x N
        dot = np.dot(grid_pts, pts_norm.T)  # M x N
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # M x N

        # For each grid point, find the min distance to any configuration point
        # Use soft-min for differentiability
        temperature = 30.0
        min_dists = np.min(ang_dists, axis=1)  # M
        
        # Soft-max of min_dists to get a smooth approximation of the covering radius
        max_d = np.max(min_dists)
        shifted = min_dists - max_d
        exp_shifted = np.exp(shifted * temperature)
        log_sum_exp = np.log(np.sum(exp_shifted)) / temperature + max_d
        
        # Gradient computation using full soft-min weights (vectorized)
        argmin_k = np.argmin(ang_dists, axis=1)  # M
        weights = exp_shifted / np.sum(exp_shifted)  # M
        denom = np.sqrt(np.maximum(1.0 - dot[np.arange(M), argmin_k]**2, 1e-10))
        grad_contrib = weights / denom  # M
        
        # Vectorized accumulation: for each grid point j, add to point argmin_k[j]
        # Use np.add.at for unbuffered accumulation
        grad_pts_norm = np.zeros((N, 3))
        # grad_pts_norm[argmin_k] += grad_contrib[:, None] * (-grid_pts)  # doesn't work with repeated indices
        # Use np.add.at
        np.add.at(grad_pts_norm, argmin_k, grad_contrib[:, None] * (-grid_pts))
        
        # Chain rule: d(pts_norm)/d(pts) = (I - pts_norm pts_norm^T) / norm
        # Vectorized: grad_pts_norm[i] - dot(grad_pts_norm[i], p_i)*p_i, then divide by norm_i
        dot_prod = np.sum(grad_pts_norm * pts_norm, axis=1, keepdims=True)  # (N,1)
        grad_projected = (grad_pts_norm - dot_prod * pts_norm) / norms  # (N,3)
        
        grad = grad_projected.flatten()
        
        return float(log_sum_exp), grad

    def optimize_with_lbfgsb(points, max_time=30.0, grid_pts=None):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        call_count = [0]
        start_t = [time.time()]
        
        def objective(x):
            call_count[0] += 1
            if time.time() - start_t[0] > max_time:
                return np.inf
            val, _ = grid_objective_and_grad(x, grid_pts)
            return val
        
        def gradient(x):
            call_count[0] += 1
            if time.time() - start_t[0] > max_time:
                return np.zeros_like(x)
            _, g = grid_objective_and_grad(x, grid_pts)
            return g
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 800, 'ftol': 1e-9, 'gtol': 1e-7}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    def make_fib_grid(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    grid_pts_coarse = make_fib_grid(2000)
    grid_pts_fine = make_fib_grid(8000)

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 50.0:
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
            points = points + 0.5 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        points = gradient_descent_repulsion(points, 2.0, 0.1, 0.99, 100)
        
        # Then optimize with L-BFGS-B on the grid-based objective (coarse grid)
        time_left = deadline - time.time() - 10.0
        if time_left > 10.0:
            points, _ = optimize_with_lbfgsb(points, max_time=min(time_left, 40.0), grid_pts=grid_pts_coarse)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with smaller L-BFGS-B runs using fine grid
    if best_points is not None and time.time() < deadline - 20.0:
        for trial in range(3):
            if time.time() > deadline - 10.0:
                break
            rng = np.random.RandomState(500 + trial)
            candidate = best_points + 0.05 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 8.0
            if time_left > 5.0:
                candidate, _ = optimize_with_lbfgsb(candidate, max_time=min(time_left, 20.0), grid_pts=grid_pts_fine)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Targeted facet refinement
    # Identify the worst facets and try to reduce their circumradius
    if best_points is not None and time.time() < deadline - 15.0:
        points = best_points.copy()
        
        # Find worst facets
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
            
            radii = np.zeros(len(faces))
            valid = norms_c > 1e-10
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms_c[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            # Get top 5 worst facets
            top_k = min(5, len(radii))
            worst_indices = np.argsort(radii)[-top_k:]
        except Exception:
            worst_indices = []
        
        # For each worst facet, try to move its vertices to reduce circumradius
        for wi in worst_indices:
            if time.time() > deadline - 10.0:
                break
            
            face = faces[wi]
            for vi in face:
                old_p = points[vi].copy()
                rng = np.random.RandomState(int(wi * 100 + vi))
                
                # Try moving vertex toward the circumcenter direction
                # The circumcenter of the face points outward; moving vertex
                # slightly toward the face center can reduce the cap
                face_pts = points[face]
                face_center = np.mean(face_pts, axis=0)
                face_center /= np.linalg.norm(face_center)
                
                for step in [0.03, 0.015, 0.008]:
                    # Move toward face center (reduces the cap size)
                    new_p = old_p + step * (face_center - old_p)
                    new_p /= np.linalg.norm(new_p)
                    points[vi] = new_p
                    
                    rad = compute_covering_radius_exact(points)
                    if rad < best_rad - 1e-12:
                        best_rad = rad
                        best_points = points.copy()
                    else:
                        points[vi] = old_p
                        break
                
                # Also try random tangent perturbations
                for trial in range(2):
                    direction = rng.randn(3)
                    direction -= np.dot(direction, old_p) * old_p
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        continue
                    direction /= dir_norm
                    
                    for step in [0.02, 0.01]:
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        points[vi] = new_p
                        
                        rad = compute_covering_radius_exact(points)
                        if rad < best_rad - 1e-12:
                            best_rad = rad
                            best_points = points.copy()
                        else:
                            points[vi] = old_p
                            break

    # Phase 4: Final polish with L-BFGS-B using fine grid
    if best_points is not None and time.time() < deadline - 5.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 3.0
        if time_left > 3.0:
            final, _ = optimize_with_lbfgsb(final, max_time=min(time_left, 12.0), grid_pts=grid_pts_fine)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final

    # Phase 5: One more targeted refinement on worst facets
    if best_points is not None and time.time() < deadline - 3.0:
        points = best_points.copy()
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
            
            radii = np.zeros(len(faces))
            valid = norms_c > 1e-10
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms_c[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            top_k = min(3, len(radii))
            worst_indices = np.argsort(radii)[-top_k:]
            
            for wi in worst_indices:
                if time.time() > deadline - 2.0:
                    break
                face = faces[wi]
                for vi in face:
                    old_p = points[vi].copy()
                    face_pts = points[face]
                    face_center = np.mean(face_pts, axis=0)
                    face_center /= np.linalg.norm(face_center)
                    
                    for step in [0.01, 0.005, 0.002]:
                        new_p = old_p + step * (face_center - old_p)
                        new_p /= np.linalg.norm(new_p)
                        points[vi] = new_p
                        rad = compute_covering_radius_exact(points)
                        if rad < best_rad - 1e-12:
                            best_rad = rad
                            best_points = points.copy()
                        else:
                            points[vi] = old_p
                            break
        except Exception:
            pass

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END

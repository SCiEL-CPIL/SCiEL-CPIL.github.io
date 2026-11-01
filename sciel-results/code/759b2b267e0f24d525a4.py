"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull

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

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces (vectorized)."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices  # shape (F, 3)
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
            # Fallback: grid-based estimate
            lat = np.linspace(-np.pi/2, np.pi/2, 50)
            lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
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

    def get_worst_facet(pts):
        """Return index of the point with the worst (largest) facet circumradius, or -1."""
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
            
            worst_face_idx = np.argmax(radii)
            return int(faces[worst_face_idx, 0])
        except Exception:
            return -1

    def soft_max_covering_radius_and_grad(pts, grid_pts, beta=20.0):
        """Compute soft-max covering radius and its gradient w.r.t. pts.
        
        Uses log-sum-exp trick for numerical stability.
        Returns (radius, grads) where grads has shape (N, 3).
        """
        # Compute angular distances from each grid point to each config point
        dot = np.dot(grid_pts, pts.T)  # (G, N)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # (G, N)
        
        # Soft-max: approx max of min distances
        # For each grid point, min distance to any config point
        min_dists = np.min(ang_dists, axis=1)  # (G,)
        
        # Soft-max over grid points
        # s = (1/beta) * log(sum(exp(-beta * min_dists)))
        # This approximates max(min_dists) as beta -> inf
        neg_beta_dists = -beta * min_dists
        max_neg = np.max(neg_beta_dists)
        log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_dists - max_neg)))
        radius = -log_sum / beta
        
        # Gradient: d(radius)/d(pts[i]) = (1/beta) * d(log_sum)/d(pts[i])
        # d(log_sum)/d(min_dists[g]) = exp(-beta * min_dists[g]) / sum(exp(-beta * min_dists)))
        weights = np.exp(neg_beta_dists - max_neg)
        weights /= np.sum(weights)
        
        # For each grid point g, find which config point i_g gives the min distance
        min_indices = np.argmin(ang_dists, axis=1)  # (G,)
        
        # Vectorized gradient computation:
        # d(arccos(dot))/d(pts[i]) = -grid_pts[g] / sqrt(1 - dot^2)
        # grads[i] = -sum_g (weights[g] / denom[g]) * grid_pts[g]  for g where min_indices[g]==i
        
        dot_gi = dot[np.arange(len(grid_pts)), min_indices]  # (G,)
        denom = np.sqrt(np.maximum(1.0 - dot_gi * dot_gi, 1e-12))  # (G,)
        coeff = weights / denom  # (G,)
        
        # Accumulate: for each point i, sum over all g where min_indices[g]==i
        # Use np.add.at for unbuffered accumulation
        grads = np.zeros_like(pts)
        np.add.at(grads, min_indices, -coeff[:, np.newaxis] * grid_pts)
        
        grads /= beta
        return radius, grads

    def repulsion_force(points, exponent=1.5):
        """Compute repulsion forces between all point pairs."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        return forces

    def gradient_descent(points, alpha_init, decay, max_iter, exponent=1.5):
        """Optimize points via gradient descent on the repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            forces = repulsion_force(points, exponent)
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    best_points = None
    best_rad = np.inf

    # Precompute a grid for soft-max surrogate
    lat = np.linspace(-np.pi/2, np.pi/2, 40)
    lon = np.linspace(0, 2*np.pi, 80, endpoint=False)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    z_grid = np.sin(LA)
    x_grid = np.cos(LA) * np.cos(LO)
    y_grid = np.cos(LA) * np.sin(LO)
    grid_pts = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    def gradient_descent_smooth(points, alpha_init, decay, max_iter, beta=20.0):
        """Optimize using smooth surrogate with gradient descent."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            radius, grads = soft_max_covering_radius_and_grad(points, grid_pts, beta)
            points = points + alpha * grads
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    # Phase 1: Multiple random starts with repulsion-based optimization
    n_restarts = 4
    exponents = [1.0, 1.5, 2.0, 3.0]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            # Random start
            rng = np.random.RandomState(42 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Perturbed fibonacci
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100 + restart)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Use best found so far with perturbation
            if best_points is not None:
                rng = np.random.RandomState(200 + restart)
                points = best_points + 0.15 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart % len(exponents)]
        points = gradient_descent(points, 0.05, 0.995, 200, exp)
        rad = compute_covering_radius(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Smooth surrogate optimization with multi-scale annealing
    if best_points is not None and time.time() < deadline - 30.0:
        # Multi-scale: start with low beta (smooth) then increase (sharper)
        beta_schedule = [8.0, 15.0, 25.0, 40.0, 60.0]
        alpha_schedule = [0.03, 0.02, 0.015, 0.01, 0.008]
        decay_schedule = [0.997, 0.996, 0.995, 0.994, 0.993]
        iter_schedule = [100, 100, 80, 80, 60]
        
        for idx, (b, a0, dcy, n_it) in enumerate(zip(beta_schedule, alpha_schedule, decay_schedule, iter_schedule)):
            if time.time() > deadline - 15.0:
                break
            candidate = best_points.copy()
            candidate = gradient_descent_smooth(candidate, a0, dcy, n_it, beta=b)
            rad = compute_covering_radius(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Targeted worst-facet refinement (efficient version)
    if best_points is not None and time.time() < deadline - 15.0:
        points = best_points.copy()
        
        # Build hull once and cache
        try:
            hull = ConvexHull(points)
            faces = hull.simplices
        except Exception:
            faces = None
        
        max_iter = 60
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Recompute hull every 10 iterations or at start
            if faces is None or it % 10 == 0:
                try:
                    hull = ConvexHull(points)
                    faces = hull.simplices
                except Exception:
                    break
            
            a = points[faces[:, 0]]
            b = points[faces[:, 1]]
            c = points[faces[:, 2]]
            
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
            
            worst_face_local = np.argmax(radii)
            worst_face = faces[worst_face_local]
            worst_radius = radii[worst_face_local]
            
            # Try each of the 3 vertices of the worst facet
            improved = False
            for vi in range(3):
                worst_idx = worst_face[vi]
                
                # Compute circumcenter direction for this facet
                fa, fb, fc = points[worst_face[0]], points[worst_face[1]], points[worst_face[2]]
                ab2 = np.cross(fa, fb)
                bc2 = np.cross(fb, fc)
                ca2 = np.cross(fc, fa)
                c_dir2 = ab2 + bc2 + ca2
                norm_c2 = np.linalg.norm(c_dir2)
                if norm_c2 < 1e-10:
                    continue
                c_dir2 /= norm_c2
                
                # Try multiple step sizes
                for step in [0.02, 0.01, 0.005]:
                    new_pt = points[worst_idx] + step * (c_dir2 - points[worst_idx])
                    new_pt /= np.linalg.norm(new_pt)
                    
                    old_pt = points[worst_idx].copy()
                    points[worst_idx] = new_pt
                    
                    # Quick check: only rebuild hull if step seems promising
                    # Use a cheaper local check first
                    new_rad = compute_covering_radius(points)
                    
                    if new_rad < best_rad - 1e-10:
                        best_rad = new_rad
                        best_points = points.copy()
                        improved = True
                        break
                    else:
                        points[worst_idx] = old_pt
                
                if improved:
                    break
            
            if not improved:
                # No improvement from worst facet vertices; try a small random perturbation
                if it % 20 == 10:
                    rng = np.random.RandomState(it * 7 + 13)
                    trial = best_points.copy()
                    trial += 0.05 * rng.randn(N, 3)
                    trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                    trial_rad = compute_covering_radius(trial)
                    if trial_rad < best_rad:
                        best_rad = trial_rad
                        best_points = trial.copy()
                        points = trial.copy()
                        faces = None  # force hull rebuild

    # Final polish with smooth surrogate (two passes with different betas)
    if best_points is not None and time.time() < deadline - 5.0:
        final = gradient_descent_smooth(best_points.copy(), 0.008, 0.995, 60, beta=30.0)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final.copy()
        
        if best_points is not None and time.time() < deadline - 2.0:
            final2 = gradient_descent_smooth(best_points.copy(), 0.004, 0.99, 40, beta=50.0)
            rad_final2 = compute_covering_radius(final2)
            if rad_final2 < best_rad:
                best_rad = rad_final2
                best_points = final2.copy()

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END

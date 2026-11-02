"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

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

    def facet_gradient_step(pts):
        """
        CVT-style step: Move each point towards the circumcenter of its Voronoi cell (facet).
        This directly optimizes the covering radius by reducing the largest cells.
        """
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            
            # For each point, find the faces that contain it
            # and compute the weighted average of their circumcenters.
            # The weight is the area of the face (or just 1 for simplicity).
            
            # Build a mapping from point index to list of face indices
            point_to_faces = {i: [] for i in range(N)}
            for idx, face in enumerate(faces):
                for p_idx in face:
                    point_to_faces[p_idx].append(idx)
            
            # Compute circumcenters for all faces
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            circumcenters = np.zeros((len(faces), 3))
            valid = norms > 1e-10
            
            if np.any(valid):
                circumcenters[valid] = c_dir[valid] / norms[valid, np.newaxis]
            
            # For each point, compute the average of the circumcenters of adjacent faces
            # This is the centroid of the Voronoi cell
            new_pts = np.zeros_like(pts)
            for i in range(N):
                faces_for_i = point_to_faces[i]
                if len(faces_for_i) > 0:
                    # Average the circumcenters
                    centers = circumcenters[faces_for_i]
                    # Weight by the area of the face? For now, simple average.
                    # A better weight would be the solid angle or area.
                    # Let's use the norm of the cross product as a proxy for area.
                    areas = np.linalg.norm(np.cross(a[faces_for_i], b[faces_for_i]), axis=1)
                    if np.sum(areas) > 1e-10:
                        new_pts[i] = np.sum(centers * areas[:, np.newaxis], axis=0) / np.sum(areas)
                    else:
                        new_pts[i] = np.mean(centers, axis=0)
                else:
                    new_pts[i] = pts[i]
            
            # Normalize to sphere
            norms_new = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms_new[norms_new < 1e-10] = 1e-10
            new_pts = new_pts / norms_new
            
            return new_pts
        except Exception:
            return pts

    def smooth_objective_and_grad(pts, grid_pts, T_min, T_max):
        """
        Compute the smooth approximation of the covering radius and its gradient.
        pts: (N, 3)
        grid_pts: (M, 3)
        Returns: (value, gradient (N, 3))
        """
        norms_p = np.linalg.norm(pts, axis=1, keepdims=True)
        norms_p[norms_p < 1e-10] = 1e-10
        pts_norm = pts / norms_p
        
        M = len(grid_pts)
        
        dot = np.dot(grid_pts, pts_norm.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)
        
        one_minus_u_sq = 1.0 - dot * dot
        one_minus_u_sq = np.clip(one_minus_u_sq, 1e-12, None)
        sqrt_one_minus_u_sq = np.sqrt(one_minus_u_sq)
        
        g_exp = grid_pts[:, np.newaxis, :]
        p_exp = pts_norm[np.newaxis, :, :]
        u_exp = dot[:, :, np.newaxis]
        denom_exp = sqrt_one_minus_u_sq[:, :, np.newaxis]
        
        grad_d = -(g_exp - u_exp * p_exp) / denom_exp
        
        exp_neg_T_min_d = np.exp(-T_min * ang_dists)
        sum_exp_neg = np.sum(exp_neg_T_min_d, axis=1)
        sum_exp_neg = np.clip(sum_exp_neg, 1e-15, None)
        
        soft_min_dists = -1.0 / T_min * np.log(sum_exp_neg)
        
        w_jk = exp_neg_T_min_d / sum_exp_neg[:, np.newaxis]
        
        exp_T_max_s = np.exp(T_max * soft_min_dists)
        sum_exp_T_max = np.sum(exp_T_max_s)
        sum_exp_T_max = np.clip(sum_exp_T_max, 1e-15, None)
        
        value = 1.0 / T_max * np.log(sum_exp_T_max)
        
        v_j = exp_T_max_s / sum_exp_T_max
        
        coeff_jk = v_j[:, np.newaxis] * w_jk
        
        grad_L = np.sum(coeff_jk[:, :, np.newaxis] * grad_d, axis=0)
        
        return float(value), grad_L

    def gradient_descent_smooth(pts, grid_pts, max_iter, time_limit):
        """
        Optimize points using gradient descent on the smooth objective.
        """
        T_min = 20.0
        T_max = 20.0
        alpha = 0.05
        decay = 0.995
        
        best_val = np.inf
        best_pts = pts.copy()
        
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            
            val, grad = smooth_objective_and_grad(pts, grid_pts, T_min, T_max)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            pts = pts - alpha * grad
            
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            if val < best_val:
                best_val = val
                best_pts = pts.copy()
            
            if it % 10 == 0:
                T_min *= 1.05
                T_max *= 1.05
                alpha *= decay
        return best_pts, best_val

    # Precompute a grid for the smooth objective
    grid_pts = fibonacci_spiral(3000)
    
    best_points = None
    best_rad = np.inf
    
    n_restarts = 4
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            points = best_points + 0.15 * np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        
        # Phase 1: Repulsion-based initialization to spread points out
        # This helps avoid local minima where points cluster
        if time.time() < deadline - 20.0:
            points = gradient_descent_repulsion(points, exponent=2.0, alpha_init=0.05, decay=0.99, max_iter=100)
        
        # Phase 2: CVT-style optimization
        # Move points towards the centers of their Voronoi cells
        if time.time() < deadline - 10.0:
            for _ in range(20):
                if time.time() > deadline - 10.0:
                    break
                new_points = facet_gradient_step(points)
                # Blend with old points to ensure stability
                points = 0.5 * points + 0.5 * new_points
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                points = points / norms
        
        # Phase 3: Gradient descent on smooth objective
        time_alloc = 100.0 if restart == 0 else 30.0
        max_iter = 500
        points_opt, val_opt = gradient_descent_smooth(points, grid_pts, max_iter, start_time + time_alloc)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points_opt)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points_opt.copy()
            
        # Polish step
        if time.time() < deadline - 10.0:
            T_min_polish = 50.0
            T_max_polish = 50.0
            alpha_polish = 0.005
            
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                val, grad = smooth_objective_and_grad(best_points, grid_pts, T_min_polish, T_max_polish)
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-10:
                    grad = grad / grad_norm
                best_points = best_points - alpha_polish * grad
                norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                best_points = best_points / norms
                alpha_polish *= 0.95
                
            rad_polished = compute_covering_radius_exact(best_points)
            if rad_polished < best_rad:
                best_rad = rad_polished
                best_points = best_points.copy()

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)
    
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END

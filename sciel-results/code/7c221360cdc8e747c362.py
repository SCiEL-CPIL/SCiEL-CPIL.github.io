"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_exact_covering_radius(pts):
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

    def create_grid(n_lat=60, n_lon=120):
        """Create a uniform grid of points on the sphere."""
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        return grid_pts

    # Pre-compute grid for the soft objective
    # 60x120 = 7200 points. Fast enough for dot products.
    grid_pts = create_grid(80, 160) # 12800 points
    M = grid_pts.shape[0]

    def soft_objective_and_grad(x_flat, beta=50.0):
        """
        Compute the soft-max covering radius and its gradient.
        x_flat: (N*3,) array of points.
        Returns: (value, grad_flat)
        """
        pts = x_flat.reshape(N, 3)
        # Renormalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        # 1. Dot products: (M, N)
        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        
        # 2. Max dot and index
        max_dot = np.max(dot, axis=1) # (M,)
        max_idx = np.argmax(dot, axis=1) # (M,)
        
        # 3. Angular distances
        # d = arccos(max_dot)
        # We need to be careful with arccos gradient near +-1
        # For soft max, we work with exp(beta * d)
        # Let's compute d first
        d = np.arccos(max_dot) # (M,)
        
        # 4. Soft max value
        # To avoid overflow, shift by max d
        d_max = np.max(d)
        exp_vals = np.exp(beta * (d - d_max))
        sum_exp = np.sum(exp_vals)
        val = d_max + np.log(sum_exp) / beta
        
        # 5. Gradient
        # weights w_j = exp(beta * (d_j - d_max)) / sum_exp
        w = exp_vals / sum_exp # (M,)
        
        # Gradient of d_j w.r.t p_i
        # Only non-zero if i == max_idx[j]
        # d(d_j)/d(p_i) = -g_j / sqrt(1 - dot^2)
        # We need the tangent component.
        
        # Get the nearest points for each grid point
        nearest_pts = pts[max_idx] # (M, 3)
        
        # Dot product between grid points and their nearest config point
        # This is just max_dot, but let's recompute to be safe with clipping
        # dot_nearest = np.sum(grid_pts * nearest_pts, axis=1)
        dot_nearest = max_dot
        
        # Denominator sqrt(1 - dot^2)
        # Clip to avoid division by zero
        denom = np.sqrt(np.maximum(1.0 - dot_nearest**2, 1e-12))
        
        # Gradient vector in ambient space: -grid_pts / denom
        grad_ambient = -grid_pts / denom[:, np.newaxis] # (M, 3)
        
        # Project to tangent space of nearest_pts
        # g_tan = g_amb - (g_amb . p) p
        # g_amb . p = (-grid_pts / denom) . nearest_pts = -dot_nearest / denom
        # So g_tan = -grid_pts / denom + (dot_nearest / denom) * nearest_pts
        # g_tan = (nearest_pts * dot_nearest - grid_pts) / denom
        dot_nearest_expanded = dot_nearest[:, np.newaxis]
        grad_tangent = (nearest_pts * dot_nearest_expanded - grid_pts) / denom[:, np.newaxis]
        
        # Scale by weights and beta
        # grad_obj = sum_j w_j * beta * grad_tangent_j
        # We need to scatter add grad_tangent_j to grad_pts[max_idx[j]]
        
        weighted_grads = grad_tangent * (w * beta)[:, np.newaxis] # (M, 3)
        
        grad_pts = np.zeros_like(pts)
        # Scatter add
        np.add.at(grad_pts, max_idx, weighted_grads)
        
        # Flatten gradient
        grad_flat = grad_pts.ravel()
        
        return val, grad_flat

    def repulsion_step(points, alpha):
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** 1.5)
        np.fill_diagonal(w, 0.0)
        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        points = points + alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points

    best_points = None
    best_rad = np.inf
    
    n_restarts = 4
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        else:
            points = best_points + 0.1 * np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        
        # Phase 1: Repulsion to spread points
        alpha = 0.05
        for it in range(50):
            points = repulsion_step(points, alpha)
            alpha *= 0.98
            
        # Phase 2: L-BFGS Polish with Soft Objective
        # We optimize the flattened vector. The objective renormalizes internally.
        # However, L-BFGS-B might drift off the sphere if not careful.
        # Our objective renormalizes, so it's okay.
        
        x0 = points.ravel()
        
        # Use a moderate beta. Higher beta -> closer to hard max, but noisier gradient.
        # 50-100 is usually good.
        
        def obj_and_grad(x):
            val, grad = soft_objective_and_grad(x, beta=80.0)
            return val, grad

        try:
            res = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000})
            pts_opt = res.x.reshape(N, 3)
            # Final renormalization
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            pts_opt = pts_opt / norms
        except Exception:
            pts_opt = points

        # Phase 3: Exact Evaluation and Local Fine-tuning
        # The soft objective might be slightly off from the exact hull radius.
        # We can do a few steps of "exact" local optimization if we have time.
        # For now, we just evaluate and keep the best.
        
        rad = compute_exact_covering_radius(pts_opt)
        if rad < best_rad:
            best_rad = rad
            best_points = pts_opt.copy()
            
        # Optional: If we have time, do a second pass with higher beta or different start
        if time.time() < deadline - 10.0 and restart > 0:
             # Try a small perturbation of the best found so far
             points2 = best_points + 0.01 * np.random.randn(N, 3)
             norms = np.linalg.norm(points2, axis=1, keepdims=True)
             points2 = points2 / norms
             
             # Quick L-BFGS
             x0_2 = points2.ravel()
             try:
                 res2 = minimize(obj_and_grad, x0_2, method='L-BFGS-B', jac=True,
                                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500})
                 pts_opt2 = res2.x.reshape(N, 3)
                 norms = np.linalg.norm(pts_opt2, axis=1, keepdims=True)
                 norms[norms < 1e-10] = 1.0
                 pts_opt2 = pts_opt2 / norms
                 
                 rad2 = compute_exact_covering_radius(pts_opt2)
                 if rad2 < best_rad:
                     best_rad = rad2
                     best_points = pts_opt2.copy()
             except Exception:
                 pass

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)
    
    return best_points

# EVOLVE-BLOCK-END

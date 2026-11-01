"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

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

    def smooth_covering_radius(points, beta=50.0):
        """
        Smooth surrogate for the covering radius using log-sum-exp over a fine grid.
        Returns (value, gradient) where value approximates the max angular distance.
        """
        # Use a moderately fine grid for the surrogate
        lat = np.linspace(-np.pi/2, np.pi/2, 40)
        lon = np.linspace(0, 2*np.pi, 80, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        M = grid_pts.shape[0]
        
        # Compute angular distances: arccos(max dot product with any point)
        dot = np.dot(grid_pts, points.T)  # (M, N)
        dot = np.clip(dot, -1.0, 1.0)
        min_ang = np.arccos(np.max(dot, axis=1))  # (M,)
        
        # Soft-max via log-sum-exp
        m = np.max(min_ang)
        exp_vals = np.exp(beta * (min_ang - m))
        soft_max = m + np.log(np.sum(exp_vals)) / beta
        
        # Gradient: d(soft_max)/d(points_i)
        # soft_max = m + (1/beta) * log(sum_j exp(beta*(d_j - m)))
        # where d_j = arccos(max_i dot(grid_j, points_i))
        # This is complex; approximate gradient using the weights
        weights = exp_vals / np.sum(exp_vals)  # (M,)
        
        # For each grid point, the nearest point index
        nearest_idx = np.argmax(dot, axis=1)  # (M,)
        
        # Gradient accumulation
        grad = np.zeros_like(points)
        for j in range(M):
            w_j = weights[j]
            if w_j < 1e-8:
                continue
            i = nearest_idx[j]
            g_j = grid_pts[j]
            p_i = points[i]
            d_j = min_ang[j]
            # d(d_j)/d(p_i) = -g_j / sqrt(1 - dot^2)
            dot_val = np.dot(g_j, p_i)
            denom = np.sqrt(max(1.0 - dot_val**2, 1e-12))
            grad[i] += w_j * beta * (-g_j / denom)
        
        return soft_max, grad

    def facet_shrink_step(points, alpha_shrink):
        """
        Identify the facet with the largest circumradius and move its vertices
        towards the circumcenter to reduce the radius.
        """
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
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if not np.any(valid):
                return points
            
            c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
            cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
            r = np.arccos(cos_r)
            r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
            radii[valid] = r
            
            max_idx = np.argmax(radii)
            if radii[max_idx] < 1e-6:
                return points
                
            # Get the face index in the original faces array
            valid_indices = np.where(valid)[0]
            face_idx = valid_indices[max_idx]
            
            v1_idx = faces[face_idx, 0]
            v2_idx = faces[face_idx, 1]
            v3_idx = faces[face_idx, 2]
            
            # Circumcenter direction
            center = c_dir[face_idx]
            center = center / np.linalg.norm(center)
            
            # Move vertices towards center
            step = alpha_shrink
            
            points[v1_idx] = points[v1_idx] + step * (center - points[v1_idx])
            points[v2_idx] = points[v2_idx] + step * (center - points[v2_idx])
            points[v3_idx] = points[v3_idx] + step * (center - points[v3_idx])
            
            # Renormalize
            points[v1_idx] /= np.linalg.norm(points[v1_idx])
            points[v2_idx] /= np.linalg.norm(points[v2_idx])
            points[v3_idx] /= np.linalg.norm(points[v3_idx])
            
        except Exception:
            pass
            
        return points

    def polish_with_l_bfgs(points, max_iter=200, tol=1e-8):
        """
        Use L-BFGS-B with a smooth surrogate to polish the configuration.
        Flattens the (N,3) array, optimizes, then renormalizes each point.
        """
        from scipy.optimize import minimize
        
        def objective_and_grad(x):
            pts = x.reshape(N, 3)
            # Renormalize to keep on sphere during optimization
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms
            val, grad = smooth_covering_radius(pts, beta=80.0)
            # Project gradient back to tangent space
            grad_flat = np.zeros_like(x)
            for i in range(N):
                p = pts[i]
                g = grad[i]
                # Tangent component
                g_tan = g - np.dot(g, p) * p
                grad_flat[i] = g_tan
            return val, grad_flat
        
        x0 = points.ravel()
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                          jac=True, options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxfun': 5000})
            pts_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            pts_opt = pts_opt / norms
            return pts_opt
        except Exception:
            return points

    best_points = None
    best_rad = np.inf
    
    n_restarts = 3
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        else:
            points = best_points + 0.1 * np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        
        alpha_rep = 0.05
        alpha_shrink = 0.01
        decay = 0.995
        max_iter = 200
        
        for it in range(max_iter):
            if time.time() > deadline - 35.0:
                break
            
            # Alternate between global repulsion and local facet shrinking
            if it % 2 == 0:
                points = repulsion_step(points, alpha_rep)
            else:
                points = facet_shrink_step(points, alpha_shrink)
            
            alpha_rep *= decay
            alpha_shrink *= decay
            
            if it % 10 == 0:
                rad = compute_covering_radius(points)
                if rad < best_rad:
                    best_rad = rad
                    best_points = points.copy()
            
            if alpha_rep < 1e-6:
                break
        
        # Final polish with L-BFGS-B on the smooth surrogate
        if time.time() < deadline - 25.0:
            points_polished = polish_with_l_bfgs(points, max_iter=150, tol=1e-9)
            rad_polished = compute_covering_radius(points_polished)
            if rad_polished < best_rad:
                best_rad = rad_polished
                best_points = points_polished.copy()
    
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)
    
    return best_points

# EVOLVE-BLOCK-END

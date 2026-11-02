"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    
    def smooth_max_obj(pts, beta):
        """
        Smooth surrogate for covering radius using log-sum-exp on facet radii.
        Returns (objective_value, gradient_flat) where gradient is w.r.t. flat params.
        """
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
            circumcenters = np.zeros((len(faces), 3))
            valid = norms > 1e-10
            
            if np.any(valid):
                cc = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
                circumcenters[valid] = cc
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                circumcenters[~valid] = (a_deg + b_deg + c_deg) / 3.0
                norms_cc = np.linalg.norm(circumcenters[~valid], axis=1, keepdims=True)
                circumcenters[~valid] /= norms_cc
            
            # Log-sum-exp smooth max
            max_r = np.max(radii)
            exp_r = np.exp(beta * (radii - max_r))
            logsumexp = max_r + np.log(np.sum(exp_r)) / beta
            weights = exp_r / np.sum(exp_r)
            
            # Gradient w.r.t. each point: sum over faces containing that point of weight * d(r_i)/d(p)
            # Approximate: gradient direction is toward circumcenter, projected to tangent
            n_points = len(pts)
            grad_3d = np.zeros((n_points, 3))
            
            for fi in range(len(faces)):
                w = weights[fi]
                if w < 1e-10:
                    continue
                cc = circumcenters[fi]
                for vi in faces[fi]:
                    p = pts[vi]
                    # Direction from p to cc, projected to tangent plane at p
                    force = cc - p
                    radial = np.dot(force, p) * p
                    force = force - radial
                    grad_3d[vi] += w * force
            
            # The objective increases when points move AWAY from circumcenters of large facets
            # So gradient of objective w.r.t. p is -grad_3d (we want to minimize, move toward cc)
            grad_flat = -grad_3d.ravel()
            
            return logsumexp, grad_flat
        except Exception:
            # Fallback: use grid-based smooth objective
            lat = np.linspace(-np.pi/2, np.pi/2, 30)
            lon = np.linspace(0, 2*np.pi, 60, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            max_d = np.max(min_dists)
            exp_d = np.exp(beta * (min_dists - max_d))
            logsumexp = max_d + np.log(np.sum(exp_d)) / beta
            return logsumexp, np.zeros(3 * len(pts))
    
    def smooth_obj_and_grad(flat_params, beta):
        """Wrapper for scipy minimize: flat params -> (obj, grad)."""
        n = len(flat_params) // 3
        pts = flat_params.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        obj, grad = smooth_max_obj(pts, beta)
        return obj, grad
    
    def compute_face_radii_and_circumcenters(pts):
        """Return arrays of face circumradii and circumcenters for gradient-aware optimization."""
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
            circumcenters = np.zeros((len(faces), 3))
            valid = norms > 1e-10
            
            if np.any(valid):
                cc = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
                circumcenters[valid] = cc
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                # Approximate circumcenter for degenerate faces
                circumcenters[~valid] = (a_deg + b_deg + c_deg) / 3.0
                norms_cc = np.linalg.norm(circumcenters[~valid], axis=1, keepdims=True)
                circumcenters[~valid] /= norms_cc
            
            return radii, circumcenters, faces
        except Exception:
            return None, None, None
    
    def facet_gradient_step(points, alpha, radii, circumcenters, faces, beta=20.0):
        """
        Move each vertex toward the circumcenters of facets it belongs to,
        weighted by a soft-max over facet radii to smooth the max.
        """
        n_points = len(points)
        n_faces = len(faces)
        forces = np.zeros((n_points, 3))
        
        if n_faces == 0:
            return points
        
        # Soft-max weights: w_i = exp(beta * r_i) / sum(exp(beta * r_j))
        # This makes facets with larger radii get more weight
        max_r = np.max(radii)
        weights = np.exp(beta * (radii - max_r))
        weights /= np.sum(weights)
        
        # For each face, distribute the force to its three vertices
        for fi in range(n_faces):
            w = weights[fi]
            if w < 1e-8:
                continue
            cc = circumcenters[fi]
            for vi in faces[fi]:
                # Force direction: move point toward circumcenter
                # Project to tangent plane
                p = points[vi]
                force = cc - p
                # Remove radial component
                radial = np.dot(force, p) * p
                force = force - radial
                forces[vi] += w * force
        
        points = points + alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points
    
    def repulsion_step(points, alpha):
        """Standard repulsion step to prevent clustering."""
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
    
    def polish_step(points, alpha, n_iter=50):
        """High-precision polish with very small steps."""
        for it in range(n_iter):
            radii, ccs, faces = compute_face_radii_and_circumcenters(points)
            if radii is not None:
                points = facet_gradient_step(points, alpha, radii, ccs, faces, beta=30.0)
            # Small repulsion to maintain separation
            points = repulsion_step(points, alpha * 0.1)
        return points
    
    def scipy_polish(points, time_budget=10.0):
        """
        Use scipy L-BFGS-B to polish with the smooth log-sum-exp objective.
        """
        start = time.time()
        n = len(points)
        
        def objective_and_grad(flat_params):
            elapsed = time.time() - start
            if elapsed > time_budget:
                return np.inf, np.zeros(3 * n)
            pts = flat_params.reshape(n, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts = pts / norms
            obj, grad = smooth_max_obj(pts, beta=50.0)
            return obj, grad
        
        x0 = points.ravel()
        try:
            result = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 50, 'gtol': 1e-6, 'maxfun': 100}
            )
            if result.success or result.fun < np.inf:
                pts_new = result.x.reshape(n, 3)
                norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
                norms[norms < 1e-15] = 1.0
                pts_new = pts_new / norms
                return pts_new
        except Exception:
            pass
        return points
    
    best_points = None
    best_rad = np.inf
    
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            points = best_points + 0.08 * np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        
        alpha = 0.03
        decay = 0.995
        max_iter = 400
        stall_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 15.0:
                break
            
            # Compute facet data
            radii, ccs, faces = compute_face_radii_and_circumcenters(points)
            
            if radii is not None:
                # Adaptive beta: start smooth, sharpen later
                beta = 10.0 + 30.0 * (it / max_iter)
                # Facet gradient step
                points = facet_gradient_step(points, alpha, radii, ccs, faces, beta=beta)
                # Small repulsion to prevent collapse
                points = repulsion_step(points, alpha * 0.05)
            
            alpha *= decay
            
            if it % 5 == 0:
                rad = compute_covering_radius(points)
                if rad < best_rad:
                    best_rad = rad
                    best_points = points.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 25:
                        alpha *= 0.9
                        stall_count = 0
            
            if alpha < 1e-8:
                break
        
        # Polish phase with facet gradient
        if time.time() < deadline - 15.0 and best_points is not None:
            polished = polish_step(best_points.copy(), 0.003, 50)
            rad_polished = compute_covering_radius(polished)
            if rad_polished < best_rad:
                best_rad = rad_polished
                best_points = polished
    
    # Final global polish: facet gradient + scipy L-BFGS-B
    if time.time() < deadline - 10.0 and best_points is not None:
        final = polish_step(best_points.copy(), 0.002, 60)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final
        
        # SciPy polish with smooth objective
        if time.time() < deadline - 5.0 and best_points is not None:
            time_left = deadline - 5.0 - time.time()
            if time_left > 3.0:
                scipy_polished = scipy_polish(best_points.copy(), time_budget=min(time_left - 1.0, 15.0))
                rad_scipy = compute_covering_radius(scipy_polished)
                if rad_scipy < best_rad:
                    best_rad = rad_scipy
                    best_points = scipy_polished
        
        # Final tiny facet polish
        if time.time() < deadline - 3.0 and best_points is not None:
            final2 = polish_step(best_points.copy(), 0.001, 50)
            rad_final2 = compute_covering_radius(final2)
            if rad_final2 < best_rad:
                best_rad = rad_final2
                best_points = final2
    
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END

"""Improved program for covering the sphere with 130 equal caps."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _facet_circumradius_batch(pts, simplices):
    """Vectorized computation of facet circumradii for all simplices at once."""
    n_facets = len(simplices)
    if n_facets == 0:
        return np.array([])
    
    v1 = pts[simplices[:, 0]]
    v2 = pts[simplices[:, 1]]
    v3 = pts[simplices[:, 2]]
    
    e1 = v2 - v1
    e2 = v3 - v1
    normals = np.cross(e1, e2)
    n_norms = np.linalg.norm(normals, axis=1)
    
    valid = n_norms > 1e-15
    angles = np.zeros(n_facets)
    
    if np.any(valid):
        d_vals = np.einsum('ij,ij->i', normals[valid], v1[valid])
        dists = np.abs(d_vals) / n_norms[valid]
        dists = np.clip(dists, -1.0, 1.0)
        angles[valid] = np.arccos(dists)
    
    return angles

def _facet_circumcenters(pts, simplices):
    """Vectorized computation of facet circumcenters (unit vectors)."""
    n_facets = len(simplices)
    if n_facets == 0:
        return np.zeros((0, 3))
    
    v1 = pts[simplices[:, 0]]
    v2 = pts[simplices[:, 1]]
    v3 = pts[simplices[:, 2]]
    
    e1 = v2 - v1
    e2 = v3 - v1
    normals = np.cross(e1, e2)
    n_norms = np.linalg.norm(normals, axis=1)
    
    valid = n_norms > 1e-15
    centers = np.zeros((n_facets, 3))
    
    if np.any(valid):
        d_vals = np.einsum('ij,ij->i', normals[valid], v1[valid])
        # Foot of perpendicular from origin to plane
        centers_valid = (d_vals[:, None] / (n_norms[valid] ** 2)[:, None]) * normals[valid]
        c_norms = np.linalg.norm(centers_valid, axis=1)
        valid2 = c_norms > 1e-15
        centers_valid[valid2] /= c_norms[valid2, None]
        
        # Build index mapping
        valid_indices = np.where(valid)[0]
        centers[valid_indices[valid2]] = centers_valid[valid2]
    
    return centers

def _covering_radius(pts):
    """Compute the exact covering radius using convex hull facets."""
    if len(pts) < 4:
        return _covering_radius_grid(pts, 360, 720)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_grid(pts, 360, 720)
    
    angles = _facet_circumradius_batch(pts, hull.simplices)
    if len(angles) == 0:
        return 0.0
    return np.degrees(np.max(angles))

def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_grid(pts, n_theta=180, n_phi=360):
    """Compute covering radius in degrees using a grid approximation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)
    max_radius_rad = np.arccos(np.min(min_dots))
    return np.degrees(max_radius_rad)

def _lloyd_iteration(pts, n_theta=360, n_phi=720):
    """One Lloyd iteration: move each point to the centroid of its Voronoi cell."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    dots = grid @ pts.T
    assignments = dots.argmax(axis=1)
    
    new_pts = pts.copy()
    for i in range(len(pts)):
        mask = assignments == i
        if np.sum(mask) > 0:
            centroid = grid[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_pts[i] = centroid / norm
    
    return new_pts

def _refine_lloyd(pts, start_time, max_time=120.0, max_iters=30):
    """Lloyd-based refinement with annealing."""
    current = pts.copy()
    best = pts.copy()
    best_radius = _covering_radius_grid(current, 180, 360)
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 10.0:
            break
        
        new_pts = _lloyd_iteration(current)
        
        # Check for coincident points and nudge apart
        for i in range(N):
            for j in range(i + 1, N):
                dot = np.dot(new_pts[i], new_pts[j])
                if dot > 0.999999:
                    diff = new_pts[i] - new_pts[j]
                    norm = np.linalg.norm(diff)
                    if norm < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                        norm = 1.0
                    diff /= norm
                    new_pts[i] = new_pts[i] + 0.01 * diff
                    new_pts[j] = new_pts[j] - 0.01 * diff
                    new_pts[i] /= np.linalg.norm(new_pts[i])
                    new_pts[j] /= np.linalg.norm(new_pts[j])
        
        new_radius = _covering_radius_grid(new_pts, 180, 360)
        if new_radius < best_radius:
            current = new_pts
            best = new_pts.copy()
            best_radius = new_radius
        else:
            # Try blending
            alpha = 0.5
            blended = (1 - alpha) * current + alpha * new_pts
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            blended = blended / norms
            blended_radius = _covering_radius_grid(blended, 180, 360)
            if blended_radius < best_radius:
                current = blended
                best = blended.copy()
                best_radius = blended_radius
            else:
                break
    
    return best, best_radius

def _gradient_descent_facets(pts, start_time, max_time=120.0, max_iters=500):
    """
    Gradient descent targeting the worst facets using vectorized computations.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    step_size = 0.02
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 15.0:
            break
        
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            break
        
        simplices = hull.simplices
        angles = _facet_circumradius_batch(best_pts, simplices)
        
        if len(angles) == 0:
            break
        
        max_angle = np.max(angles)
        if max_angle < np.radians(11.3):
            break
        
        # Soft-max weights over facets
        beta = 30.0
        exp_angles = np.exp(beta * (angles - max_angle))
        weights = exp_angles / np.sum(exp_angles)
        
        # Compute circumcenters for all facets
        centers = _facet_circumcenters(best_pts, simplices)
        
        # Compute gradient: weighted sum of "move toward circumcenter" directions
        grad = np.zeros_like(best_pts)
        
        # Only process facets with significant weight
        significant = weights > 1e-4
        if not np.any(significant):
            break
        
        sig_indices = np.where(significant)[0]
        sig_weights = weights[sig_indices]
        sig_simplices = simplices[sig_indices]
        sig_centers = centers[sig_indices]
        
        for w, simplex, center in zip(sig_weights, sig_simplices, sig_centers):
            for idx in simplex:
                v = best_pts[idx]
                direction = center - np.dot(v, center) * v
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-10:
                    grad[idx] += w * direction / dir_norm
        
        # Apply gradient step
        if np.linalg.norm(grad) > 1e-15:
            grad_norm = np.linalg.norm(grad)
            grad = grad / grad_norm * step_size
            
            new_pts = best_pts + grad
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius - 1e-10:
                best_pts = new_pts
                best_radius = new_radius
                no_improve_count = 0
                step_size = min(step_size * 1.03, 0.08)
            else:
                step_size *= 0.75
                no_improve_count += 1
                if no_improve_count > 5 or step_size < 0.001:
                    break
    
    return best_pts, best_radius

def _polish_step(pts, start_time, max_time=120.0, max_iters=150):
    """
    Final polish: smooth surrogate for max circumradius with vectorized ops.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    beta = 40.0
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 10.0:
            break
        
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            break
        
        simplices = hull.simplices
        angles = _facet_circumradius_batch(best_pts, simplices)
        
        if len(angles) == 0:
            break
        
        max_angle = np.max(angles)
        if max_angle < 1e-15:
            break
        
        # Soft-max weights
        exp_angles = np.exp(beta * (angles - max_angle))
        weights = exp_angles / np.sum(exp_angles)
        
        # Compute circumcenters
        centers = _facet_circumcenters(best_pts, simplices)
        
        # Gradient
        grad = np.zeros_like(best_pts)
        
        significant = weights > 1e-4
        if not np.any(significant):
            break
        
        sig_indices = np.where(significant)[0]
        sig_weights = weights[sig_indices]
        sig_simplices = simplices[sig_indices]
        sig_centers = centers[sig_indices]
        
        for w, simplex, center in zip(sig_weights, sig_simplices, sig_centers):
            for idx in simplex:
                v = best_pts[idx]
                direction = center - np.dot(v, center) * v
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-10:
                    grad[idx] += w * direction / dir_norm
        
        # Vectorized repulsion for close points
        diff_mat = best_pts[:, None, :] - best_pts[None, :, :]
        dist_mat = np.linalg.norm(diff_mat, axis=2)
        np.fill_diagonal(dist_mat, 1.0)
        
        close_mask = dist_mat < 0.05
        np.fill_diagonal(close_mask, False)
        
        repulsion = np.zeros_like(best_pts)
        if np.any(close_mask):
            force_mat = np.where(close_mask, 0.01 / (dist_mat + 1e-10), 0.0)
            repulsion = np.sum(force_mat[:, :, None] * diff_mat, axis=1)
        
        total_grad = grad + 0.05 * repulsion
        
        if np.linalg.norm(total_grad) < 1e-15:
            break
        
        # Adaptive step size with backtracking
        step = 0.01
        improved = False
        for _ in range(4):
            new_pts = best_pts + step * total_grad
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius - 1e-12:
                best_pts = new_pts
                best_radius = new_radius
                improved = True
                break
            step *= 0.5
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 8:
                break
    
    return best_pts, best_radius

def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    max_time = 115.0  # Safety margin before 120s limit
    
    # Start with Fibonacci spiral
    pts = _initial_fib()
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Phase 1: Lloyd refinement
    elapsed = time.time() - start_time
    if elapsed < max_time - 40.0:
        lloyd_pts, lloyd_radius = _refine_lloyd(pts, start_time, max_time, max_iters=20)
        if lloyd_radius < best_radius:
            best_pts = lloyd_pts
            best_radius = lloyd_radius
    
    # Phase 2: Gradient descent on facets
    elapsed = time.time() - start_time
    if elapsed < max_time - 30.0:
        gd_pts, gd_radius = _gradient_descent_facets(best_pts, start_time, max_time)
        if gd_radius < best_radius:
            best_pts = gd_pts
            best_radius = gd_radius
    
    # Phase 3: Polish with soft-max surrogate
    elapsed = time.time() - start_time
    if elapsed < max_time - 15.0:
        polish_pts, polish_radius = _polish_step(best_pts, start_time, max_time)
        if polish_radius < best_radius:
            best_pts = polish_pts
            best_radius = polish_radius
    
    # Phase 4: Final Lloyd refinement
    elapsed = time.time() - start_time
    if elapsed < max_time - 8.0:
        final_pts, final_radius = _refine_lloyd(best_pts, start_time, max_time, max_iters=10)
        if final_radius < best_radius:
            best_pts = final_pts
            best_radius = final_radius
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-10:
                best_pts[j] += np.random.randn(3) * 0.001
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    # Final normalization
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END

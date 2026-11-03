import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        max_circum = 0.0
        for tri in hull.simplices:
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            # Compute circumcenter on sphere
            n = np.cross(a - b, a - c)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
            rad = np.arccos(dot_val)
            if rad > max_circum:
                max_circum = rad
        return np.degrees(max_circum)
    except Exception:
        return 180.0

def _get_worst_facet(pts):
    """Get the facet with largest circumradius and its circumcenter."""
    try:
        hull = ConvexHull(pts)
        max_facet_rad = 0.0
        worst_indices = None
        worst_circumcenter = None
        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            n = np.cross(a - b, a - c)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
            rad = np.arccos(dot_val)
            if rad > max_facet_rad:
                max_facet_rad = rad
                worst_indices = simplex
                worst_circumcenter = n
        return worst_indices, worst_circumcenter, max_facet_rad
    except Exception:
        return None, None, 0.0

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)

def _local_optimize(pts, max_time, initial_step=0.03, momentum=0.7):
    """Local optimization using facet-driven forces plus repulsion."""
    velocities = np.zeros_like(pts)
    step_size = initial_step
    best_pts = pts.copy()
    best_rad = _covering_radius_deg(pts)
    
    it = 0
    while time.time() < max_time:
        it += 1
        worst_indices, worst_circumcenter, max_facet_rad = _get_worst_facet(pts)
        if worst_indices is None:
            break
        
        # Gradient: move worst facet vertices towards circumcenter
        forces = np.zeros_like(pts)
        for j in range(3):
            idx = worst_indices[j]
            v = pts[idx]
            grad = worst_circumcenter - v
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            forces[idx] += grad
        
        # Add pairwise repulsion to prevent clustering
        for i in range(N):
            for j in range(i+1, N):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 0.3:  # Only repel nearby points
                    rep = diff / (dist + 1e-10)
                    strength = (0.3 - dist) / 0.3
                    forces[i] += strength * rep * 0.5
                    forces[j] -= strength * rep * 0.5
        
        # Update velocities with momentum
        velocities = momentum * velocities + step_size * forces
        new_pts = pts + velocities
        new_pts = _normalize(new_pts)
        
        # Check if improved
        new_rad = _covering_radius_deg(new_pts)
        if new_rad < best_rad:
            best_rad = new_rad
            best_pts = new_pts.copy()
            pts = new_pts.copy()
        else:
            # Revert and reduce step size
            step_size *= 0.95
            velocities *= 0.5
            if step_size < 1e-6:
                break
        
        # Add small noise for exploration
        noise = np.random.normal(0, step_size * 0.1, pts.shape)
        pts = _normalize(pts + noise)
    
    return best_pts, best_rad

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin
    
    # Initialize with Fibonacci spiral
    points = _fibonacci_spiral()
    best_points = points.copy()
    best_rad = _covering_radius_deg(best_points)
    
    # Multiple restarts with local optimization
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() - start_time > max_time - 5.0:
            break
        
        if restart == 0:
            pts = points.copy()
        else:
            # Perturbed initialization
            pts = points.copy()
            noise = np.random.normal(0, 0.1 * (restart + 1), pts.shape)
            pts = _normalize(pts + noise)
        
        # Local optimization
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time < 5.0:
            break
        
        opt_pts, opt_rad = _local_optimize(pts, start_time + remaining_time - 2.0)
        
        if opt_rad < best_rad:
            best_rad = opt_rad
            best_points = opt_pts.copy()
    
    # Final polish: very fine local optimization
    if time.time() - start_time < max_time - 10.0:
        polish_pts, polish_rad = _local_optimize(best_points, max_time - 2.0, 
                                                  initial_step=0.005, momentum=0.8)
        if polish_rad < best_rad:
            best_rad = polish_rad
            best_points = polish_pts.copy()
    
    return best_points

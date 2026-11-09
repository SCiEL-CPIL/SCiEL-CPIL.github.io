"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _normalize(v):
    """Normalize a vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n


def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of points on the unit sphere.
    Returns (max_rad, worst_simplex).
    """
    try:
        hull = ConvexHull(pts)
        max_rad = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            centroid = (a + b + c) / 3.0
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                continue
            centroid = centroid / centroid_norm
            
            if np.dot(n, centroid) < 0:
                n = -n
            
            cos_R = np.dot(n, a)
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            
            if R > max_rad:
                max_rad = R
                worst_simplex = simplex
        
        return max_rad, worst_simplex
    except Exception:
        return np.pi, None


def _covering_radius_degrees(pts):
    rad, _ = _covering_radius(pts)
    return np.degrees(rad)


def _slerp(v1, v2, t):
    """Spherical linear interpolation between two unit vectors."""
    v1 = _normalize(v1)
    v2 = _normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-12:
        return v1
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * v1 + (np.sin(t * theta) / sin_theta) * v2


def _get_face_normals_and_radii(pts):
    """
    Get all face normals and their circumradii from the convex hull.
    Returns (normals, radii, simplices) where normals[i] is the outward normal
    of face i, radii[i] is the circumradius of that face, and simplices[i] is
    the vertex indices of that face.
    """
    try:
        hull = ConvexHull(pts)
        normals = []
        radii = []
        simplices = []
        
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            centroid = (a + b + c) / 3.0
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                continue
            centroid = centroid / centroid_norm
            
            if np.dot(n, centroid) < 0:
                n = -n
            
            cos_R = np.dot(n, a)
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            
            normals.append(n)
            radii.append(R)
            simplices.append(simplex)
        
        return np.array(normals), np.array(radii), simplices
    except Exception:
        return np.array([]), np.array([]), []


def _uniformize_normals(pts, max_time=60.0, tol=1e-6):
    """
    Optimize face normals to have uniform circumradii using slerp-based updates.
    This treats the covering radius minimization as a constraint satisfaction problem
    on the dual polytope: we want all face normals to be at the same angular distance
    from their respective vertices.
    """
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_rad, _ = _covering_radius(pts)
    
    learning_rate = 0.1
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        normals, radii, simplices = _get_face_normals_and_radii(current_pts)
        
        if len(normals) == 0:
            break
        
        # Target radius: mean of all radii
        target_rad = np.mean(radii)
        
        # For each face, compute the direction to adjust its normal
        # to bring its circumradius closer to target_rad
        improved = False
        for i, (n, R, simplex) in enumerate(zip(normals, radii, simplices)):
            if time.time() - start_time > max_time:
                break
            
            # If this face has a larger radius than target, we need to move
            # its vertices closer to the normal (or move the normal closer to vertices)
            # We'll adjust the vertices of this face
            a_idx, b_idx, c_idx = simplex[0], simplex[1], simplex[2]
            
            # For each vertex of this face, compute the direction to move
            # to reduce the circumradius if R > target_rad, or increase if R < target_rad
            error = R - target_rad
            if abs(error) < tol:
                continue
            
            # Move each vertex of this face towards/away from the normal
            # to adjust the circumradius
            for idx in [a_idx, b_idx, c_idx]:
                p = current_pts[idx]
                
                # Direction on sphere towards normal n
                proj = np.dot(n, p)
                grad_dir = n - proj * p
                grad_norm = np.linalg.norm(grad_dir)
                if grad_norm < 1e-12:
                    continue
                grad_dir = grad_dir / grad_norm
                
                # Step size proportional to error
                step = learning_rate * error
                
                # Use slerp for smooth movement
                t = min(abs(step), 0.5)
                if step > 0:
                    # Move towards normal to reduce radius
                    new_p = _slerp(p, n, t)
                else:
                    # Move away from normal to increase radius
                    new_p = _slerp(p, -n, t)
                
                current_pts[idx] = new_p
                improved = True
        
        if not improved:
            break
        
        # Evaluate and keep best
        rad, _ = _covering_radius(current_pts)
        if rad < best_rad - 1e-10:
            best_rad = rad
            best_pts = current_pts.copy()
            no_improve_count = 0
            # Increase learning rate if improving
            learning_rate = min(learning_rate * 1.05, 0.5)
        else:
            no_improve_count += 1
            # Decrease learning rate if not improving
            learning_rate *= 0.95
            if no_improve_count > 20:
                break
    
    return best_pts


def _local_refinement(pts, max_time=30.0):
    """
    Local refinement by moving individual points to minimize covering radius.
    """
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_rad, _ = _covering_radius(pts)
    
    for iteration in range(100):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for idx in range(N):
            if time.time() - start_time > max_time:
                break
            
            orig_pt = current_pts[idx].copy()
            best_local = orig_pt
            best_local_rad, _ = _covering_radius(current_pts)
            
            # Try random perturbations
            for trial in range(20):
                if time.time() - start_time > max_time:
                    break
                
                noise = np.random.randn(3) * 0.05
                candidate = orig_pt + noise
                candidate = _normalize(candidate)
                
                current_pts[idx] = candidate
                rad, _ = _covering_radius(current_pts)
                
                if rad < best_local_rad:
                    best_local_rad = rad
                    best_local = candidate.copy()
            
            current_pts[idx] = best_local
            
            if best_local_rad < best_rad - 1e-10:
                best_rad = best_local_rad
                best_pts = current_pts.copy()
                improved = True
        
        if not improved:
            break
    
    return best_pts


def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    best_pts = pts.copy()
    best_rad, _ = _covering_radius(pts)
    
    # Phase 1: Uniformize face normals using slerp-based optimization
    if time.time() < deadline - 40.0:
        uniformized = _uniformize_normals(pts, max_time=35.0)
        rad, _ = _covering_radius(uniformized)
        if rad < best_rad:
            best_rad = rad
            best_pts = uniformized.copy()
    
    # Phase 2: Local refinement
    if time.time() < deadline - 20.0:
        locally_opt = _local_refinement(best_pts, max_time=18.0)
        rad, _ = _covering_radius(locally_opt)
        if rad < best_rad:
            best_rad = rad
            best_pts = locally_opt.copy()
    
    # Phase 3: Final uniformization pass
    if time.time() < deadline - 5.0:
        final = _uniformize_normals(best_pts, max_time=4.0)
        rad, _ = _covering_radius(final)
        if rad < best_rad:
            best_rad = rad
            best_pts = final.copy()
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END

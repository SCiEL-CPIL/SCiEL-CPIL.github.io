"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n


def _covering_radius_degrees(pts):
    """Compute the covering radius in degrees using vectorized operations."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0  # fallback: worst case
    
    max_radius = 0.0
    simplices = hull.simplices
    
    for simplex in simplices:
        face_pts = pts[simplex]
        
        if len(simplex) == 3:
            a, b, c = face_pts
            n1 = np.cross(b - a, c - a)
            n1_norm = np.linalg.norm(n1)
            if n1_norm < 1e-15:
                continue
            circumcenter = n1 / n1_norm
            
            # Check sign against facet normal
            facet_normal = n1 / n1_norm
            if np.dot(circumcenter, facet_normal) < 0:
                circumcenter = -circumcenter
            
            # Vectorized angular distance computation
            cos_angles = np.clip(np.dot(face_pts, circumcenter), -1.0, 1.0)
            angles = np.arccos(cos_angles)
            face_radius = np.max(angles)
            max_radius = max(max_radius, face_radius)
        
        elif len(simplex) == 4:
            a, b, c, d = face_pts
            M = np.array([b - a, c - a, d - a])
            try:
                u, s, vh = np.linalg.svd(M)
                x = vh[-1]
                x_norm = np.linalg.norm(x)
                if x_norm < 1e-15:
                    continue
                x = x / x_norm
                
                # Check sign
                centroid = a + b + c + d
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-15:
                    centroid = centroid / centroid_norm
                    if np.dot(x, centroid) < 0:
                        x = -x
                
                # Vectorized angular distance computation
                cos_angles = np.clip(np.dot(face_pts, x), -1.0, 1.0)
                angles = np.arccos(cos_angles)
                face_radius = np.max(angles)
                max_radius = max(max_radius, face_radius)
            except:
                continue
    
    return np.degrees(max_radius)


def _objective(pts):
    """Objective function: covering radius in degrees."""
    return _covering_radius_degrees(pts)


def _find_worst_facet(pts):
    """Find the facet with the largest circumradius and return its circumcenter and indices."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None
    
    max_radius = 0.0
    worst_simplex = None
    worst_circumcenter = None
    
    for simplex in hull.simplices:
        face_pts = pts[simplex]
        
        if len(simplex) == 3:
            a, b, c = face_pts
            n1 = np.cross(b - a, c - a)
            n1_norm = np.linalg.norm(n1)
            if n1_norm < 1e-15:
                continue
            circumcenter = n1 / n1_norm
            
            facet_normal = n1 / n1_norm
            if np.dot(circumcenter, facet_normal) < 0:
                circumcenter = -circumcenter
            
            cos_angles = np.clip(np.dot(face_pts, circumcenter), -1.0, 1.0)
            angles = np.arccos(cos_angles)
            face_radius = np.max(angles)
            
            if face_radius > max_radius:
                max_radius = face_radius
                worst_simplex = simplex
                worst_circumcenter = circumcenter
        
        elif len(simplex) == 4:
            a, b, c, d = face_pts
            M = np.array([b - a, c - a, d - a])
            try:
                u, s, vh = np.linalg.svd(M)
                x = vh[-1]
                x_norm = np.linalg.norm(x)
                if x_norm < 1e-15:
                    continue
                x = x / x_norm
                
                centroid = a + b + c + d
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-15:
                    centroid = centroid / centroid_norm
                    if np.dot(x, centroid) < 0:
                        x = -x
                
                cos_angles = np.clip(np.dot(face_pts, x), -1.0, 1.0)
                angles = np.arccos(cos_angles)
                face_radius = np.max(angles)
                
                if face_radius > max_radius:
                    max_radius = face_radius
                    worst_simplex = simplex
                    worst_circumcenter = x
            except:
                continue
    
    return worst_simplex, worst_circumcenter


def _gradient_step(pts, step_size=0.01):
    """Perform one gradient descent step on the sphere."""
    worst_simplex, circumcenter = _find_worst_facet(pts)
    
    if worst_simplex is None:
        return pts
    
    new_pts = pts.copy()
    for idx in worst_simplex:
        v = new_pts[idx]
        direction = circumcenter - v
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-15:
            continue
        direction = direction / dir_norm
        new_pts[idx] = _normalize(v + step_size * direction)
    
    return new_pts


def _optimize_from_start(pts, deadline, max_iterations=5000):
    """Optimize from a given starting configuration."""
    best_pts = pts.copy()
    best_score = _objective(pts)
    current_pts = pts.copy()
    step_size = 0.05
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        new_pts = _gradient_step(current_pts, step_size)
        new_score = _objective(new_pts)
        
        if new_score < best_score - 1e-10:
            best_score = new_score
            best_pts = new_pts.copy()
            current_pts = new_pts
        else:
            step_size *= 0.85
            if step_size < 1e-7:
                break
    
    return best_pts, best_score


def _fibonacci_spiral(n, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _icosahedron_like(n, rng):
    """Generate points based on icosahedron vertices with jitter."""
    # Icosahedron vertices
    t = (1 + np.sqrt(5)) / 2
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    
    # Start with icosahedron and add more points
    pts = base.copy()
    while len(pts) < n:
        # Add points near existing ones with jitter
        idx = rng.integers(0, len(pts))
        new_pt = pts[idx] + rng.standard_normal(3) * 0.3
        new_pt = new_pt / np.linalg.norm(new_pt)
        pts = np.vstack([pts, new_pt])
    
    return pts[:n]


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety
    
    rng = np.random.default_rng(42)
    
    # Generate multiple diverse starting configurations
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # 2. Fibonacci spiral with different offset
    starts.append(_fibonacci_spiral(N, offset=0.3))
    
    # 3. Random points
    starts.append(_random_points(N, rng))
    
    # 4. Another random set
    starts.append(_random_points(N, rng))
    
    # 5. Icosahedron-based
    starts.append(_icosahedron_like(N, rng))
    
    # 6. Random with different seed
    rng2 = np.random.default_rng(123)
    starts.append(_random_points(N, rng2))
    
    # 7. Fibonacci with larger offset
    starts.append(_fibonacci_spiral(N, offset=0.1))
    
    # 8. Another icosahedron-based
    rng3 = np.random.default_rng(456)
    starts.append(_icosahedron_like(N, rng3))
    
    best_pts = None
    best_score = np.inf
    
    # Optimize from each starting configuration
    for i, start_pts in enumerate(starts):
        if time.time() > deadline:
            break
        
        local_best, local_score = _optimize_from_start(start_pts, deadline)
        
        if local_score < best_score:
            best_score = local_score
            best_pts = local_best.copy()
    
    # Final refinement: try small perturbations of the best solution
    if best_pts is not None and time.time() < deadline - 5.0:
        for _ in range(5):
            if time.time() > deadline:
                break
            perturbed = best_pts + rng.standard_normal((N, 3)) * 0.01
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            perturbed = perturbed / norms
            
            local_best, local_score = _optimize_from_start(perturbed, deadline, max_iterations=2000)
            if local_score < best_score:
                best_score = local_score
                best_pts = local_best.copy()
    
    if best_pts is None:
        # Fallback to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END

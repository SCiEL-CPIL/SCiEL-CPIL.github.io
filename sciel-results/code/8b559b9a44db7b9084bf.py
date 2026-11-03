import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _fibonacci_init(seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _circumcenter_robust(p1, p2, p3):
    """Compute circumcenter of triangle on unit sphere, robust to degeneracy."""
    edges = [
        (p2 - p1, p3 - p1),
        (p1 - p2, p3 - p2),
        (p1 - p3, p2 - p3)
    ]
    
    for v1, v2 in edges:
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        
        n = n / n_norm
        center = n.copy()
        
        dot1 = np.dot(center, p1)
        dot2 = np.dot(center, p2)
        dot3 = np.dot(center, p3)
        
        if dot1 > 0 and dot2 > 0 and dot3 > 0:
            return center
        else:
            return -center
    
    # Fallback: average of the three points, normalized
    center = (p1 + p2 + p3) / 3.0
    norm = np.linalg.norm(center)
    if norm < 1e-10:
        return p1.copy()
    return center / norm

def _compute_circumradii(pts, hull):
    """
    Compute angular circumradii for all triangular facets of the convex hull.
    Returns array of circumradii (in radians) and the circumcenters.
    """
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    
    for k, tri in enumerate(simplices):
        v0 = pts[tri[0]]
        v1 = pts[tri[1]]
        v2 = pts[tri[2]]
        
        c = _circumcenter_robust(v0, v1, v2)
        centers[k] = c
        
        dot = np.clip(np.dot(c, v0), -1.0, 1.0)
        radii[k] = np.arccos(dot)
        
    return radii, centers

def _covering_radius(pts):
    """Compute the covering radius in degrees using convex hull."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0, None, None
        
    radii, centers = _compute_circumradii(pts, hull)
    max_radius = np.max(radii)
    return np.degrees(max_radius), radii, centers

def _repulsive_force(pts):
    """Compute repulsive force between all pairs of points to prevent clustering."""
    n = len(pts)
    rep_force = np.zeros_like(pts)
    
    # Compute pairwise distances
    dots = pts @ pts.T
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    D = np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, 4.0))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                d_ij = D[i, j]
                if d_ij < 1e-6:
                    f = 100.0
                else:
                    f = 1.0 / (d_ij * d_ij)
                rep_force[i] += f * (pts[i] - pts[j])
                
    return rep_force

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initialize with Fibonacci spiral
    pts = _fibonacci_init()
    
    best_cr = 180.0
    best_pts = pts.copy()
    
    # Pre-compute initial hull
    try:
        hull = ConvexHull(pts)
        radii, centers = _compute_circumradii(pts, hull)
        current_cr = np.degrees(np.max(radii))
    except Exception:
        return pts
        
    if current_cr < best_cr:
        best_cr = current_cr
        best_pts = pts.copy()
        
    step_size_greedy = 0.05
    step_size_lloyd = 0.02
    no_improve_count = 0
    max_no_improve = 50
    iteration = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Alternate between Lloyd and Greedy
        use_lloyd = (iteration % 6 == 0)
        
        if use_lloyd:
            # Lloyd step: move each point towards centroid of adjacent facet circumcenters
            simplices = hull.simplices
            point_to_facets = [[] for _ in range(N)]
            for k, tri in enumerate(simplices):
                for idx in tri:
                    point_to_facets[idx].append(k)
                    
            new_pts = pts.copy()
            for i in range(N):
                facets = point_to_facets[i]
                if not facets:
                    continue
                    
                sum_c = np.sum(centers[facets], axis=0)
                norm_c = np.linalg.norm(sum_c)
                if norm_c < 1e-10:
                    continue
                    
                target = sum_c / norm_c
                new_pts[i] = pts[i] + step_size_lloyd * (target - pts[i])
                
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            pts = new_pts
        else:
            # Greedy: Shrink the largest facet
            max_idx = np.argmax(radii)
            max_rad = radii[max_idx]
            
            if max_rad < 1e-6:
                break
                
            tri = hull.simplices[max_idx]
            c = centers[max_idx]
            
            moves = np.zeros(3)
            for idx in tri:
                v = pts[idx]
                # Project c onto tangent plane at v
                proj = np.dot(v, c) * c
                tangent = c - proj
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-10:
                    moves += tangent / norm_t
                    
            move_norm = np.linalg.norm(moves)
            if move_norm < 1e-10:
                # If stuck, try a small random perturbation
                pts += np.random.normal(0, 0.001, (N, 3))
                pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            else:
                moves = moves / move_norm
                for idx in tri:
                    pts[idx] += step_size_greedy * moves
                    pts[idx] /= np.linalg.norm(pts[idx])
        
        # Update hull and radii
        try:
            hull = ConvexHull(pts)
            radii, centers = _compute_circumradii(pts, hull)
            new_cr = np.degrees(np.max(radii))
            
            if new_cr < best_cr:
                best_cr = new_cr
                best_pts = pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Apply repulsive force to break symmetry
                    rep = _repulsive_force(pts)
                    pts = pts + 0.01 * rep
                    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
                    
                    # Reduce step sizes
                    step_size_greedy *= 0.9
                    step_size_lloyd *= 0.9
                    no_improve_count = 0
                    if step_size_greedy < 1e-5:
                        break
        except Exception:
            # Hull failed, revert to best
            pts = best_pts.copy()
            break
            
    return best_pts

construct()

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_DEADLINE = 140.0  # safety margin before 150s kill

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull Delaunay triangulation."""
    pts = np.asarray(pts, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    if np.any(norms < 1e-15):
        return 180.0
    pts = pts / norms
    
    # Check for coincident points
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    off_diag = dots - np.eye(N)
    if np.min(off_diag) > 1 - 1e-10:
        return 180.0
    
    # Check if origin is inside convex hull (not in a hemisphere)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # Check origin inside hull
    from scipy.spatial import HalfspaceIntersection
    # Simpler: check if all face normals point away from origin
    inside = True
    for simplex in hull.simplices:
        face = pts[simplex]
        if len(face) < 3:
            continue
        # Compute face normal
        v1 = face[1] - face[0]
        v2 = face[2] - face[0]
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            continue
        n = n / nn
        # Distance from origin to face plane
        dist = np.dot(n, face[0])
        # For origin to be inside, it should be on the opposite side of the outward normal
        # The outward normal should point away from origin, so dist should be > 0
        # if n is outward. But we need consistent orientation.
        # Check: the centroid of the hull should be on the same side as origin
        # Actually, for a convex hull containing origin, for each face with outward normal n,
        # we should have dot(n, face[0]) > 0 (origin is behind the face)
        # But the orientation from ConvexHull may not be consistent.
        # Let's just compute the circumradii and take the max.
        pass
    
    # Compute covering radius as max circumradius over Delaunay triangles
    max_cr = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Spherical circumcenter: point equidistant from v0, v1, v2 on the sphere
        # Solve: ||x - v0|| = ||x - v1|| = ||x - v2||, ||x|| = 1
        # This gives: x·(v1-v0) = 0, x·(v2-v0) = 0
        # So x is parallel to cross(v1-v0, v2-v0)
        u = v1 - v0
        v = v2 - v0
        w = np.cross(u, v)
        wn = np.linalg.norm(w)
        if wn < 1e-15:
            # Degenerate, use midpoint of longest edge
            d01 = np.linalg.norm(v0 - v1)
            d02 = np.linalg.norm(v0 - v2)
            d12 = np.linalg.norm(v1 - v2)
            if d01 >= d02 and d01 >= d12:
                c = (v0 + v1) / 2.0
            elif d02 >= d01 and d02 >= d12:
                c = (v0 + v2) / 2.0
            else:
                c = (v1 + v2) / 2.0
            cn = np.linalg.norm(c)
            if cn < 1e-15:
                continue
            c = c / cn
        else:
            c = w / wn
            # Choose the hemisphere containing the face
            if np.dot(c, v0) < 0:
                c = -c
        
        # Angular distance from c to v0 (should be same for v1, v2)
        cos_d = np.clip(np.dot(c, v0), -1.0, 1.0)
        ang = np.arccos(cos_d)
        if ang > max_cr:
            max_cr = ang
    
    return np.degrees(max_cr)


def _smooth_objective(x):
    """Smooth objective: log-sum-exp of k-th nearest neighbor distances."""
    pts = x.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Compute all pairwise angular distances
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    angles = np.arccos(dots)  # (N, N)
    
    # For each point, find the 8th nearest neighbor distance
    # (excluding self)
    k = 8
    # Sort distances for each point
    sorted_angles = np.sort(angles, axis=1)
    # k-th nearest is at index k (index 0 is self)
    kth_nn = sorted_angles[:, k]
    
    # Log-sum-exp for smooth max
    # Use a temperature to control smoothness
    beta = 50.0
    max_val = np.max(kth_nn)
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * (kth_nn - max_val)))) / beta
    
    return log_sum_exp


def _fibonacci_sphere(seed=0):
    """Generate n points on the sphere using the Fibonacci spiral with random rotation."""
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Apply random rotation
    # Random rotation matrix
    q = rng.randn(4)
    q = q / np.linalg.norm(q)
    # Quaternion to rotation matrix
    w, x, y, zq = q
    R = np.array([
        [1 - 2*(y*y + zq*zq), 2*(x*y - w*zq), 2*(x*zq + w*y)],
        [2*(x*y + w*zq), 1 - 2*(x*x + zq*zq), 2*(y*zq - w*x)],
        [2*(x*zq - w*y), 2*(y*zq + w*x), 1 - 2*(x*x + y*y)]
    ])
    pts = pts @ R.T
    return pts


def _random_sphere(seed=0):
    """Generate n random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _optimize_lbgfsb(pts0, maxiter=200, deadline=140.0):
    """Run L-BFGS-B optimization."""
    x0 = pts0.flatten()
    
    def obj(x):
        return _smooth_objective(x)
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B',
            maxiter=maxiter,
            options={'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        opt_pts = result.x.reshape(-1, 3)
        opt_pts /= np.linalg.norm(opt_pts, axis=1, keepdims=True)
        return opt_pts
    except Exception:
        return pts0


def _greedy_refine(points, max_iter=20, deadline=140.0):
    """Greedy refinement: move points to reduce the worst circumradius."""
    points = points.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the worst triangle
        worst_rad = 0.0
        worst_tri = None
        worst_center = None
        
        for simplex in hull.simplices:
            if len(simplex) != 3:
                continue
            v0, v1, v2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            u = v1 - v0
            v = v2 - v0
            w = np.cross(u, v)
            wn = np.linalg.norm(w)
            if wn < 1e-15:
                continue
            c = w / wn
            if np.dot(c, v0) < 0:
                c = -c
            cos_d = np.clip(np.dot(c, v0), -1.0, 1.0)
            ang = np.degrees(np.arccos(cos_d))
            if ang > worst_rad:
                worst_rad = ang
                worst_tri = simplex
                worst_center = c
        
        if worst_tri is None:
            break
        
        # Try moving each vertex of the worst triangle toward the circumcenter
        best_pts = points.copy()
        best_rad = worst_rad
        
        for vi in worst_tri:
            new_pts = points.copy()
            # Move point vi toward worst_center
            step = 0.02
            direction = worst_center - points[vi]
            dn = np.linalg.norm(direction)
            if dn < 1e-10:
                continue
            direction = direction / dn
            new_pts[vi] = points[vi] + step * direction
            new_pts[vi] /= np.linalg.norm(new_pts[vi])
            
            new_rad = _covering_radius_deg(new_pts)
            if new_rad < best_rad - 1e-6:
                best_rad = new_rad
                best_pts = new_pts.copy()
        
        if best_rad < worst_rad - 1e-6:
            points = best_pts
        else:
            break
    
    return points


def construct():
    start_time = time.time()
    
    # Generate multiple initializations
    inits = []
    for seed in range(6):
        inits.append(_fibonacci_sphere(seed))
    for seed in range(4):
        inits.append(_random_sphere(seed))
    
    # Evaluate all initializations
    init_scores = []
    for pts in inits:
        if time.time() > _DEADLINE - 60:
            break
        cr = _covering_radius_deg(pts)
        init_scores.append((cr, pts))
    
    # Sort by covering radius
    init_scores.sort(key=lambda x: x[0])
    
    best_pts = init_scores[0][1].copy()
    best_cr = init_scores[0][0]
    
    # Optimize from the best initializations
    for cr, pts in init_scores[:4]:
        if time.time() > _DEADLINE - 20:
            break
        try:
            opt_pts = _optimize_lbgfsb(pts, maxiter=150, deadline=_DEADLINE)
            cr_opt = _covering_radius_deg(opt_pts)
            if cr_opt < best_cr:
                best_cr = cr_opt
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # Final greedy refinement
    if time.time() < _DEADLINE - 5:
        refined = _greedy_refine(best_pts, max_iter=15, deadline=_DEADLINE)
        cr_ref = _covering_radius_deg(refined)
        if cr_ref < best_cr:
            best_pts = refined
    
    # Final normalization and validation
    best_pts = np.asarray(best_pts, dtype=np.float64)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts

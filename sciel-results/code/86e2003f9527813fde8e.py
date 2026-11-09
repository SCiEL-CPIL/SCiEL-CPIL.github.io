"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
from scipy.optimize import minimize


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n


def _slerp(v1, v2, t):
    """Spherical linear interpolation between two unit vectors."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-10:
        return v1
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * v1 + (np.sin(t * theta) / sin_theta) * v2


def _vectorized_slerp(v1, v2, t):
    """Vectorized slerp for arrays of unit vectors."""
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    dots = np.clip(np.sum(v1 * v2, axis=-1, keepdims=True), -1.0, 1.0)
    thetas = np.arccos(dots)
    sin_thetas = np.sin(thetas)
    sin_thetas[sin_thetas < 1e-10] = 1e-10
    coeff1 = np.sin((1 - t) * thetas) / sin_thetas
    coeff2 = np.sin(t * thetas) / sin_thetas
    return coeff1 * v1 + coeff2 * v2


def _compute_facet_radii(pts):
    """Compute circumradii for all facets. Returns (radii, simplices, circumcenters)."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None
    
    simplices = hull.simplices
    n_faces = len(simplices)
    radii = np.zeros(n_faces)
    circumcenters = np.zeros((n_faces, 3))
    
    for i, simplex in enumerate(simplices):
        face_pts = pts[simplex]
        
        if len(simplex) == 3:
            a, b, c = face_pts
            n1 = np.cross(b - a, c - a)
            n1_norm = np.linalg.norm(n1)
            if n1_norm < 1e-15:
                radii[i] = np.pi
                continue
            circumcenter = n1 / n1_norm
            if np.dot(circumcenter, face_pts.mean(axis=0)) < 0:
                circumcenter = -circumcenter
            circumcenters[i] = circumcenter
            cos_angles = np.clip(np.dot(face_pts, circumcenter), -1.0, 1.0)
            angles = np.arccos(cos_angles)
            radii[i] = np.max(angles)
        
        elif len(simplex) == 4:
            a, b, c, d = face_pts
            M = np.array([b - a, c - a, d - a])
            try:
                u, s, vh = np.linalg.svd(M)
                x = vh[-1]
                x_norm = np.linalg.norm(x)
                if x_norm < 1e-15:
                    radii[i] = np.pi
                    continue
                x = x / x_norm
                centroid = face_pts.mean(axis=0)
                if np.dot(x, centroid) < 0:
                    x = -x
                circumcenters[i] = x
                cos_angles = np.clip(np.dot(face_pts, x), -1.0, 1.0)
                angles = np.arccos(cos_angles)
                radii[i] = np.max(angles)
            except:
                radii[i] = np.pi
    
    return radii, simplices, circumcenters


def _compute_face_normals(pts):
    """Compute outward face normals for the convex hull."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None
    
    simplices = hull.simplices
    normals = np.zeros((len(simplices), 3))
    
    for i, simplex in enumerate(simplices):
        face_pts = pts[simplex]
        if len(simplex) == 3:
            a, b, c = face_pts
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                normals[i] = face_pts.mean(axis=0)
            else:
                normals[i] = n / n_norm
                if np.dot(normals[i], face_pts.mean(axis=0)) < 0:
                    normals[i] = -normals[i]
        elif len(simplex) == 4:
            a, b, c, d = face_pts
            M = np.array([b - a, c - a, d - a])
            try:
                u, s, vh = np.linalg.svd(M)
                x = vh[-1]
                x_norm = np.linalg.norm(x)
                if x_norm < 1e-15:
                    normals[i] = face_pts.mean(axis=0)
                else:
                    x = x / x_norm
                    centroid = face_pts.mean(axis=0)
                    if np.dot(x, centroid) < 0:
                        x = -x
                    normals[i] = x
            except:
                normals[i] = face_pts.mean(axis=0)
    
    return normals, simplices


def _covering_radius_degrees(pts):
    """Compute the covering radius in degrees."""
    radii, _, _ = _compute_facet_radii(pts)
    if radii is None:
        return 180.0
    return np.degrees(np.max(radii))


def _objective(pts):
    """Objective function: covering radius in degrees."""
    return _covering_radius_degrees(pts)


def _equalize_circumradii(pts, deadline, max_iterations=3000):
    """Equalize facet circumradii using adaptive step sizing."""
    best_pts = pts.copy()
    best_score = _objective(pts)
    current_pts = pts.copy()
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        radii, simplices, circumcenters = _compute_facet_radii(current_pts)
        if radii is None:
            break
        
        max_rad = np.max(radii)
        min_rad = np.min(radii)
        target_rad = (max_rad + min_rad) / 2
        
        if max_rad - min_rad < 1e-8:
            break
        
        new_pts = current_pts.copy()
        moved = False
        
        for i, simplex in enumerate(simplices):
            r_i = radii[i]
            if r_i > target_rad + 1e-6:
                # This facet is too large, move its vertices toward its circumcenter
                excess = r_i - target_rad
                step = min(0.02, 0.005 * excess)
                cc = circumcenters[i]
                for idx in simplex:
                    v = new_pts[idx]
                    direction = cc - v
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm > 1e-10:
                        direction = direction / dir_norm
                        new_pts[idx] = _normalize(v + step * direction)
                        moved = True
            elif r_i < target_rad - 1e-6:
                # This facet is too small, move its vertices away from its circumcenter
                deficit = target_rad - r_i
                step = min(0.01, 0.003 * deficit)
                cc = circumcenters[i]
                for idx in simplex:
                    v = new_pts[idx]
                    direction = v - cc
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm > 1e-10:
                        direction = direction / dir_norm
                        new_pts[idx] = _normalize(v + step * direction)
                        moved = True
        
        if not moved:
            break
        
        new_score = _objective(new_pts)
        if new_score < best_score - 1e-10:
            best_score = new_score
            best_pts = new_pts.copy()
            current_pts = new_pts
        else:
            # Try smaller step
            new_pts2 = current_pts.copy()
            for i, simplex in enumerate(simplices):
                r_i = radii[i]
                if r_i > target_rad + 1e-6:
                    excess = r_i - target_rad
                    step = min(0.01, 0.002 * excess)
                    cc = circumcenters[i]
                    for idx in simplex:
                        v = new_pts2[idx]
                        direction = cc - v
                        dir_norm = np.linalg.norm(direction)
                        if dir_norm > 1e-10:
                            direction = direction / dir_norm
                            new_pts2[idx] = _normalize(v + step * direction)
            new_score2 = _objective(new_pts2)
            if new_score2 < best_score - 1e-10:
                best_score = new_score2
                best_pts = new_pts2.copy()
                current_pts = new_pts2
            else:
                break
    
    return best_pts, best_score


def _delaunay_face_normal_optimize(pts, deadline, max_iterations=2000):
    """Optimize using Delaunay triangulation and face normal slerp."""
    best_pts = pts.copy()
    best_score = _objective(pts)
    current_pts = pts.copy()
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        radii, simplices, circumcenters = _compute_facet_radii(current_pts)
        if radii is None:
            break
        
        max_rad = np.max(radii)
        min_rad = np.min(radii)
        
        if max_rad - min_rad < 1e-7:
            break
        
        target_rad = (max_rad + min_rad) / 2
        new_pts = current_pts.copy()
        
        # For each face, compute the optimal position via slerp
        for i, simplex in enumerate(simplices):
            r_i = radii[i]
            cc = circumcenters[i]
            
            if r_i > target_rad + 1e-5:
                # Face too large: slerp vertices toward circumcenter
                excess = r_i - target_rad
                t = min(0.1, 0.01 * excess)
                for idx in simplex:
                    v = new_pts[idx]
                    new_pts[idx] = _slerp(v, cc, t)
            elif r_i < target_rad - 1e-5:
                # Face too small: slerp vertices away from circumcenter
                deficit = target_rad - r_i
                t = min(0.05, 0.005 * deficit)
                for idx in simplex:
                    v = new_pts[idx]
                    # Move away from circumcenter
                    away_dir = v - cc
                    away_dir = away_dir / np.linalg.norm(away_dir)
                    new_pts[idx] = _slerp(v, away_dir, t)
        
        new_score = _objective(new_pts)
        if new_score < best_score - 1e-10:
            best_score = new_score
            best_pts = new_pts.copy()
            current_pts = new_pts
        else:
            # Try with half step
            new_pts2 = current_pts.copy()
            for i, simplex in enumerate(simplices):
                r_i = radii[i]
                cc = circumcenters[i]
                
                if r_i > target_rad + 1e-5:
                    excess = r_i - target_rad
                    t = min(0.05, 0.005 * excess)
                    for idx in simplex:
                        v = new_pts2[idx]
                        new_pts2[idx] = _slerp(v, cc, t)
                elif r_i < target_rad - 1e-5:
                    deficit = target_rad - r_i
                    t = min(0.025, 0.0025 * deficit)
                    for idx in simplex:
                        v = new_pts2[idx]
                        away_dir = v - cc
                        away_dir = away_dir / np.linalg.norm(away_dir)
                        new_pts2[idx] = _slerp(v, away_dir, t)
            
            new_score2 = _objective(new_pts2)
            if new_score2 < best_score - 1e-10:
                best_score = new_score2
                best_pts = new_pts2.copy()
                current_pts = new_pts2
            else:
                break
    
    return best_pts, best_score


def _gradient_step(pts, step_size=0.01):
    """Perform one gradient descent step on the sphere."""
    radii, simplices, circumcenters = _compute_facet_radii(pts)
    if radii is None:
        return pts
    
    max_idx = np.argmax(radii)
    worst_simplex = simplices[max_idx]
    circumcenter = circumcenters[max_idx]
    
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


def _high_precision_polish(pts, deadline):
    """Add multiple micro-refinement passes with tighter decay and smaller steps."""
    best_pts = pts.copy()
    best_score = _objective(pts)
    
    for pass_num in range(3):
        if time.time() > deadline:
            break
        
        current_pts = best_pts.copy()
        step_size = 0.005 - pass_num * 0.001
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            new_pts = _gradient_step(current_pts, step_size)
            new_score = _objective(new_pts)
            
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts
            else:
                step_size *= 0.9998
                if step_size < 1e-9:
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
    t = (1 + np.sqrt(5)) / 2
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    
    pts = base.copy()
    while len(pts) < n:
        idx = rng.integers(0, len(pts))
        new_pt = pts[idx] + rng.standard_normal(3) * 0.3
        new_pt = new_pt / np.linalg.norm(new_pt)
        pts = np.vstack([pts, new_pt])
    
    return pts[:n]


def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    rng = np.random.default_rng(42)
    
    starts = []
    starts.append(_fibonacci_spiral(N))
    starts.append(_fibonacci_spiral(N, offset=0.3))
    starts.append(_random_points(N, rng))
    starts.append(_random_points(N, rng))
    starts.append(_icosahedron_like(N, rng))
    rng2 = np.random.default_rng(123)
    starts.append(_random_points(N, rng2))
    starts.append(_fibonacci_spiral(N, offset=0.1))
    rng3 = np.random.default_rng(456)
    starts.append(_icosahedron_like(N, rng3))
    
    best_pts = None
    best_score = np.inf
    
    for i, start_pts in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        local_best, local_score = _optimize_from_start(start_pts, deadline - 30.0)
        
        if local_score < best_score:
            best_score = local_score
            best_pts = local_best.copy()
    
    # Delaunay face normal optimization phase (new)
    if best_pts is not None and time.time() < deadline - 20.0:
        del_pts, del_score = _delaunay_face_normal_optimize(best_pts, deadline - 20.0)
        if del_score < best_score:
            best_score = del_score
            best_pts = del_pts.copy()
    
    # Equalize circumradii phase
    if best_pts is not None and time.time() < deadline - 10.0:
        eq_pts, eq_score = _equalize_circumradii(best_pts, deadline - 10.0)
        if eq_score < best_score:
            best_score = eq_score
            best_pts = eq_pts.copy()
    
    # High precision polish
    if best_pts is not None and time.time() < deadline - 5.0:
        polish_pts, polish_score = _high_precision_polish(best_pts, deadline)
        if polish_score < best_score:
            best_score = polish_score
            best_pts = polish_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END

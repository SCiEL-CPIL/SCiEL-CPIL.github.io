import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _compute_covering_radius(points):
    """Compute exact covering radius via convex hull facet circumradii (in radians)."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    faces = hull.simplices
    max_circ = 0.0
    
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        if circumradius > max_circ:
            max_circ = circumradius
    
    return max_circ

def _compute_facet_circumradii_and_centers(points):
    """
    For each point, compute the maximum circumradius among its adjacent facets,
    and the direction (toward the Voronoi vertex of the worst facet) to move.
    """
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.ones(N) * np.pi, np.zeros((N, 3))
    
    faces = hull.simplices
    max_circ_per_point = np.zeros(N)
    move_dirs = np.zeros((N, 3))
    
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        for idx, angle in zip(face, angles):
            if angle >= max_circ_per_point[idx]:
                max_circ_per_point[idx] = angle
                move_dirs[idx] = o
    
    norms = np.linalg.norm(move_dirs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    move_dirs = move_dirs / norms
    
    return max_circ_per_point, move_dirs

def _force_directed_step(pts, lr=0.02):
    """
    Fast force-directed step using Coulomb repulsion (no ConvexHull).
    O(N^2) per step, enabling many more iterations.
    """
    pts = _normalize(pts.copy())
    
    # Compute pairwise 3D distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist3_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist3_sq, 1.0)
    dist3 = np.sqrt(dist3_sq)
    
    # Coulomb force: F_i = sum_j (pts[i] - pts[j]) / dist3^3
    inv_dist3 = 1.0 / (dist3 ** 3)
    inv_dist3[dist3 < 1e-6] = 0.0
    F_coulomb = (diff * inv_dist3[:, :, np.newaxis]).sum(axis=1)
    
    # Add small regularization to prevent numerical issues
    F_norms = np.linalg.norm(F_coulomb, axis=1, keepdims=True)
    F_norms[F_norms < 1e-12] = 1.0
    F_coulomb = F_coulomb / F_norms
    
    new_pts = pts + lr * F_coulomb
    return _normalize(new_pts)

def _force_directed_optimize(points, max_iters=500, lr_init=0.02, lr_decay=0.999, deadline=None):
    """
    Fast force-directed optimization using Coulomb repulsion.
    No ConvexHull calls - pure O(N^2) per step.
    """
    P = points.copy()
    best_P = P.copy()
    best_radius = _compute_covering_radius(P)
    lr = lr_init
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        P = _force_directed_step(P, lr)
        
        # Check improvement periodically (every 20 iterations to save time)
        if (it + 1) % 20 == 0:
            current_radius = _compute_covering_radius(P)
            if current_radius < best_radius:
                best_radius = current_radius
                best_P = P.copy()
        
        lr *= lr_decay
    
    # Final check
    current_radius = _compute_covering_radius(P)
    if current_radius < best_radius:
        best_radius = current_radius
        best_P = P.copy()
    
    return best_P, best_radius

def _facet_based_optimize(points, max_iters=100, deadline=None):
    """
    Lloyd-type optimization: move each point toward the Voronoi vertex of its worst facet.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        max_circ, move_dirs = _compute_facet_circumradii_and_centers(points)
        current_radius = np.max(max_circ)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Adaptive step size
        step = 0.05 * np.sqrt(current_radius)
        step = np.clip(step, 0.001, 0.05)
        
        # Move points toward their worst facet Voronoi vertices
        new_points = points + step * move_dirs
        new_points = _normalize(new_points)
        
        # Check if this improved things; if not, try smaller step
        new_radius = _compute_covering_radius(new_points)
        if new_radius < current_radius:
            points = new_points
        else:
            # Try half step
            new_points = points + 0.5 * step * move_dirs
            new_points = _normalize(new_points)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < current_radius:
                points = new_points
            else:
                # Try quarter step
                new_points = points + 0.25 * step * move_dirs
                new_points = _normalize(new_points)
                new_radius = _compute_covering_radius(new_points)
                if new_radius < current_radius:
                    points = new_points
                else:
                    break
    
    return best_points, best_radius

def _nelder_mead_polish(points, max_iters=30, deadline=None):
    """
    Polish the configuration using Nelder-Mead on the exact covering radius.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return best_radius
        pts = x.reshape(N, 3)
        pts = _normalize(pts)
        return _compute_covering_radius(pts)
    
    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if res.fun < best_radius:
            new_points = res.x.reshape(N, 3)
            new_points = _normalize(new_points)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_radius:
                best_points = new_points
                best_radius = new_radius
    except Exception:
        pass
    
    return best_points, best_radius

def _fibonacci_spiral(n, seed=42):
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations to try
    inits = [_fibonacci_spiral(N)]
    
    # Add perturbed versions
    rng = np.random.RandomState(123)
    for _ in range(3):
        perturbed = inits[0].copy()
        noise = rng.randn(N, 3) * 0.05
        perturbed = perturbed + noise
        perturbed = _normalize(perturbed)
        inits.append(perturbed)
    
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        
        current_points = init.copy()
        current_radius = _compute_covering_radius(current_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
        
        # Phase 1: Fast force-directed optimization (no ConvexHull in loop)
        if time.time() < deadline - 35.0:
            points, radius = _force_directed_optimize(
                current_points, max_iters=500, lr_init=0.02, deadline=deadline - 35.0
            )
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 2: Facet-based Lloyd optimization (exact)
        if time.time() < deadline - 20.0:
            points, radius = _facet_based_optimize(
                current_points, max_iters=80, deadline=deadline - 20.0
            )
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 3: Nelder-Mead polish
        if time.time() < deadline - 10.0:
            points, radius = _nelder_mead_polish(
                current_points, max_iters=30, deadline=deadline - 10.0
            )
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
    
    # Final polish on the best configuration
    if best_points is not None and time.time() < deadline - 5.0:
        points, radius = _facet_based_optimize(
            best_points, max_iters=40, deadline=deadline - 5.0
        )
        if radius < best_radius:
            best_points = points
            best_radius = radius
        
        points, radius = _nelder_mead_polish(
            best_points, max_iters=20, deadline=deadline - 2.0
        )
        if radius < best_radius:
            best_points = points
            best_radius = radius
    
    # Ensure output is normalized
    if best_points is not None:
        best_points = _normalize(best_points)
    else:
        best_points = _fibonacci_spiral(N)
    
    return best_points

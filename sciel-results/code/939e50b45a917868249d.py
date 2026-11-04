import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

# Precompute all triple indices
_triples = np.array(list(itertools.combinations(range(N), 3)))
_idx_i = _triples[:, 0]
_idx_j = _triples[:, 1]
_idx_k = _triples[:, 2]

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles. points: (N,2)"""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area(points):
    """Convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """min triangle area / hull area."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        areas = _triangle_areas(points)
        return np.min(areas) / ha
    except Exception:
        return 0.0

def _exact_objective(x):
    """Negative of the score (to minimize). Nonsmooth."""
    points = x.reshape(-1, 2)
    s = _score(points)
    return -s

def _regular_polygon(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _halton_sequence(n, base):
    seq = []
    f = 1.0
    i = 1
    while i <= n:
        f /= base
        i = (i + 1) * base
        seq.append(f)
    return np.array(seq)

def _halton_points(n, seed_offset=1):
    """Halton sequence in 2D for diverse initial points."""
    xs = _halton_sequence(n, 2)
    ys = _halton_sequence(n, 3)
    # Map to unit disk
    angles = 2 * np.pi * xs
    radii = np.sqrt(ys)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _concentric_rings(n):
    """Points on concentric rings."""
    # 7 outer, 5 middle, 1 center
    pts = []
    a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts.append(np.column_stack([np.cos(a1), np.sin(a1)]))
    a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + 0.3
    pts.append(0.6 * np.column_stack([np.cos(a2), np.sin(a2)]))
    pts.append(np.array([[0.0, 0.0]]))
    return np.vstack(pts)

def _structured_grid(n):
    """4x4 grid minus 3 points."""
    gx, gy = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    pts = np.column_stack([gx.flatten(), gy.flatten()])
    # Remove 3 points
    mask = np.ones(16, dtype=bool)
    mask[0] = False
    mask[3] = False
    mask[12] = False
    return pts[mask]

def _angular_to_cartesian(radii, angles):
    """Convert (radii, angles) to Cartesian, sorted by angle for convexity."""
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _cartesian_to_angular(points):
    """Convert Cartesian to (radii, angles) sorted by angle."""
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    radii = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx)
    # Sort by angle
    order = np.argsort(angles)
    return radii[order], angles[order]

def _local_search_angular(points, max_iter=200, tol=1e-10):
    """Nelder-Mead on angular representation for local refinement."""
    radii, angles = _cartesian_to_angular(points)
    
    def obj(x):
        r = np.abs(x[:N]) + 1e-10  # ensure positive radii
        a = x[N:]
        pts = _angular_to_cartesian(r, a)
        return _exact_objective(pts)
    
    x0 = np.concatenate([radii, angles])
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        r = np.abs(res.x[:N]) + 1e-10
        a = res.x[N:]
        pts = _angular_to_cartesian(r, a)
        return pts
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    rng = np.random.default_rng(123)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular polygon
    candidates.append(_regular_polygon(N))
    
    # 2. Halton sequence
    candidates.append(_halton_points(N))
    
    # 3. Concentric rings
    candidates.append(_concentric_rings(N))
    
    # 4. Structured grid
    candidates.append(_structured_grid(N))
    
    # 5. Random points in disk (multiple)
    for seed in range(8):
        r = np.sqrt(rng.uniform(0.1, 1.0, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(pts)
    
    # 6. Circle with noise
    for seed in range(4):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1)
        radii = 1.0 + rng.uniform(-0.15, 0.15, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    best_score = 0.0
    best_points = candidates[0]
    
    # Phase 1: Nelder-Mead on exact objective for each candidate
    for i, x0 in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
        
        # Try Nelder-Mead directly on Cartesian
        try:
            res = minimize(_exact_objective, x0.flatten(), method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            pts = res.x.reshape(-1, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass
        
        # Also try angular local search
        if time.time() > deadline - 10.0:
            break
        try:
            pts_angular = _local_search_angular(x0, max_iter=200)
            s = _score(pts_angular)
            if s > best_score:
                best_score = s
                best_points = pts_angular
        except Exception:
            pass
    
    # Phase 2: Multi-start perturbations around best point
    if time.time() < deadline - 5.0:
        for trial in range(15):
            if time.time() > deadline - 3.0:
                break
            
            # Perturb best points
            noise_scale = 0.05 * (1.0 - trial / 15.0)
            pts_perturbed = best_points + rng.normal(0, noise_scale, (N, 2))
            
            try:
                res = minimize(_exact_objective, pts_perturbed.flatten(), method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                                       'adaptive': True})
                pts = res.x.reshape(-1, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
            except Exception:
                pass
            
            # Angular search from perturbed
            if time.time() > deadline - 3.0:
                break
            try:
                pts_a = _local_search_angular(pts_perturbed, max_iter=150)
                s = _score(pts_a)
                if s > best_score:
                    best_score = s
                    best_points = pts_a
            except Exception:
                pass
    
    # Phase 3: Final polish with more iterations
    if time.time() < deadline - 2.0:
        try:
            res = minimize(_exact_objective, best_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            pts = res.x.reshape(-1, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass
        
        # Angular final polish
        if time.time() < deadline - 1.0:
            try:
                pts_a = _local_search_angular(best_points, max_iter=300)
                s = _score(pts_a)
                if s > best_score:
                    best_score = s
                    best_points = pts_a
            except Exception:
                pass
    
    return best_points

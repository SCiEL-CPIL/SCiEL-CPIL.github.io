"""Heilbronn convex regions: maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas_vectorized(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    return 0.5 * np.abs(cross)

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas_vectorized(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _soft_min_objective(x, beta=100.0):
    """Soft-min of triangle areas / hull area, negative for minimization."""
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas_vectorized(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _support_to_points(log_dists, n_sides=13):
    """
    Convert log support distances to polygon vertices.
    The polygon is the intersection of half-planes n_i · x <= d_i.
    """
    d = np.exp(log_dists)
    d = np.maximum(d, 1e-10)
    
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    vertices = []
    n = n_sides
    for i in range(n):
        n1 = normals[i]
        n2 = normals[(i + 1) % n]
        d1 = d[i]
        d2 = d[(i + 1) % n]
        
        A = np.array([n1, n2])
        b = np.array([d1, d2])
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        
        valid = True
        for j in range(n):
            if j == i or j == (i + 1) % n:
                continue
            if normals[j] @ x > d[j] + 1e-10:
                valid = False
                break
        
        if valid:
            vertices.append(x)
    
    if len(vertices) < 3:
        return None
    
    return np.array(vertices)

def _halton_sequence(n, base):
    """Generate first n Halton sequence values in base."""
    result = []
    f = 1.0
    i = 1
    for _ in range(n):
        result.append(f)
        f /= base
        i += 1
        while i % base == 0:
            i //= base
            f /= base
    return result

def _halton_points(n, offset=0.0):
    """Generate n Halton sequence points in [0,1]^2."""
    x = _halton_sequence(n, 2)
    y = _halton_sequence(n, 3)
    pts = np.column_stack([x, y])
    pts = pts * 2 - 1
    return pts + offset

def _initial_log_dists(seed=0):
    """Generate initial log support distances for a regular polygon."""
    rng = np.random.RandomState(seed)
    base = np.log(1.0)
    return base + rng.normal(0, 0.1, size=N)

def _log_dist_to_score(x, beta=100.0):
    """Objective for log-distances: -soft_min_triangle_area / hull_area."""
    pts = _support_to_points(x, n_sides=N)
    if pts is None or len(pts) != N:
        return 1e6
    
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas_vectorized(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_score = 0.0
    best_pts = None
    
    # Phase 1: Differential evolution on log-support-distances
    def _de_objective(x):
        return _log_dist_to_score(x, beta=100.0)
    
    bounds = [(-2.3, 2.3)] * N
    
    try:
        result_de = differential_evolution(
            _de_objective,
            bounds,
            seed=42,
            maxiter=80,
            tol=1e-8,
            popsize=12,
            polish=True,
            init='sobol'
        )
        
        if result_de.fun < 1e5:
            x_best = result_de.x
            pts = _support_to_points(x_best, n_sides=N)
            if pts is not None and len(pts) == N:
                score = _score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 40:
        if best_pts is not None:
            return best_pts
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Phase 2: Multi-start L-BFGS-B on point coordinates
    for seed in range(25):
        if time.time() > deadline - 15:
            break
        
        if seed == 0:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        elif seed == 1:
            x0 = _halton_points(N).flatten()
        else:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
            rng = np.random.RandomState(seed)
            pts += rng.normal(0, 0.15, size=pts.shape)
            x0 = pts.flatten()
        
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14}
            )
            
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
# EVOLVE-BLOCK-END

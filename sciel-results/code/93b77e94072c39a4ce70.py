"""Heilbronn convex regions: maximize minimum triangle area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 13
_START_TIME = time.time()
_TIME_LIMIT = 140  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0, _TIME_LIMIT - _elapsed())

def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                ax = pts[j, 0] - pts[i, 0]
                ay = pts[j, 1] - pts[i, 1]
                bx = pts[k, 0] - pts[i, 0]
                by = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(ax * by - ay * bx)
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(N, 2)
    return -_score_points(pts)

def _normalize_unit_area(pts):
    """Scale points so convex hull has unit area."""
    hull_area = _hull_area(pts)
    if hull_area > 1e-12:
        scale = 1.0 / np.sqrt(hull_area)
        return pts * scale
    return pts

def _initial_ring(n_points, radius=1.0, n_rings=3):
    """Generate initial points on concentric rings for diversity."""
    pts = []
    # Place some points on outer ring
    n_outer = n_points // 2
    n_inner = n_points - n_outer
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    
    for a in angles_outer:
        pts.append([radius * np.cos(a), radius * np.sin(a)])
    for a in angles_inner:
        pts.append([0.5 * radius * np.cos(a), 0.5 * radius * np.sin(a)])
    
    return np.array(pts)

def _refine_local(pts, max_iter=500):
    """Refine points using Powell then Nelder-Mead."""
    x0 = pts.flatten()
    
    # Powell method first (good for smooth but non-convex)
    try:
        result = minimize(_neg_score, x0, method='Powell', 
                         options={'maxiter': max_iter, 'ftol': 1e-10, 'xtol': 1e-10})
        if result.fun < _neg_score(x0):
            x0 = result.x
    except Exception:
        pass
    
    # Then Nelder-Mead for fine-tuning
    try:
        result = minimize(_neg_score, x0, method='Nelder-Mead',
                         options={'maxiter': max_iter * 2, 'fatol': 1e-10, 'xatol': 1e-10})
        if result.fun < _neg_score(x0):
            x0 = result.x
    except Exception:
        pass
    
    return x0.reshape(N, 2)

def construct():
    best_score = -float('inf')
    best_pts = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Concentric rings with different parameters
    for radius in [0.8, 1.0, 1.2]:
        for n_outer_frac in [0.4, 0.5, 0.6]:
            n_outer = int(N * n_outer_frac)
            n_inner = N - n_outer
            angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
            angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / max(1, n_inner)
            
            pts = []
            for a in angles_outer:
                pts.append([radius * np.cos(a), radius * np.sin(a)])
            for a in angles_inner:
                pts.append([0.5 * radius * np.cos(a), 0.5 * radius * np.sin(a)])
            initial_configs.append(np.array(pts))
    
    # Regular polygon + center
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(pts_poly)
    
    # Random configurations
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts_rand = rng.rand(N, 2) * 2 - 1
        initial_configs.append(pts_rand)
    
    # Try each initial configuration with local refinement
    for i, init_pts in enumerate(initial_configs):
        if _remaining() < 5:
            break
        
        # Normalize to unit area
        pts_norm = _normalize_unit_area(init_pts)
        
        # Local refinement
        refined = _refine_local(pts_norm, max_iter=200)
        
        # Compute score
        score = _score_points(refined)
        if score > best_score:
            best_score = score
            best_pts = refined
    
    # If we have time, try differential evolution for global search
    if _remaining() > 30 and best_pts is not None:
        try:
            # Use differential evolution with bounded search space
            bounds = [(-2, 2)] * (2 * N)
            de_result = differential_evolution(
                _neg_score, bounds, seed=42, maxiter=50, tol=1e-8,
                polish=True, init='sobol', popsize=15
            )
            de_pts = de_result.x.reshape(N, 2)
            de_score = _score_points(de_pts)
            
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts
        except Exception:
            pass
    
    # Final normalization to ensure unit area
    if best_pts is not None:
        best_pts = _normalize_unit_area(best_pts)
    else:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts

import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# EVOLVE-BLOCK-START
import random

# Pre-compute triangle indices for vectorized area calculation
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def calc_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        if len(points) < 3:
            return 0.0
        
        # Calculate convex hull area
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        
        if hull_area < 1e-10:
            return 0.0
            
        # Vectorized triangle area calculation
        pi = points[_I]
        pj = points[_J]
        pk = points[_K]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        min_area = 0.5 * np.abs(cross).min()
        
        return min_area / hull_area
    except Exception:
        return 0.0

def objective(points_flat):
    """
    Objective function for minimization.
    Minimizes negative score.
    """
    points = points_flat.reshape(-1, 2)
    score = calc_score(points)
    return -score

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        ha = hull.volume
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_4ring_points(params):
    """
    Generate 14 points from 8 parameters:
    [r1, r2, r3, r4, phi1, phi2, phi3, phi4]
    Ring 1: 5 points (pentagon)
    Ring 2: 4 points (square)
    Ring 3: 3 points (triangle)
    Ring 4: 2 points (diameter)
    """
    r1, r2, r3, r4 = params[0], params[1], params[2], params[3]
    phi1, phi2, phi3, phi4 = params[4], params[5], params[6], params[7]
    
    # Ensure radii are positive
    r1 = max(r1, 0.01)
    r2 = max(r2, 0.01)
    r3 = max(r3, 0.01)
    r4 = max(r4, 0.01)
    
    pts = []
    # Ring 1: 5 points
    angles1 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + phi1
    pts.extend([r1 * np.cos(a), r1 * np.sin(a)] for a in angles1)
    
    # Ring 2: 4 points
    angles2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + phi2
    pts.extend([r2 * np.cos(a), r2 * np.sin(a)] for a in angles2)
    
    # Ring 3: 3 points
    angles3 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + phi3
    pts.extend([r3 * np.cos(a), r3 * np.sin(a)] for a in angles3)
    
    # Ring 4: 2 points
    angles4 = np.linspace(0, 2 * np.pi, 2, endpoint=False) + phi4
    pts.extend([r4 * np.cos(a), r4 * np.sin(a)] for a in angles4)
    
    return np.array(pts).reshape(-1, 2)

def _angular_objective_4ring(params):
    """
    Objective for 4-ring parameters.
    """
    try:
        pts = _generate_4ring_points(params)
        pts = _normalize(pts)
        sc = calc_score(pts)
        return -sc
    except Exception:
        return 100.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts_norm = _normalize(pts)
        sc = calc_score(pts_norm)
        if sc > best_score:
            best_score = sc
            best_points = pts_norm.copy()

    def _try_optimize_2d(init_points, method='Nelder-Mead', maxiter=2000, ftol=1e-10, xtol=1e-8):
        """Try optimizing in full 2D space from init_points."""
        if time.time() > deadline:
            return None
        try:
            init_norm = _normalize(init_points)
            _update_best(init_norm)
            
            if method == 'Nelder-Mead':
                res = scipy.optimize.minimize(
                    objective,
                    init_norm.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': xtol, 'fatol': ftol, 'adaptive': True}
                )
            elif method == 'Powell':
                res = scipy.optimize.minimize(
                    objective,
                    init_norm.flatten(),
                    method='Powell',
                    options={'maxiter': maxiter, 'ftol': ftol, 'xtol': xtol}
                )
            else:
                return None
            
            opt_pts = _normalize(res.x.reshape(-1, 2))
            _update_best(opt_pts)
            
            return (calc_score(opt_pts), opt_pts)
        except Exception:
            return None

    # Phase 1: Optimize 4-ring structure (8 parameters)
    # This is a structural change from the 7-7 ring.
    # We try multiple random starts for the 8 parameters.
    for i in range(20):
        if time.time() > deadline - 30:
            break
        
        rng = np.random.RandomState(1000 + i)
        # Initial radii: spread out
        r1 = rng.uniform(0.8, 1.2)
        r2 = rng.uniform(0.5, 0.9)
        r3 = rng.uniform(0.2, 0.6)
        r4 = rng.uniform(0.0, 0.4)
        
        # Initial rotations: random
        phi1 = rng.uniform(0, 2 * np.pi)
        phi2 = rng.uniform(0, 2 * np.pi)
        phi3 = rng.uniform(0, 2 * np.pi)
        phi4 = rng.uniform(0, 2 * np.pi)
        
        init_params = np.array([r1, r2, r3, r4, phi1, phi2, phi3, phi4])
        
        try:
            res = scipy.optimize.minimize(
                _angular_objective_4ring,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            opt_pts = _generate_4ring_points(res.x)
            _update_best(opt_pts)
            
        except Exception:
            pass

    # Phase 2: Refine best in full 2D space
    if best_points is not None:
        if time.time() < deadline - 20:
            _try_optimize_2d(best_points, 'Nelder-Mead', maxiter=3000, ftol=1e-11, xtol=1e-9)
        
        if time.time() < deadline - 10:
            _try_optimize_2d(best_points, 'Powell', maxiter=300, ftol=1e-10, xtol=1e-8)
    
    # Fallback if 4-ring failed or didn't find anything
    if best_points is None:
        # Use regular 14-gon as fallback
        best_points = _normalize(_regular_polygon(N, 1.0, 0.0))
        _update_best(best_points)
        
        # Try 7-7 ring as well if time permits
        if time.time() < deadline - 20:
             # Quick 7-7 search
             equal_gap = 2 * np.pi / 7
             for r_inner in [0.5, 0.6, 0.7]:
                 if time.time() > deadline - 10: break
                 # This part is simplified, just using a few starts
                 init_pts = np.vstack([
                     _regular_polygon(7, 1.0, 0.0),
                     _regular_polygon(7, r_inner, 0.0)
                 ])
                 _try_optimize_2d(init_pts, 'Nelder-Mead', maxiter=1000, ftol=1e-10, xtol=1e-8)

    # Final normalization
    if best_points is not None:
        best_points = _normalize(best_points)
    
    return best_points

# EVOLVE-BLOCK-END

import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
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

def _triangle_areas(points):
    """Compute all triangle areas (positive) for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    """Compute minimum triangle area for given points (assumes valid shape)."""
    return _triangle_areas(points).min()

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _soft_min(x, beta=100.0):
    """Soft-min approximation using log-sum-exp."""
    x_min = x.min()
    shifted = x - x_min
    # Clip to avoid overflow
    shifted = np.clip(shifted, 0, 50.0 / beta)
    return x_min - np.log(np.sum(np.exp(-beta * shifted))) / beta

def _neg_smooth_score(points_flat, beta=100.0):
    """Negative smooth score for minimization - smooth surrogate for the nonsmooth min."""
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 1e6
        areas = _triangle_areas(points)
        soft_min_area = _soft_min(areas, beta)
        return -soft_min_area / ha
    except Exception:
        return 1e6

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _coordinate_descent_polish(points, max_iter=5000, tol=1e-10, deadline=None):
    """
    Coordinate descent focusing on the bottleneck triangle.
    At each step, identify the triangle with minimum area, then try to
    move each of its 3 vertices in small steps to increase that area
    while not decreasing the global minimum too much.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        areas = _triangle_areas(best_pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        i, j, k = _TRI_INDICES[min_idx]
        bottleneck_vertices = [i, j, k]
        
        improved = False
        for vi in bottleneck_vertices:
            # Try small perturbations of vertex vi
            base = best_pts[vi].copy()
            best_local = base.copy()
            best_local_score = best_score
            
            # Try different step sizes and directions
            for step in [0.05, 0.02, 0.01, 0.005, 0.002]:
                for dx in [-step, 0.0, step]:
                    for dy in [-step, 0.0, step]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        trial = best_pts.copy()
                        trial[vi] = base + np.array([dx, dy])
                        trial_score = _score(trial)
                        if trial_score > best_local_score + 1e-12:
                            best_local_score = trial_score
                            best_local = trial[vi].copy()
            
            if best_local_score > best_score + 1e-12:
                best_pts[vi] = best_local
                best_score = best_local_score
                improved = True
        
        if not improved:
            # Try slightly larger steps if no improvement
            for vi in bottleneck_vertices:
                base = best_pts[vi].copy()
                best_local = base.copy()
                best_local_score = best_score
                
                for step in [0.1, 0.2, 0.3]:
                    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
                    for a in angles:
                        dx = step * np.cos(a)
                        dy = step * np.sin(a)
                        trial = best_pts.copy()
                        trial[vi] = base + np.array([dx, dy])
                        trial_score = _score(trial)
                        if trial_score > best_local_score + 1e-12:
                            best_local_score = trial_score
                            best_local = trial[vi].copy()
                
                if best_local_score > best_score + 1e-12:
                    best_pts[vi] = best_local
                    best_score = best_local_score
                    improved = True
            
            if not improved:
                break
    
    return best_pts, best_score

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_circles(n, seed=42):
    """Points on two concentric circles."""
    rng = np.random.RandomState(seed)
    # 7 points on outer, 7 on inner
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(0, 2 * np.pi)
    inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(0, 2 * np.pi)
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return pts

def _random_in_disk(n, seed=None):
    """Random points in unit disk."""
    rng = np.random.RandomState(seed)
    r = np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def _perturbed_polygon(n, base_radius=1.0, noise_scale=0.05, seed=None, rotation=0.0):
    """Perturbed regular polygon."""
    rng = np.random.RandomState(seed)
    base = _regular_polygon(n, base_radius, rotation)
    noise = rng.uniform(-noise_scale, noise_scale, (n, 2))
    return base + noise

def _points_from_16d_params(params):
    """
    Map 16 parameters to 14 points.
    Params:
    0: R_out
    1: R_in
    2-8: 7 angular offsets for outer ring (relative to regular spacing)
    9-15: 7 angular offsets for inner ring (relative to regular spacing)
    """
    R_out = params[0]
    R_in = params[1]
    
    # Outer ring: 7 points with individual angular perturbations
    base_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_out = base_angles + params[2:9]
    
    x_out = R_out * np.cos(angles_out)
    y_out = R_out * np.sin(angles_out)
    
    # Inner ring: 7 points with individual angular perturbations
    angles_in = base_angles + params[9:16]
    
    x_in = R_in * np.cos(angles_in)
    y_in = R_in * np.sin(angles_in)
    
    pts = np.vstack([np.column_stack([x_out, y_out]), np.column_stack([x_in, y_in])])
    return pts

def _neg_score_16d(params):
    """Negative score for 16D parameter space."""
    try:
        pts = _points_from_16d_params(params)
        return -_score(pts)
    except Exception:
        return 1e6

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    
    # 1. Differential Evolution on 16D space (2 radii + 14 angles)
    bounds = [
        (0.5, 2.0),  # R_out
        (0.1, 1.5),  # R_in
    ]
    # 7 angle offsets for outer ring
    bounds += [(-0.8, 0.8)] * 7
    # 7 angle offsets for inner ring
    bounds += [(-0.8, 0.8)] * 7
    
    try:
        res = scipy.optimize.differential_evolution(
            _neg_score_16d, 
            bounds, 
            seed=42, 
            maxiter=150, 
            popsize=15, 
            tol=1e-8,
            polish=False,
            mutation=(0.3, 0.8),
            recombination=0.7
        )
        
        best_params = res.x
        pts = _points_from_16d_params(best_params)
        pts = _normalize(pts)
        sc = _score(pts)
        
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
            
    except Exception:
        pass
    
    # 1b. Second DE run with different seed if time permits
    if time.time() < deadline - 40:
        try:
            res2 = scipy.optimize.differential_evolution(
                _neg_score_16d, 
                bounds, 
                seed=123, 
                maxiter=100, 
                popsize=12, 
                tol=1e-8,
                polish=False,
                mutation=(0.4, 0.9),
                recombination=0.6
            )
            pts2 = _points_from_16d_params(res2.x)
            pts2 = _normalize(pts2)
            sc2 = _score(pts2)
            if sc2 > best_score:
                best_score = sc2
                best_points = pts2.copy()
        except Exception:
            pass
    
    # 2. Local refinement with Nelder-Mead on full 28D from best found
    if best_points is not None and time.time() < deadline - 15:
        try:
            res = scipy.optimize.minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt_pts)
            if sc > best_score:
                best_score = sc
                best_points = opt_pts.copy()
        except Exception:
            pass

    # 3. Polish the best solution
    if best_points is not None and time.time() < deadline - 5:
        polished, pol_score = _coordinate_descent_polish(best_points, max_iter=3000, deadline=deadline - 2)
        polished = _normalize(polished)
        final_score = _score(polished)
        if final_score > best_score:
            best_score = final_score
            best_points = polished.copy()
        
        # 4. Another round of local refinement after polishing
        if time.time() < deadline - 3:
            try:
                res2 = scipy.optimize.minimize(
                    _neg_score,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
                )
                opt_pts = _normalize(res2.x.reshape(-1, 2))
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_points = opt_pts.copy()
            except Exception:
                pass

    # Final fallback
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
    
    return best_points

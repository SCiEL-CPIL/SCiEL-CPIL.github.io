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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    
    def _try_optimize_smooth(init_points, maxiter=2000, ftol=1e-10, xtol=1e-8, beta=100.0):
        """Try optimizing with smooth surrogate from init_points."""
        nonlocal best_score, best_points
        if time.time() > deadline:
            return None
        try:
            init_norm = _normalize(init_points)
            
            res = scipy.optimize.minimize(
                _neg_smooth_score,
                init_norm.flatten(),
                method='Nelder-Mead',
                args=(beta,),
                options={'maxiter': maxiter, 'xatol': xtol, 'fatol': ftol, 'adaptive': True}
            )
            
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt_pts)
            
            if sc > best_score:
                best_score = sc
                best_points = opt_pts.copy()
            
            return (sc, opt_pts)
        except Exception:
            return None
    
    def _try_optimize(init_points, method='Nelder-Mead', maxiter=2000, ftol=1e-10, xtol=1e-8):
        """Try optimizing from init_points, return (score, points) or None."""
        nonlocal best_score, best_points
        if time.time() > deadline:
            return None
        try:
            init_norm = _normalize(init_points)
            init_score = _score(init_norm)
            
            if method == 'Nelder-Mead':
                res = scipy.optimize.minimize(
                    _neg_score,
                    init_norm.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': xtol, 'fatol': ftol, 'adaptive': True}
                )
            elif method == 'Powell':
                res = scipy.optimize.minimize(
                    _neg_score,
                    init_norm.flatten(),
                    method='Powell',
                    options={'maxiter': maxiter, 'ftol': ftol, 'xtol': xtol}
                )
            else:
                return None
            
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt_pts)
            
            if sc > best_score:
                best_score = sc
                best_points = opt_pts.copy()
            
            return (sc, opt_pts)
        except Exception:
            return None
    
    def _polish_and_update(points):
        """Apply coordinate descent polish and update best if improved."""
        nonlocal best_score, best_points
        if time.time() > deadline - 2:
            return
        try:
            polished, pol_score = _coordinate_descent_polish(points, max_iter=2000, deadline=deadline - 2)
            polished = _normalize(polished)
            final_score = _score(polished)
            if final_score > best_score:
                best_score = final_score
                best_points = polished.copy()
        except Exception:
            pass
    
    # Generate diverse initializations
    inits = []
    
    # 1. Regular polygon with various rotations
    for rot in [0.0, np.pi / 14, np.pi / 7, np.pi / 3]:
        inits.append(_regular_polygon(N, 1.0, rot))
    
    # 2. Perturbed regular polygons with different noise levels and seeds
    for seed in range(10):
        for ns in [0.03, 0.08, 0.15]:
            inits.append(_perturbed_polygon(N, 1.0, ns, seed=seed + 100, rotation=seed * 0.1))
    
    # 3. Concentric circles
    for seed in range(5):
        inits.append(_concentric_circles(N, seed=seed))
    
    # 4. Random in disk
    for seed in range(5):
        inits.append(_random_in_disk(N, seed=seed))
    
    # 5. Points on two circles with different radii ratios
    for ratio in [0.3, 0.4, 0.6, 0.7]:
        n_outer = 8
        n_inner = 6
        outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
        outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inner = np.column_stack([ratio * np.cos(inner_angles), ratio * np.sin(inner_angles)])
        inits.append(np.vstack([outer, inner]))
    
    # Sort inits by initial score (best first) to prioritize good starts
    scored_inits = []
    for init in inits:
        try:
            init_norm = _normalize(init)
            sc = _score(init_norm)
            scored_inits.append((sc, init_norm))
        except Exception:
            continue
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Update best with best initial
    if scored_inits:
        best_init_score, best_init_pts = scored_inits[0]
        if best_init_score > best_score:
            best_score = best_init_score
            best_points = best_init_pts.copy()
    
    # Phase 1: Optimize from top initializations with both smooth and exact objectives
    for rank, (init_score, init_pts) in enumerate(scored_inits):
        if time.time() > deadline - 30:
            break
        
        # Try smooth surrogate optimization (better gradient flow)
        time_left = deadline - time.time()
        if time_left > 10:
            _try_optimize_smooth(init_pts, maxiter=1500, ftol=1e-10, xtol=1e-8, beta=100.0)
        
        # Try exact Nelder-Mead
        time_left = deadline - time.time()
        if time_left > 8:
            _try_optimize(init_pts, 'Nelder-Mead', maxiter=2000, ftol=1e-11, xtol=1e-9)
        
        # Try Powell for top ranks
        time_left = deadline - time.time()
        if time_left > 10 and rank < 10:
            _try_optimize(init_pts, 'Powell', maxiter=300, ftol=1e-10, xtol=1e-8)
    
    # Phase 2: Polish the best solution with coordinate descent
    if best_points is not None:
        _polish_and_update(best_points)
    
    # Phase 3: More random restarts with smooth optimization
    rng = np.random.RandomState(12345)
    while time.time() < deadline - 20:
        seed = rng.randint(0, 100000)
        init = _perturbed_polygon(N, 1.0, 0.1, seed=seed, rotation=seed * 0.01)
        _try_optimize_smooth(init, maxiter=1000, ftol=1e-10, xtol=1e-8, beta=150.0)
        
        # Occasionally try exact
        if time.time() < deadline - 25:
            _try_optimize(init, 'Nelder-Mead', maxiter=1500, ftol=1e-11, xtol=1e-9)
    
    # Phase 4: Final polish
    if best_points is not None:
        _polish_and_update(best_points)
    
    # Final fallback
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
    
    return best_points

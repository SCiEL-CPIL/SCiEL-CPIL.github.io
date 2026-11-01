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

def _min_triangle_area(points):
    """Compute minimum triangle area for given points (assumes valid shape)."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

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

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _smooth_min_triangle_area(points, beta=50.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Returns a smooth approximation to min(areas) where areas are the 364 triangle areas.
    Higher beta = closer to true min.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # log-sum-exp for min: min(x) = -logsumexp(-x)
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    if not np.isfinite(max_neg):
        return 0.0
    logsumexp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    return -logsumexp

def _smooth_neg_score(points_flat, beta=50.0):
    """
    Smooth negative score: -(smooth_min_area / hull_area).
    Uses log-sum-exp with temperature beta for smoothness.
    """
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        # Use scaled log-sum-exp: min approximation with softness
        pi = points[_I]
        pj = points[_J]
        pk = points[_K]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Soft-min via log-sum-exp with temperature
        # As beta -> inf, this approaches true min
        neg_scaled = -beta * areas
        max_neg = np.max(neg_scaled)
        if not np.isfinite(max_neg):
            return 0.0
        logsumexp = max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))
        smooth_min = -logsumexp / beta
        
        return -(smooth_min / ha)
    except Exception:
        return 0.0

def _smooth_neg_score_grad(points_flat, beta=50.0):
    """
    Smooth negative score with numerical gradient (for BFGS/L-BFGS-B).
    """
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0, None
        pi = points[_I]
        pj = points[_J]
        pk = points[_K]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        neg_scaled = -beta * areas
        max_neg = np.max(neg_scaled)
        if not np.isfinite(max_neg):
            return 0.0, None
        logsumexp = max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))
        smooth_min = -logsumexp / beta
        
        return -(smooth_min / ha), None
    except Exception:
        return 0.0, None

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

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
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
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
            elif method == 'L-BFGS-B-smooth':
                # Use smooth surrogate with gradient-based optimizer
                # Start with moderate beta, increase for polish
                best_local = None
                best_local_score = 0.0
                
                for beta in [20.0, 50.0, 100.0, 200.0]:
                    if time.time() > deadline:
                        break
                    res = scipy.optimize.minimize(
                        lambda x: _smooth_neg_score(x, beta=beta)[0],
                        init_norm.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': max(500, maxiter // 4), 'ftol': ftol, 'gtol': 1e-6}
                    )
                    opt_pts = _normalize(res.x.reshape(-1, 2))
                    sc = _score(opt_pts)
                    if sc > best_local_score:
                        best_local_score = sc
                        best_local = opt_pts.copy()
                
                if best_local is not None:
                    opt_pts = best_local
                    sc = best_local_score
                else:
                    return None
                    
            elif method == 'BFGS-smooth':
                # BFGS with smooth surrogate, progressively increasing beta
                best_local = None
                best_local_score = 0.0
                
                for beta in [30.0, 80.0, 150.0, 300.0]:
                    if time.time() > deadline:
                        break
                    res = scipy.optimize.minimize(
                        lambda x: _smooth_neg_score(x, beta=beta)[0],
                        init_norm.flatten(),
                        method='BFGS',
                        options={'maxiter': max(300, maxiter // 5), 'gtol': 1e-5}
                    )
                    opt_pts = _normalize(res.x.reshape(-1, 2))
                    sc = _score(opt_pts)
                    if sc > best_local_score:
                        best_local_score = sc
                        best_local = opt_pts.copy()
                
                if best_local is not None:
                    opt_pts = best_local
                    sc = best_local_score
                else:
                    return None
            else:
                return None
            
            opt_pts = _normalize(opt_pts)
            sc = _score(opt_pts)
            
            if sc > best_score:
                best_score = sc
                best_points = opt_pts.copy()
            
            return (sc, opt_pts)
        except Exception:
            return None
    
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
    
    # Optimize from top initializations
    for rank, (init_score, init_pts) in enumerate(scored_inits):
        if time.time() > deadline:
            break
        
        # Try smooth gradient-based methods first (L-BFGS-B with log-sum-exp surrogate)
        time_left = deadline - time.time()
        if time_left > 15:
            _try_optimize(init_pts, 'L-BFGS-B-smooth', maxiter=2000, ftol=1e-12, xtol=1e-10)
        
        # Try Nelder-Mead (faster per iteration, good for low-dim)
        time_left = deadline - time.time()
        if time_left > 8:
            _try_optimize(init_pts, 'Nelder-Mead', maxiter=2000, ftol=1e-11, xtol=1e-9)
        
        # Try Powell if time permits
        time_left = deadline - time.time()
        if time_left > 10 and rank < 10:
            _try_optimize(init_pts, 'Powell', maxiter=400, ftol=1e-10, xtol=1e-8)
        
        # Try BFGS smooth for top few initializations
        time_left = deadline - time.time()
        if time_left > 20 and rank < 5:
            _try_optimize(init_pts, 'BFGS-smooth', maxiter=1500, ftol=1e-12, xtol=1e-10)
    
    # If we still have time, do a few more random restarts with smooth optimization
    while time.time() < deadline - 15:
        seed = int(time.time() * 1000) % 100000
        init = _perturbed_polygon(N, 1.0, 0.08, seed=seed, rotation=seed * 0.01)
        # Use smooth method for random restarts (more robust)
        _try_optimize(init, 'L-BFGS-B-smooth', maxiter=1500, ftol=1e-12, xtol=1e-10)
        
        # Also try Nelder-Mead from the same init
        time_left = deadline - time.time()
        if time_left > 10:
            _try_optimize(init, 'Nelder-Mead', maxiter=1500, ftol=1e-11, xtol=1e-9)
    
    # Final polish: if we have time, do a high-precision Nelder-Mead polish on the best point
    if best_points is not None and time.time() < deadline - 5:
        try:
            final_res = scipy.optimize.minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            final_pts = _normalize(final_res.x.reshape(-1, 2))
            final_sc = _score(final_pts)
            if final_sc > best_score:
                best_score = final_sc
                best_points = final_pts.copy()
        except Exception:
            pass
    
    # Final fallback
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
    
    return best_points

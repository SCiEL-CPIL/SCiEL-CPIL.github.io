import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 140.0  # wall-clock guard

def _triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=np.int32)

_TRI = _triangle_indices(N)
_I = _TRI[:, 0]
_J = _TRI[:, 1]
_K = _TRI[:, 2]
_NTRI = _TRI.shape[0]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        return float(scipy.spatial.ConvexHull(points).volume)
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        return _min_triangle_area(points) / ha
    except Exception:
        return 0.0

def _neg_score(flat):
    return -_score(flat.reshape(-1, 2))

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])

def _two_heptagons(r_inner, phi):
    """Two nested regular heptagons: outer at radius 1, inner at r_inner rotated by phi."""
    outer = _regular_polygon(7, 1.0, 0.0)
    inner = _regular_polygon(7, r_inner, phi)
    return np.vstack([outer, inner])

def _ellipse_config(n, a=1.0, b=0.8, rotation=0.0):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    pts = np.column_stack([x, y])
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T

def _build_diverse_population():
    """
    Generate a diverse set of initial points for Differential Evolution.
    We use structured configurations to guide the global search.
    """
    starts = []
    
    # 1. Regular polygon variations
    for rot in np.linspace(0, 2*np.pi, 10, endpoint=False):
        starts.append(_regular_polygon(N, 1.0, rot))
        
    # 2. Two heptagons variations
    for r_inner in np.linspace(0.2, 0.9, 10):
        for phi in np.linspace(0, 2*np.pi, 10, endpoint=False):
            starts.append(_two_heptagons(r_inner, phi))
            
    # 3. Elliptical variations
    for a in [0.8, 1.0, 1.2]:
        for b in [0.6, 0.8, 1.0]:
            for rot in np.linspace(0, np.pi/2, 5, endpoint=False):
                starts.append(_ellipse_config(N, a, b, rot))

    # 4. Random perturbations of the best structured starts
    # We'll just take the first 50 unique-ish ones
    unique_starts = []
    seen = set()
    for s in starts:
        # Simple hash based on rounded coords
        key = tuple(np.round(s, 3).flatten().tolist())
        if key not in seen:
            seen.add(key)
            unique_starts.append(s)
            
    # Ensure we have enough for DE population (DE usually wants ~10-15 * N_params)
    # N_params = 28. So ~280-420 points is ideal, but we are limited by time.
    # Let's aim for ~100-200 initial points.
    if len(unique_starts) < 100:
        # Pad with random noise if needed
        while len(unique_starts) < 100:
            rand_pts = np.random.uniform(-1, 1, (N, 2))
            unique_starts.append(rand_pts)
            
    return np.array(unique_starts[:200]) # Cap at 200 for memory/time

def _lse_min_area(points, tau=0.01):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_a = np.min(areas)
    # Stable log-sum-exp for soft min
    # soft_min = -tau * log(sum(exp(-areas/tau)))
    # To avoid underflow/overflow, shift by min
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    lse_val = -tau * (max_scaled + np.log(sum_exp))
    return lse_val

def _smooth_neg_score(flat, tau=0.01):
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    min_a_soft = _lse_min_area(pts, tau)
    return -min_a_soft / ha

def construct():
    t0 = time.time()
    def left():
        return DEADLINE - (time.time() - t0)

    # Phase 0: Generate diverse initial population
    if left() <= 5.0:
        return _normalize(_regular_polygon(N, 1.0, 0.0))
    
    init_points = _build_diverse_population()
    # Flatten for DE
    init_flat = init_points.reshape(init_points.shape[0], -1)
    
    # Phase 1: Differential Evolution Global Search
    # DE is robust to non-smoothness. We use the raw score (negated) here.
    # Bounds: [-1, 1] for each coordinate. Normalization happens after.
    bounds = [(-1.0, 1.0)] * (N * 2)
    
    best_score = 0.0
    best_pts = _normalize(_regular_polygon(N, 1.0, 0.0))
    
    if left() > 30.0:
        try:
            # maxiter is the number of generations. 
            # With popsize ~100, and 28 dims, function evaluations can be high.
            # We limit maxfev to ensure we finish in time.
            # 150s / 0.01s per eval (approx) = 15000 evals. 
            # Let's be conservative: 5000 evals.
            de_res = scipy.optimize.differential_evolution(
                _neg_score, 
                bounds, 
                init=init_flat, 
                maxiter=50, 
                maxfev=5000, 
                tol=1e-5, 
                seed=42,
                polish=False  # We will do our own polish
            )
            
            if de_res.success or de_res.fun < 0:
                opt_pts = _normalize(de_res.x.reshape(-1, 2))
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score, best_pts = sc, opt_pts.copy()
        except Exception:
            pass

    # Phase 2: Local Refinement with Smooth Surrogate (L-BFGS-B)
    # Use the best point from DE as start.
    if left() > 20.0:
        # Coarse to fine tau annealing
        for tau in [0.1, 0.05, 0.02, 0.01]:
            if left() <= 5.0:
                break
            try:
                obj_func = lambda f: _smooth_neg_score(f, tau)
                opts = {
                    'maxiter': 500, 
                    'fatol': 1e-12, 
                    'xatol': 1e-12,
                    'adaptive': True, 
                    'maxfev': 2000
                }
                res = scipy.optimize.minimize(
                    obj_func, 
                    best_pts.flatten(), 
                    method='L-BFGS-B', 
                    options=opts
                )
                opt_pts = _normalize(res.x.reshape(-1, 2))
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score, best_pts = sc, opt_pts.copy()
            except Exception:
                pass

    # Phase 3: Final Polish with Nelder-Mead on Raw Score
    if left() > 10.0:
        try:
            opts = {
                'maxiter': 10000, 
                'fatol': 1e-14, 
                'xatol': 1e-14,
                'adaptive': True
            }
            res = scipy.optimize.minimize(
                _neg_score, 
                best_pts.flatten(), 
                method='Nelder-Mead', 
                options=opts
            )
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt_pts)
            if sc > best_score:
                best_score, best_pts = sc, opt_pts.copy()
        except Exception:
            pass

    # Phase 4: One last quick check with a different local optimizer (Powell) if time permits
    if left() > 5.0:
        try:
            opts = {
                'maxiter': 2000, 
                'fatol': 1e-14, 
                'xatol': 1e-14
            }
            res = scipy.optimize.minimize(
                _neg_score, 
                best_pts.flatten(), 
                method='Powell', 
                options=opts
            )
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt_pts)
            if sc > best_score:
                best_score, best_pts = sc, opt_pts.copy()
        except Exception:
            pass

    return _normalize(best_pts)

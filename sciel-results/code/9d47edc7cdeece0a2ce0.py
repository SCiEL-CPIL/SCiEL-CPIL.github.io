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

def _generate_symmetric_inits():
    """
    Generate initializations based on symmetric subgroup hypotheses:
    1. 7+7: Two concentric regular heptagons.
    2. 7+1+6: Outer heptagon, center point, inner hexagon.
    3. 7+1+1+5: Outer heptagon, center, another point, inner pentagon (less likely but diverse).
    4. Perturbed versions of these.
    """
    inits = []
    
    # Hypothesis 1: 7 outer + 7 inner (Heptagon symmetry)
    # Parameters: r_out, r_in, phase_out, phase_in
    for r_out in [1.0, 1.2, 0.8]:
        for r_in in [0.3, 0.5, 0.7]:
            for phase_out in [0.0, np.pi/7]:
                for phase_in in [0.0, np.pi/7, 2*np.pi/7]:
                    p_out = _regular_polygon(7, r_out, phase_out)
                    p_in = _regular_polygon(7, r_in, phase_in)
                    pts = np.vstack([p_out, p_in])
                    inits.append(pts)
                    
                    # Add slight perturbations to break perfect symmetry and help escape local minima
                    rng = np.random.RandomState(hash((r_out, r_in, phase_out, phase_in)) % 10000)
                    for noise in [0.01, 0.05]:
                        pts_pert = pts + rng.uniform(-noise, noise, (14, 2))
                        inits.append(pts_pert)

    # Hypothesis 2: 7 outer + 1 center + 6 inner (Hexagon inside Heptagon)
    # This breaks the D7 symmetry of the inner part but keeps the outer D7.
    for r_out in [1.0, 1.1]:
        for r_in in [0.4, 0.6]:
            p_out = _regular_polygon(7, r_out, 0.0)
            p_center = np.array([[0.0, 0.0]])
            # Inner hexagon, rotated to align with heptagon symmetry axes as best as possible
            p_in = _regular_polygon(6, r_in, np.pi/6) 
            pts = np.vstack([p_out, p_center, p_in])
            inits.append(pts)
            
            # Perturbed
            rng = np.random.RandomState(123)
            for noise in [0.02, 0.05]:
                pts_pert = pts + rng.uniform(-noise, noise, (14, 2))
                inits.append(pts_pert)

    # Hypothesis 3: 7 outer + 7 inner, but inner rotated to maximize min area
    # We will let the optimizer find the best phase, but provide a few guesses
    for r_out in [1.0]:
        for r_in in [0.4, 0.5, 0.6]:
            for phase in np.linspace(0, 2*np.pi, 7, endpoint=False):
                p_out = _regular_polygon(7, r_out, 0.0)
                p_in = _regular_polygon(7, r_in, phase)
                pts = np.vstack([p_out, p_in])
                inits.append(pts)

    # Hypothesis 4: Regular 14-gon (baseline)
    inits.append(_regular_polygon(14, 1.0, 0.0))
    
    # Hypothesis 5: Two rings 8+6 (Octagon + Hexagon) - less symmetric but possible
    for r_out in [1.0]:
        for r_in in [0.5, 0.7]:
            p_out = _regular_polygon(8, r_out, 0.0)
            p_in = _regular_polygon(6, r_in, np.pi/6)
            pts = np.vstack([p_out, p_in])
            inits.append(pts)

    return inits

def _optimize_ring_params(params, ring_type):
    """
    Optimize radii and phases for a specific ring structure.
    This is a lower-dimensional search to find good basins for the 28D optimizer.
    """
    def neg_ring_score(p):
        try:
            if ring_type == '7_7':
                r_out, r_in, ph_out, ph_in = p
                p_out = _regular_polygon(7, r_out, ph_out)
                p_in = _regular_polygon(7, r_in, ph_in)
                pts = np.vstack([p_out, p_in])
            elif ring_type == '7_1_6':
                r_out, r_in, ph_in = p
                p_out = _regular_polygon(7, r_out, 0.0)
                p_center = np.array([[0.0, 0.0]])
                p_in = _regular_polygon(6, r_in, ph_in)
                pts = np.vstack([p_out, p_center, p_in])
            else:
                return 0.0
            
            pts_norm = _normalize(pts)
            return -_score(pts_norm)
        except Exception:
            return 0.0

    # Bounds: radii > 0, phases in [0, 2pi]
    if ring_type == '7_7':
        bounds = [(0.2, 2.0), (0.1, 1.5), (0, 2*np.pi), (0, 2*np.pi)]
        x0 = [1.0, 0.5, 0.0, np.pi/7]
    elif ring_type == '7_1_6':
        bounds = [(0.2, 2.0), (0.1, 1.5), (0, 2*np.pi)]
        x0 = [1.0, 0.5, np.pi/6]
    else:
        return None

    try:
        res = scipy.optimize.minimize(
            neg_ring_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        best_params = res.x
        # Reconstruct points
        if ring_type == '7_7':
            r_out, r_in, ph_out, ph_in = best_params
            p_out = _regular_polygon(7, r_out, ph_out)
            p_in = _regular_polygon(7, r_in, ph_in)
            pts = np.vstack([p_out, p_in])
        else:
            r_out, r_in, ph_in = best_params
            p_out = _regular_polygon(7, r_out, 0.0)
            p_center = np.array([[0.0, 0.0]])
            p_in = _regular_polygon(6, r_in, ph_in)
            pts = np.vstack([p_out, p_center, p_in])
        
        return _normalize(pts)
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
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
            
            if init_score > best_score:
                best_score = init_score
                best_points = init_norm.copy()
            
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

    # Phase 1: Optimize specific symmetric ring structures in low-dim space
    # This quickly finds good basins for the 7+7 and 7+1+6 configurations
    
    ring_7_7_pts = _optimize_ring_params(None, '7_7')
    if ring_7_7_pts is not None:
        _try_optimize(ring_7_7_pts, 'Nelder-Mead', maxiter=1000)
        
    ring_7_1_6_pts = _optimize_ring_params(None, '7_1_6')
    if ring_7_1_6_pts is not None:
        _try_optimize(ring_7_1_6_pts, 'Nelder-Mead', maxiter=1000)

    # Phase 2: Generate diverse initializations and sort by score
    inits = _generate_symmetric_inits()
    
    # Add some random perturbations of the best found so far
    if best_points is not None:
        rng = np.random.RandomState(42)
        for i in range(5):
            noise = rng.uniform(-0.05, 0.05, (N, 2))
            inits.append(best_points + noise)

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
    
    # Phase 3: Optimize from top initializations in full 28D space
    # Prioritize the symmetric ones (which are likely at the top)
    for rank, (init_score, init_pts) in enumerate(scored_inits):
        if time.time() > deadline:
            break
        
        # Skip if this init is too similar to best_points (avoid redundant work)
        if rank > 0:
            diff = np.linalg.norm(init_pts - best_points)
            if diff < 1e-4:
                continue

        # Try Nelder-Mead first (robust for this problem size)
        time_left = deadline - time.time()
        if time_left > 5:
            _try_optimize(init_pts, 'Nelder-Mead', maxiter=2000, ftol=1e-11, xtol=1e-9)
        
        # Try Powell if time permits and we are in a promising region
        time_left = deadline - time.time()
        if time_left > 10 and rank < 10:
            _try_optimize(init_pts, 'Powell', maxiter=500, ftol=1e-10, xtol=1e-8)
        
        # Early stop if we've spent too much time on one init
        if time.time() - start_time > 120:
            break

    # Phase 4: Final polish with tighter tolerances on the best point
    if best_points is not None and time.time() < deadline - 5:
        _try_optimize(best_points, 'Nelder-Mead', maxiter=3000, ftol=1e-12, xtol=1e-10)
        _try_optimize(best_points, 'Powell', maxiter=1000, ftol=1e-12, xtol=1e-10)

    # Final fallback
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
    
    return best_points

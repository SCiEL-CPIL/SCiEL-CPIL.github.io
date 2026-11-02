import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

N = 14
DEADLINE = 140.0  # Wall-clock guard (evaluator kills at 150s, 10s margin)

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
    """
    Calculate the minimum area of all triangles formed by the points.
    points: (N, 2) array
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    """
    Score = min_triangle_area / hull_area
    """
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

def _neg_score(flat):
    """
    Negative score for minimization algorithms.
    """
    points = flat.reshape(-1, 2)
    return -_score(points)

def _normalize(points):
    """
    Normalize points such that the convex hull area is 1.
    """
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        # Scaling factor: Area scales with s^2. So s = 1/sqrt(ha)
        return points / np.sqrt(ha)
    except Exception:
        return points

def _generate_two_rings(n1, n2, r1, r2, phi1, phi2):
    """
    Generate points from two concentric regular polygons.
    """
    if n1 > 0:
        a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + phi1
        p1 = np.column_stack((r1 * np.cos(a1), r1 * np.sin(a1)))
    else:
        p1 = np.empty((0, 2))
    
    if n2 > 0:
        a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phi2
        p2 = np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2)))
    else:
        p2 = np.empty((0, 2))
        
    return np.vstack([p1, p2])

def _search_two_rings(n1, n2, time_limit):
    """
    Search for optimal parameters for a two-ring configuration.
    Returns best points and score.
    """
    best_score = -np.inf
    best_pts = None
    t0 = time.time()
    
    # Grid search over r2 and phi2 (relative to r1=1, phi1=0)
    # r2 in [0, 1], phi2 in [0, 2*pi/n2] (symmetry)
    n_r = 200
    n_phi = 100
    
    r_vals = np.linspace(0.0, 1.0, n_r)
    phi_vals = np.linspace(0.0, 2 * np.pi / max(1, n2), n_phi, endpoint=False)
    
    # We can vectorize the evaluation for speed
    # But for simplicity and robustness, loop with time check
    for r2 in r_vals:
        if time.time() - t0 > time_limit:
            break
        for phi2 in phi_vals:
            pts = _generate_two_rings(n1, n2, 1.0, r2, 0.0, phi2)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
                
    # Refine with golden section if time permits
    if best_pts is not None and time.time() - t0 < time_limit * 0.8:
        # Simple local refinement on r2 and phi2
        # Use Nelder-Mead on the 2 parameters
        def neg_ring_score(params):
            r2, phi2 = params
            if r2 < 0: r2 = 0
            if r2 > 1.5: r2 = 1.5
            pts = _generate_two_rings(n1, n2, 1.0, r2, 0.0, phi2)
            return -_score(pts)
            
        # Initialize from best grid point
        # We need to recover r2, phi2 from best_pts? 
        # It's easier to store them during grid search.
        # Let's restart the search with storage of params.
        pass # Skipping complex recovery, grid is dense enough.
        
    return best_pts, best_score

def _search_three_rings(n1, n2, n3, time_limit):
    """
    Search for optimal parameters for a three-ring configuration.
    """
    best_score = -np.inf
    best_pts = None
    t0 = time.time()
    
    # Coarser grid for 3 rings due to 3 params
    n_r = 50
    n_phi = 20
    
    r1 = 1.0
    r2_vals = np.linspace(0.3, 0.8, n_r)
    r3_vals = np.linspace(0.0, 0.5, n_r)
    phi2_vals = np.linspace(0.0, 2 * np.pi / max(1, n2), n_phi, endpoint=False)
    phi3_vals = np.linspace(0.0, 2 * np.pi / max(1, n3), n_phi, endpoint=False)
    
    for r2 in r2_vals:
        if time.time() - t0 > time_limit:
            break
        for r3 in r3_vals:
            if time.time() - t0 > time_limit:
                break
            for phi2 in phi2_vals:
                if time.time() - t0 > time_limit:
                    break
                for phi3 in phi3_vals:
                    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                    p1 = np.column_stack((r1 * np.cos(a1), r1 * np.sin(a1)))
                    
                    a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phi2
                    p2 = np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2)))
                    
                    a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + phi3
                    p3 = np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
                    
                    pts = np.vstack([p1, p2, p3])
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts.copy()
                        
    return best_pts, best_score

def construct():
    t0 = time.time()
    
    def left():
        return DEADLINE - (time.time() - t0)
    
    best_score = 0.0
    best_pts = None
    
    # Initialize with regular polygon
    a = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack((np.cos(a), np.sin(a)))
    best_pts = _normalize(init_pts)
    best_score = _score(best_pts)
    
    # ---- Phase 1: Symmetric Global Search ----
    # Try various topologies
    topologies = [
        (7, 7),
        (8, 6),
        (6, 8),
        (5, 9),
        (9, 5),
        (4, 10),
        (10, 4),
        (5, 5, 4),
        (4, 5, 5),
        (3, 5, 6),
        (6, 5, 3),
    ]
    
    for top in topologies:
        if left() < 10.0:
            break
            
        if len(top) == 2:
            n1, n2 = top
            pts, sc = _search_two_rings(n1, n2, time_limit=5.0)
        else:
            n1, n2, n3 = top
            pts, sc = _search_three_rings(n1, n2, n3, time_limit=5.0)
            
        if pts is not None and sc > best_score:
            best_score = sc
            best_pts = _normalize(pts)
            
    # ---- Phase 2: Local Refinement with Nelder-Mead ----
    # We will run NM from the best symmetric point and some jittered versions
    starts = [best_pts.copy()]
    
    # Add jittered versions
    rng = np.random.default_rng(42)
    for i in range(5):
        if left() < 20.0:
            break
        jitter = rng.normal(0, 0.05, (N, 2))
        start = _normalize(best_pts + jitter)
        starts.append(start)
        
    # Also add a few other symmetric configs that might be local optima
    # E.g. 7+7 with different rotation
    if left() > 30.0:
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_in = a_out + np.pi / 7 # Optimal rotation for 7+7 is often pi/7
        p_out = np.column_stack((np.cos(a_out), np.sin(a_out)))
        p_in = np.column_stack((0.5 * np.cos(a_in), 0.5 * np.sin(a_in)))
        alt_start = _normalize(np.vstack([p_out, p_in]))
        starts.append(alt_start)
        
    for start in starts:
        if left() < 15.0:
            break
            
        try:
            res = scipy.optimize.minimize(
                _neg_score, 
                start.flatten(), 
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'fatol': 1e-14,
                    'xatol': 1e-12,
                    'adaptive': True
                }
            )
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt_pts)
            if sc > best_score:
                best_score = sc
                best_pts = opt_pts.copy()
        except Exception:
            pass

    # ---- Phase 3: Final Polish ----
    # One more NM run with tighter tolerances on the best point
    if left() > 10.0:
        try:
            res = scipy.optimize.minimize(
                _neg_score, 
                best_pts.flatten(), 
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'fatol': 1e-15,
                    'xatol': 1e-13,
                    'adaptive': True
                }
            )
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt_pts)
            if sc > best_score:
                best_score = sc
                best_pts = opt_pts.copy()
        except Exception:
            pass

    return _normalize(best_pts)

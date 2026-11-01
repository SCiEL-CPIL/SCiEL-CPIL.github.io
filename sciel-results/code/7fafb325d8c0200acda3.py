import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

# Pre-compute triangle indices for speed
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
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
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

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2):
    """
    Generate points on two concentric circles.
    n1, n2: number of points on outer and inner ring.
    r1, r2: radii.
    rot1, rot2: rotations.
    """
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    return np.vstack([pts1, pts2])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_points = None

    def _update_best(points):
        nonlocal best_score, best_points
        if points is None:
            return 0.0
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    # Strategy: Focus on the "Two Nested Regular Polygons" family.
    # This reduces the 28D problem to a 3D problem (r_ratio, rot1, rot2) for each split (n1, n2).
    # We perform a coarse grid search followed by a fine Nelder-Mead polish on the top candidates.
    
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4), (3, 11), (11, 3), (2, 12), (12, 2), (1, 13), (13, 1)]
    
    top_candidates = [] # List of (score, points, params)
    
    for n_out, n_in in splits:
        if time.time() > deadline - 20:
            break
        
        # Define the parameter space for this split
        # r_ratio: inner/outer radius. Keep it between 0.2 and 0.9 to avoid degenerate hulls or points too close.
        # rot_out, rot_in: Rotations. Due to rotational symmetry of the rings, we can fix rot_out=0 
        # and only search rot_in relative to it? 
        # Actually, the relative rotation matters. Let's search rot_in in [0, 2*pi/n_in] and rot_out in [0, 2*pi/n_out]?
        # To be safe and simple, we search both, but use symmetry to reduce range if possible.
        # For regular polygons, rotating one by 2*pi/k is equivalent to identity if the other is fixed? No.
        # Let's just do a grid over relative rotations.
        
        # Grid Search
        r_ratios = np.linspace(0.3, 0.85, 12)
        rot_outs = np.linspace(0, 2*np.pi/n_out, int(n_out//2)+1, endpoint=False) # Symmetry reduction: half the circle is enough for even n? 
        # Actually, for a regular n-gon, rotation by 2*pi/n is identity. So range [0, 2*pi/n) covers all unique orientations.
        rot_in_range = np.linspace(0, 2*np.pi/n_in, int(n_in//2)+1, endpoint=False)
        
        best_local_score = -1
        best_local_params = None
        best_local_pts = None
        
        # Vectorized grid evaluation might be heavy if done all at once, but 12*4*4 = 192 points is small.
        # Let's iterate to keep memory low and allow early exit if time is tight.
        
        for r_ratio in r_ratios:
            if time.time() > deadline - 10:
                break
            for rot_out in rot_outs:
                for rot_in in rot_in_range:
                    pts = _generate_two_ring_config(n_out, n_in, 1.0, r_ratio, rot_out, rot_in)
                    sc = _score(pts)
                    if sc > best_local_score:
                        best_local_score = sc
                        best_local_params = (r_ratio, rot_out, rot_in)
                        best_local_pts = pts.copy()
        
        if best_local_score > 0:
            top_candidates.append((best_local_score, best_local_pts, best_local_params))

    # Sort candidates by score
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the top candidates with Nelder-Mead on the 3 parameters
    refined_configs = []
    
    for sc, pts, params in top_candidates[:5]: # Take top 5
        if time.time() > deadline - 15:
            break
            
        n_out, n_in = params[0], params[1] # Wait, params are (r, ro, ri). We need n_out, n_in from the loop context.
        # This is a bug in the previous logic structure. We need to associate the split with the candidate.
        
    # Let's restructure the refinement loop to know the split.
    
    # Re-doing the candidate collection to include split info
    top_candidates = []
    for n_out, n_in in splits:
        if time.time() > deadline - 20:
            break
            
        r_ratios = np.linspace(0.3, 0.85, 15)
        rot_outs = np.linspace(0, 2*np.pi/n_out, max(1, int(n_out//2)), endpoint=False)
        rot_in_range = np.linspace(0, 2*np.pi/n_in, max(1, int(n_in//2)), endpoint=False)
        
        best_local_score = -1
        best_local_params = None
        best_local_pts = None
        
        for r_ratio in r_ratios:
            if time.time() > deadline - 10:
                break
            for rot_out in rot_outs:
                for rot_in in rot_in_range:
                    pts = _generate_two_ring_config(n_out, n_in, 1.0, r_ratio, rot_out, rot_in)
                    sc = _score(pts)
                    if sc > best_local_score:
                        best_local_score = sc
                        best_local_params = (r_ratio, rot_out, rot_in)
                        best_local_pts = pts.copy()
        
        if best_local_score > 0:
            top_candidates.append({
                'score': best_local_score,
                'pts': best_local_pts,
                'params': best_local_params,
                'n_out': n_out,
                'n_in': n_in
            })

    top_candidates.sort(key=lambda x: x['score'], reverse=True)

    # Refine top 3
    for cand in top_candidates[:3]:
        if time.time() > deadline - 15:
            break
            
        n_out = cand['n_out']
        n_in = cand['n_in']
        init_params = cand['params']
        
        def _neg_ring_score(params, n_out_val=n_out, n_in_val=n_in):
            r_ratio, rot_out, rot_in = params
            if r_ratio < 0.1 or r_ratio > 0.95:
                return 1e6
            pts = _generate_two_ring_config(n_out_val, n_in_val, 1.0, r_ratio, rot_out, rot_in)
            return -_score(pts)
        
        try:
            res = scipy.optimize.minimize(
                _neg_ring_score,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            r_ratio, rot_out, rot_in = res.x
            pts = _generate_two_ring_config(n_out, n_in, 1.0, r_ratio, rot_out, rot_in)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_points = _normalize(pts).copy()
        except Exception:
            pass

    # Fallback: Regular 14-gon
    reg = _regular_polygon(N, 1.0)
    sc_reg = _update_best(reg)
    
    # If the two-ring search didn't find anything good, try a simple random restart with light optimization
    if best_score < 0.05:
        rng = np.random.RandomState(42)
        for i in range(5):
            if time.time() > deadline - 5:
                break
            r = np.sqrt(rng.uniform(0, 1, N))
            theta = rng.uniform(0, 2 * np.pi, N)
            init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            # Light Nelder-Mead on coordinates
            def _neg_coord(x):
                return -_score(x.reshape(-1, 2))
            
            try:
                res = scipy.optimize.minimize(
                    _neg_coord,
                    init.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-6}
                )
                _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points

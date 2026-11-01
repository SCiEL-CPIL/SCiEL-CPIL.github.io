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

def _smooth_min_triangle_area(points, beta=50.0):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    Returns a differentiable surrogate for the minimum triangle area.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp smoothing: -log(sum(exp(-beta * areas))) / beta approximates min(areas)
    # For stability, use the fact that min(x) = -1/beta * log(sum(exp(-beta*x)))
    # But we need to handle the case where areas can be very small
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0
    # Shift for numerical stability
    shifted = areas / max_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    smooth_min = -log_sum / beta * max_area
    return smooth_min

def _smooth_score(points, beta=50.0):
    """
    Smooth score: smooth_min_triangle_area / hull_area
    """
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        s_mta = _smooth_min_triangle_area(points, beta)
        return s_mta / ha
    except Exception:
        return 0.0

def _neg_smooth_score(points_flat, beta=50.0):
    points = points_flat.reshape(-1, 2)
    return -_smooth_score(points, beta)

def _neg_smooth_score_grad(points_flat, beta=50.0):
    """
    Gradient of -smooth_score. Uses finite differences for simplicity.
    Could be made analytic but finite diff is sufficient for L-BFGS-B.
    """
    # Use scipy's approximate gradient or just let L-BFGS-B handle it
    return None  # Let L-BFGS-B use its own gradient approximation

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _try_optimize_smooth(init_points, maxiter=5000, beta=50.0):
        """Optimize using smooth surrogate with L-BFGS-B."""
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        try:
            # Use L-BFGS-B with the smooth objective
            res = scipy.optimize.minimize(
                _neg_smooth_score,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            pts = res.x.reshape(-1, 2)
            _update_best(pts)
            
            # Polish with Nelder-Mead on the true (non-smooth) objective
            if time.time() < deadline - 5:
                def _neg_true_score(points_flat):
                    points = points_flat.reshape(-1, 2)
                    return -_score(points)
                
                res2 = scipy.optimize.minimize(
                    _neg_true_score,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                _update_best(res2.x.reshape(-1, 2))
                
        except Exception:
            pass

    def _try_optimize_points(init_points, maxiter=3000):
        """Optimize the 2N coordinates directly with Nelder-Mead."""
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        try:
            res = scipy.optimize.minimize(
                _neg_score,
                init_norm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            _update_best(res.x.reshape(-1, 2))
        except Exception:
            pass

    def _neg_score(points_flat):
        points = points_flat.reshape(-1, 2)
        return -_score(points)

    # 1. Explore the two-ring family analytically
    # Possible splits for N=14: (1,13), (2,12), (3,11), (4,10), (5,9), (6,8), (7,7)
    # (7,7) is symmetric and a very strong candidate.
    # (6,8) and (8,6) are also good.
    
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4)]
    
    for n1, n2 in splits:
        if time.time() > deadline:
            break
            
        # Optimize the parameters of the two-ring configuration
        # Params: r1 (normalized to 1), r2/r1, rot1, rot2
        # We fix r1=1 for optimization and scale at the end.
        
        def _two_ring_score(params, n1=n1, n2=n2):
            r_ratio = params[0]
            rot1 = params[1]
            rot2 = params[2]
            
            r1 = 1.0
            r2 = r_ratio * r1
            
            # Ensure r2 > 0
            if r2 <= 0.01:
                return 1e6
                
            pts = _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2)
            # Normalize and score
            sc = _score(_normalize(pts))
            return -sc

        def _two_ring_smooth_score(params, n1=n1, n2=n2, beta=50.0):
            r_ratio = params[0]
            rot1 = params[1]
            rot2 = params[2]
            
            r1 = 1.0
            r2 = r_ratio * r1
            
            # Ensure r2 > 0
            if r2 <= 0.01:
                return 1e6
                
            pts = _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2)
            # Normalize and smooth score
            sc = _smooth_score(_normalize(pts), beta)
            return -sc

        # Try a few different starting points for the ratio
        for r_init in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for rot_init in [0.0, np.pi/n1, np.pi/n2, 2*np.pi/n1, 2*np.pi/n2]:
                if time.time() > deadline:
                    continue
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    # First use smooth objective with L-BFGS-B for the 3 params
                    res = scipy.optimize.minimize(
                        _two_ring_smooth_score,
                        x0,
                        args=(n1, n2, 50.0),
                        method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _update_best(pts)
                    
                    # If this configuration is good, refine it by optimizing all 2N coordinates
                    if sc > best_score * 0.95 and time.time() < deadline - 15:
                        _try_optimize_smooth(pts, maxiter=3000, beta=50.0)
                        
                except Exception:
                    pass

    # 2. Explore regular polygon and slight perturbations
    # Regular 14-gon
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    # Perturb regular polygon and use smooth optimization
    rng = np.random.RandomState(42)
    for i in range(15):
        if time.time() > deadline - 10:
            break
        noise = rng.uniform(-0.1, 0.1, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        _try_optimize_smooth(init, maxiter=3000, beta=50.0)

    # 3. Random restarts with smooth optimization
    for i in range(8):
        if time.time() > deadline - 10:
            break
        # Generate points on a disk, then optimize
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        _try_optimize_smooth(init, maxiter=2000, beta=30.0)

    # 4. Final high-precision polish on the best solution
    if best_points is not None and time.time() < deadline - 3:
        # Try multiple beta values for the smooth objective, then polish
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 2:
                break
            try:
                res = scipy.optimize.minimize(
                    _neg_smooth_score,
                    best_points.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                pts = res.x.reshape(-1, 2)
                _update_best(pts)
            except Exception:
                pass
        
        # Final Nelder-Mead polish with very tight tolerances
        if time.time() < deadline - 1:
            try:
                res = scipy.optimize.minimize(
                    _neg_score,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points

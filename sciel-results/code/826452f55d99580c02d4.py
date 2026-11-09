"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _score_points(pts):
    """Compute Delta/n^n where Delta is product of all ordered-pair distances after scaling to diameter 2."""
    pts = np.array(pts, dtype=float)
    n = len(pts)
    # Compute all pairwise distances
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    # Diameter = max distance
    diam = d.max()
    if diam <= 1e-15:
        return 0.0
    # Scale to diameter 2
    scaled_d = d / (diam / 2.0)
    # Product over ordered pairs (i != j)
    # There are n*(n-1) ordered pairs
    # Use log to avoid underflow
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(scaled_d[mask]))
    # Delta = product of ordered pair distances = exp(log_prod)
    # score = Delta / n^n = exp(log_prod - n*log(n))
    log_score = log_prod - n * math.log(n)
    if log_score > 700:  # avoid overflow
        return float('inf')
    return math.exp(log_score)

def _soft_max(x, beta=50.0):
    """Smooth approximation to max using log-sum-exp."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _neg_log_score_smooth(x, n=8, beta=50.0):
    """Objective with smooth diameter via log-sum-exp."""
    pts = x.reshape(n, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    # Use smooth max for diameter
    mask = ~np.eye(n, dtype=bool)
    d_off = d[mask]
    if len(d_off) == 0:
        return 1e10
    diam = _soft_max(d_off, beta)
    if diam <= 1e-15:
        return 1e10
    scaled_d = d / (diam / 2.0)
    log_prod = np.sum(np.log(np.maximum(scaled_d[mask], 1e-15)))
    log_score = log_prod - n * math.log(n)
    return -log_score

def _neg_log_score_fixed_diam(x, n_var=6, n_total=8):
    """
    Objective where we fix points 0 and n_total-1 as diameter endpoints.
    x contains 2*n_var coordinates for the variable points.
    We ensure the diameter is determined by the fixed pair.
    """
    # Fixed points: (-1, 0) and (1, 0)
    fixed = np.array([[-1.0, 0.0], [1.0, 0.0]])
    var_pts = x.reshape(n_var, 2)
    all_pts = np.vstack([fixed, var_pts])
    n = len(all_pts)
    d = np.linalg.norm(all_pts[:, None, :] - all_pts[None, :, :], axis=2)
    diam = d.max()
    if diam <= 1e-15:
        return 1e10
    scaled_d = d / (diam / 2.0)
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(np.maximum(scaled_d[mask], 1e-15)))
    log_score = log_prod - n * math.log(n)
    return -log_score

def _neg_log_score_fixed_diam_smooth(x, n_var=6, n_total=8, beta=50.0):
    """
    Smooth version with fixed diameter endpoints.
    """
    fixed = np.array([[-1.0, 0.0], [1.0, 0.0]])
    var_pts = x.reshape(n_var, 2)
    all_pts = np.vstack([fixed, var_pts])
    n = len(all_pts)
    d = np.linalg.norm(all_pts[:, None, :] - all_pts[None, :, :], axis=2)
    mask = ~np.eye(n, dtype=bool)
    d_off = d[mask]
    diam = _soft_max(d_off, beta)
    if diam <= 1e-15:
        return 1e10
    scaled_d = d / (diam / 2.0)
    log_prod = np.sum(np.log(np.maximum(scaled_d[mask], 1e-15)))
    log_score = log_prod - n * math.log(n)
    return -log_score

def _make_fixed_diam_pts(x, n_var=6):
    """Reconstruct full point set from variable params with fixed diameter endpoints."""
    fixed = np.array([[-1.0, 0.0], [1.0, 0.0]])
    var_pts = x.reshape(n_var, 2)
    return np.vstack([fixed, var_pts])

def _initial_fixed_diam_params():
    """
    Initial guess for the 6 variable points, based on regular octagon
    with points 0 and 4 (antipodal) as diameter endpoints.
    Regular octagon angles: 0, 45, 90, 135, 180, 225, 270, 315
    If we put diameter endpoints at (-1,0) and (1,0), those correspond to angles 180 and 0.
    Remaining 6 points at angles 45, 90, 135, 225, 270, 315.
    """
    angles = [math.pi/4, math.pi/2, 3*math.pi/4, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
    pts = [[math.cos(a), math.sin(a)] for a in angles]
    return np.array(pts, dtype=float).flatten()

def construct():
    n = 8
    # Start from regular octagon
    angles = [2 * math.pi * k / n for k in range(n)]
    reg_pts = [[math.cos(a), math.sin(a)] for a in angles]
    best_score = _score_points(reg_pts)
    best_pts = reg_pts
    
    # Approach A: Fixed diameter endpoints optimization (6 variable points)
    x0_fixed = _initial_fixed_diam_params()
    
    # First try with smooth diameter
    try:
        res = minimize(_neg_log_score_fixed_diam_smooth, x0_fixed, 
                      method='L-BFGS-B', options={'maxiter': 5000, 'gtol': 1e-14, 'maxfun': 100000})
        if res.success and time.time() < _DEADLINE - 10:
            cand_pts = _make_fixed_diam_pts(res.x)
            d_min = np.min(np.linalg.norm(cand_pts[:, None, :] - cand_pts[None, :, :], axis=2) * (1 - np.eye(n)))
            if d_min > 1e-8:
                cand_score = _score_points(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts.tolist()
    except Exception:
        pass
    
    # Then try with exact diameter (nonsmooth but can polish)
    try:
        res = minimize(_neg_log_score_fixed_diam, x0_fixed,
                      method='L-BFGS-B', options={'maxiter': 10000, 'gtol': 1e-15, 'maxfun': 200000})
        if res.success and time.time() < _DEADLINE - 10:
            cand_pts = _make_fixed_diam_pts(res.x)
            d_min = np.min(np.linalg.norm(cand_pts[:, None, :] - cand_pts[None, :, :], axis=2) * (1 - np.eye(n)))
            if d_min > 1e-8:
                cand_score = _score_points(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts.tolist()
    except Exception:
        pass
    
    # Multi-start with fixed diameter endpoints
    rng = np.random.RandomState(42)
    n_starts = 30
    for start_i in range(n_starts):
        if time.time() > _DEADLINE - 10:
            break
        if start_i == 0:
            x_start = x0_fixed.copy()
        else:
            noise = rng.randn(2 * 6) * 0.05
            x_start = x0_fixed + noise
        
        try:
            res = minimize(_neg_log_score_fixed_diam, x_start,
                          method='L-BFGS-B', options={'maxiter': 5000, 'gtol': 1e-14, 'maxfun': 80000})
            if res.success:
                cand_pts = _make_fixed_diam_pts(res.x)
                d_min = np.min(np.linalg.norm(cand_pts[:, None, :] - cand_pts[None, :, :], axis=2) * (1 - np.eye(n)))
                if d_min > 1e-8:
                    cand_score = _score_points(cand_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts.tolist()
        except Exception:
            pass
    
    # Approach B: Full 8-point optimization with smooth diameter
    x0_full = np.array(reg_pts, dtype=float).flatten()
    
    try:
        res = minimize(_neg_log_score_smooth, x0_full, n=8,
                      method='L-BFGS-B', options={'maxiter': 5000, 'gtol': 1e-14, 'maxfun': 100000})
        if res.success and time.time() < _DEADLINE - 10:
            cand_pts = res.x.reshape(n, 2)
            d_min = np.min(np.linalg.norm(cand_pts[:, None, :] - cand_pts[None, :, :], axis=2) * (1 - np.eye(n)))
            if d_min > 1e-8:
                cand_score = _score_points(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts.tolist()
    except Exception:
        pass
    
    # Multi-start full 8-point with smooth diameter
    for start_i in range(20):
        if time.time() > _DEADLINE - 10:
            break
        if start_i == 0:
            x_start = x0_full.copy()
        else:
            noise = rng.randn(2 * n) * 0.08
            x_start = x0_full + noise
        
        try:
            res = minimize(_neg_log_score_smooth, x_start, n=8,
                          method='L-BFGS-B', options={'maxiter': 4000, 'gtol': 1e-14, 'maxfun': 60000})
            if res.success:
                cand_pts = res.x.reshape(n, 2)
                d_min = np.min(np.linalg.norm(cand_pts[:, None, :] - cand_pts[None, :, :], axis=2) * (1 - np.eye(n)))
                if d_min > 1e-8:
                    cand_score = _score_points(cand_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts.tolist()
        except Exception:
            pass
    
    # Approach C: Perturbation scan - try various anisotropic scalings
    # Scale x and y differently to break octagon symmetry
    for sx in np.linspace(0.7, 1.3, 25):
        for sy in np.linspace(0.7, 1.3, 25):
            if time.time() > _DEADLINE - 10:
                break
            pts = [[math.cos(a) * sx, math.sin(a) * sy] for a in angles]
            cand_score = _score_points(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = pts
    
    # Final polish: use the best configuration found and do a high-precision local optimization
    if best_pts and time.time() < _DEADLINE - 5:
        # Try to determine if best config has a natural diameter pair
        best_arr = np.array(best_pts, dtype=float)
        d_mat = np.linalg.norm(best_arr[:, None, :] - best_arr[None, :, :], axis=2)
        # Find the pair with max distance
        idx_max = np.unravel_index(np.argmax(d_mat), d_mat.shape)
        i_diam, j_diam = idx_max
        
        # If we can identify a clean diameter pair, re-optimize with that pair fixed
        # For now, do a final full optimization from the best point
        x_best = best_arr.flatten()
        try:
            res = minimize(_neg_log_score_smooth, x_best, n=8,
                          method='L-BFGS-B', options={'maxiter': 20000, 'gtol': 1e-16, 'maxfun': 300000})
            if res.success:
                cand_pts = res.x.reshape(n, 2)
                d_min = np.min(np.linalg.norm(cand_pts[:, None, :] - cand_pts[None, :, :], axis=2) * (1 - np.eye(n)))
                if d_min > 1e-8:
                    cand_score = _score_points(cand_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts.tolist()
        except Exception:
            pass
        
        # Also try fixed-diameter optimization from the best config
        # Identify which pair is the diameter and try to use it
        # For simplicity, try fixing points at (-1,0) and (1,0) and see if we can map
        # the other 6 points appropriately
        if time.time() < _DEADLINE - 5:
            # Try to align the diameter pair to (-1,0) and (1,0)
            # Compute the vector between the diameter pair
            p_i = best_arr[i_diam]
            p_j = best_arr[j_diam]
            vec = p_j - p_i
            dist = np.linalg.norm(vec)
            if dist > 1e-10:
                # Rotate and translate so that p_i -> (-1,0) and p_j -> (1,0)
                # Midpoint
                mid = (p_i + p_j) / 2.0
                # Unit vector
                u = vec / dist
                # Rotation matrix: u -> (1,0)
                # u = (cos theta, sin theta)
                theta = math.atan2(u[1], u[0])
                c, s = math.cos(-theta), math.sin(-theta)
                R = np.array([[c, -s], [s, c]])
                
                # Transform all points
                translated = best_arr - mid
                rotated = translated @ R.T
                # Now p_i should be at (-dist/2, 0) and p_j at (dist/2, 0)
                # Scale so that dist -> 2
                scaled = rotated * (2.0 / dist)
                
                # Find which points are now at approximately (-1,0) and (1,0)
                # They should be i_diam and j_diam
                # The other 6 points are the variable ones
                var_indices = [k for k in range(n) if k != i_diam and k != j_diam]
                var_pts_scaled = scaled[var_indices]
                x_init = var_pts_scaled.flatten()
                
                try:
                    res = minimize(_neg_log_score_fixed_diam, x_init,
                                  method='L-BFGS-B', options={'maxiter': 10000, 'gtol': 1e-15, 'maxfun': 150000})
                    if res.success:
                        cand_pts = _make_fixed_diam_pts(res.x)
                        d_min = np.min(np.linalg.norm(cand_pts[:, None, :] - cand_pts[None, :, :], axis=2) * (1 - np.eye(n)))
                        if d_min > 1e-8:
                            cand_score = _score_points(cand_pts)
                            if cand_score > best_score:
                                best_score = cand_score
                                best_pts = cand_pts.tolist()
                except Exception:
                    pass
    
    # Ensure return format
    result = [[float(p[0]), float(p[1])] for p in best_pts]
    return result
# EVOLVE-BLOCK-END

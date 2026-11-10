# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Precompute sofa points
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_SOFA_PTS = np.stack([_SOFA_X.ravel(), _SOFA_Y.ravel()], axis=1)  # shape (N, 2)
_N_SOFA = _SOFA_PTS.shape[0]

# Deadline guard
_DEADLINE = 600.0
_START_TIME = None


def _elapsed():
    return time.time() - _START_TIME if _START_TIME is not None else 0.0


def _check_deadline():
    return _elapsed() > _DEADLINE - 30.0  # 30s safety margin


def _generate_placements(waypoints):
    """
    Generate the full list of placements including start (0,0,0) and end (-2,0,90).
    Each placement is (x, y, a_degrees). Linearly interpolate between consecutive
    waypoints with m = max(1, ceil(|a2 - a1|)) steps.
    Returns array of shape (M, 3).
    """
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    full = [P0] + [np.array(p, dtype=np.float64) for p in waypoints] + [PN]
    
    placements = [full[0].copy()]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            placements.append(p1 + t * (p2 - p1))
    
    return np.array(placements, dtype=np.float64)


def _point_in_base(pts):
    """
    Check if points (in local coords, after rotation) are inside the base L-shape.
    Base: {0<=u<=1, -3<=v<=1} UNION {-3<=u<=1, 0<=v<=1}
    Returns boolean array.
    """
    u = pts[:, 0]
    v = pts[:, 1]
    arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return arm1 | arm2


def _compute_sofa_mask(waypoints):
    """
    Compute which sofa points are inside ALL placements.
    Returns boolean array of shape (_N_SOFA,).
    """
    placements = _generate_placements(waypoints)
    
    mask = np.ones(_N_SOFA, dtype=bool)
    
    for (x, y, a_deg) in placements:
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # R(-a) applied to (w - (x,y))
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        dx = _SOFA_PTS[:, 0] - x
        dy = _SOFA_PTS[:, 1] - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        local = np.column_stack([u, v])
        inside = _point_in_base(local)
        
        mask &= inside
        if not mask.any():
            break
    
    return mask


def _compute_area(waypoints):
    """Compute the area of the sofa (intersection of all placements)."""
    mask = _compute_sofa_mask(waypoints)
    return mask.sum() * 0.0001


def _soft_min_area(waypoints, beta=20.0):
    """
    Smooth surrogate: instead of hard intersection, use a soft version.
    For each sofa point, compute the minimum "margin" over all placements,
    then use soft-min to approximate the count.
    """
    placements = _generate_placements(waypoints)
    
    # For each placement, compute signed distance-like value for each sofa point
    # A point is inside if R(-a)(w-(x,y)) is in the base.
    # We can compute a "distance" from the base for each point in each placement.
    
    margins = np.empty((len(placements), _N_SOFA), dtype=np.float64)
    
    for idx, (x, y, a_deg) in enumerate(placements):
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = _SOFA_PTS[:, 0] - x
        dy = _SOFA_PTS[:, 1] - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Compute distance to base L-shape (negative if inside)
        # Base: arm1 = {0<=u<=1, -3<=v<=1}, arm2 = {-3<=u<=1, 0<=v<=1}
        
        # Distance to arm1 rectangle
        du1 = np.maximum(0, np.maximum(0, 0 - u), np.maximum(0, u - 1))
        dv1 = np.maximum(0, np.maximum(0, -3 - v), np.maximum(0, v - 1))
        d1 = np.sqrt(du1**2 + dv1**2)
        
        # Distance to arm2 rectangle
        du2 = np.maximum(0, np.maximum(0, -3 - u), np.maximum(0, u - 1))
        dv2 = np.maximum(0, np.maximum(0, 0 - v), np.maximum(0, v - 1))
        d2 = np.sqrt(du2**2 + dv2**2)
        
        # Inside if distance is 0; margin = -min(d1, d2) (positive if inside)
        min_d = np.minimum(d1, d2)
        margins[idx] = -min_d
    
    # Soft-min over placements for each point
    # For each point, we want to know if it's inside ALL placements.
    # min_margin = min over placements of margin. If > 0, inside all.
    # Soft approximation: use log-sum-exp style
    
    # For each point, compute the minimum margin
    min_margin = margins.min(axis=0)
    
    # Count points with min_margin > 0, but use soft version
    # soft_count = sum(exp(beta * min_margin) / (1 + exp(beta * min_margin))) approximates count of positive
    # Actually, let's use: for each point, contribution = sigmoid(beta * min_margin)
    # This gives ~1 if min_margin >> 0, ~0 if min_margin << 0
    
    contrib = 1.0 / (1.0 + np.exp(-beta * min_margin))
    return contrib.sum() * 0.0001


def _score_neg(params):
    """Negative area for minimization."""
    waypoints = _params_to_waypoints(params)
    return -_compute_area(waypoints)


def _score_neg_soft(params, beta=20.0):
    """Negative soft area for minimization."""
    waypoints = _params_to_waypoints(params)
    return -_soft_min_area(waypoints, beta)


def _params_to_waypoints(params):
    """Convert flat parameter array to list of waypoints."""
    n_wp = len(params) // 3
    wps = []
    for i in range(n_wp):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        wps.append([x, y, a])
    return wps


def _waypoints_to_params(waypoints):
    """Convert list of waypoints to flat parameter array."""
    params = []
    for wp in waypoints:
        params.extend([wp[0], wp[1], wp[2]])
    return np.array(params, dtype=np.float64)


def _initial_guess():
    """Generate initial guess for waypoints."""
    # Use the baseline as starting point
    baseline = [
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ]
    return _waypoints_to_params(baseline)


def moving_sofa_2d() -> np.ndarray:
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with baseline
    best_waypoints = [
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ]
    best_area = _compute_area(best_waypoints)
    
    # Try optimization with Nelder-Mead on the smooth surrogate
    x0 = _initial_guess()
    
    # First, optimize the soft surrogate
    try:
        res = minimize(
            _score_neg_soft,
            x0,
            method='Nelder-Mead',
            args=(20.0,),
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6}
        )
        if not _check_deadline():
            candidate_wp = _params_to_waypoints(res.x)
            cand_area = _compute_area(candidate_wp)
            if cand_area > best_area:
                best_area = cand_area
                best_waypoints = candidate_wp
    except Exception:
        pass
    
    # Polish with direct optimization
    if not _check_deadline():
        x0 = _waypoints_to_params(best_waypoints)
        try:
            res = minimize(
                _score_neg,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-8}
            )
            candidate_wp = _params_to_waypoints(res.x)
            cand_area = _compute_area(candidate_wp)
            if cand_area > best_area:
                best_area = cand_area
                best_waypoints = candidate_wp
        except Exception:
            pass
    
    # Try with more waypoints (12)
    if not _check_deadline():
        # Extend to 12 waypoints by interpolating
        n12 = 12
        n8 = 8
        new_wp = []
        for i in range(n12):
            t = i / (n12 - 1) * (n8 - 1)
            idx = int(t)
            frac = t - idx
            if idx >= n8 - 1:
                new_wp.append(list(best_waypoints[-1]))
            else:
                p1 = best_waypoints[idx]
                p2 = best_waypoints[idx + 1]
                new_wp.append([
                    p1[0] + frac * (p2[0] - p1[0]),
                    p1[1] + frac * (p2[1] - p1[1]),
                    p1[2] + frac * (p2[2] - p1[2]),
                ])
        
        x0 = _waypoints_to_params(new_wp)
        try:
            res = minimize(
                _score_neg_soft,
                x0,
                method='Nelder-Mead',
                args=(20.0,),
                options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if not _check_deadline():
                candidate_wp = _params_to_waypoints(res.x)
                cand_area = _compute_area(candidate_wp)
                if cand_area > best_area:
                    best_area = cand_area
                    best_waypoints = candidate_wp
        except Exception:
            pass
        
        # Polish
        if not _check_deadline():
            x0 = _waypoints_to_params(best_waypoints)
            try:
                res = minimize(
                    _score_neg,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-8}
                )
                candidate_wp = _params_to_waypoints(res.x)
                cand_area = _compute_area(candidate_wp)
                if cand_area > best_area:
                    best_area = cand_area
                    best_waypoints = candidate_wp
            except Exception:
                pass
    
    # Ensure all values are finite
    result = np.array(best_waypoints, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        result = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END

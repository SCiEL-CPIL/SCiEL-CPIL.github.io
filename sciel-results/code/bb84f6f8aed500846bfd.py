# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Precompute sofa grid points
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
    Pn = np.array([-2.0, 0.0, 90.0])
    
    all_points = [P0]
    for i in range(len(waypoints)):
        all_points.append(waypoints[i])
    all_points.append(Pn)
    
    all_points = np.array(all_points)
    
    # Generate dense placements
    dense = [all_points[0]]
    for i in range(len(all_points) - 1):
        p1 = all_points[i]
        p2 = all_points[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            dense.append(p1 + t * (p2 - p1))
    
    return np.array(dense)


def _point_in_base(points, placements):
    """
    Check if points are inside all placements.
    points: (N, 2) array
    placements: (M, 3) array of (x, y, a_degrees)
    Returns boolean mask of shape (N,)
    """
    N = points.shape[0]
    M = placements.shape[0]
    
    # Start with all True
    inside = np.ones(N, dtype=bool)
    
    for m in range(M):
        x, y, a_deg = placements[m]
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Rotate points into local frame: R(-a)(w - (x,y))
        dx = points[:, 0] - x
        dy = points[:, 1] - y
        
        # R(-a) = [[cos(-a), -sin(-a)], [sin(-a), cos(-a)]] = [[cos_a, sin_a], [-sin_a, cos_a]]
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Base corridor: union of {0<=u<=1, -3<=v<=1} and {-3<=u<=1, 0<=v<=1}
        in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = in_rect1 | in_rect2
        
        inside &= in_base
        if not np.any(inside):
            break
    
    return inside


def _compute_area(waypoints):
    """
    Compute the sofa area for given waypoints.
    Returns the area (number of points * 0.0001).
    """
    placements = _generate_placements(waypoints)
    inside = _point_in_base(_SOFA_PTS, placements)
    return np.sum(inside) * 0.0001


def _neg_area(waypoints_flat):
    """
    Negative area for minimization.
    waypoints_flat: flat array of length 3*n
    """
    n = len(waypoints_flat) // 3
    waypoints = waypoints_flat.reshape(n, 3)
    return -_compute_area(waypoints)


def _angle_bounds(n):
    """
    Generate bounds for n waypoints.
    x in [-1.5, 0.5], y in [-0.5, 0.5], a in [5, 85]
    """
    lb = np.empty(3 * n)
    ub = np.empty(3 * n)
    for i in range(n):
        lb[3*i] = -1.5
        ub[3*i] = 0.5
        lb[3*i+1] = -0.5
        ub[3*i+1] = 0.5
        lb[3*i+2] = 5.0
        ub[3*i+2] = 85.0
    return list(zip(lb, ub))


def _initial_guesses(n):
    """
    Generate diverse initial guesses for n waypoints.
    Returns list of flat arrays.
    """
    guesses = []
    
    # Arc-based guess
    for i in range(n):
        t = (i + 1) / (n + 1)
        a = 10 + 70 * t
        x = -0.5 * t
        y = 0.2 * np.sin(np.pi * t)
        guesses.append(np.array([x, y, a]))
    guesses[0] = np.array([0.5 * np.array(guesses[0])])
    guesses_flat = np.array(guesses).ravel()
    guesses = [guesses_flat]
    
    # Linear path guess
    for i in range(n):
        t = (i + 1) / (n + 1)
        a = 10 + 70 * t
        x = -0.7 * t
        y = 0.1
        guesses.append(np.array([x, y, a]))
    guesses_flat = np.array(guesses).ravel()
    guesses.append(guesses_flat)
    
    # S-curve guess
    for i in range(n):
        t = (i + 1) / (n + 1)
        a = 10 + 70 * t
        x = -0.5 * t + 0.1 * np.sin(2 * np.pi * t)
        y = 0.15 * np.sin(np.pi * t)
        guesses.append(np.array([x, y, a]))
    guesses_flat = np.array(guesses).ravel()
    guesses.append(guesses_flat)
    
    # Asymmetric turn guess
    for i in range(n):
        t = (i + 1) / (n + 1)
        if t < 0.4:
            a = 10 + 50 * (t / 0.4)
            x = -0.3 * (t / 0.4)
            y = 0.1
        else:
            a = 60 + 30 * ((t - 0.4) / 0.6)
            x = -0.3 - 0.4 * ((t - 0.4) / 0.6)
            y = 0.1 + 0.1 * np.sin(np.pi * (t - 0.4) / 0.6)
        guesses.append(np.array([x, y, a]))
    guesses_flat = np.array(guesses).ravel()
    guesses.append(guesses_flat)
    
    return guesses


def moving_sofa_2d() -> np.ndarray:
    global _START_TIME
    _START_TIME = time.time()
    
    best_waypoints = None
    best_area = 0.0
    
    # Try different numbers of waypoints
    for n in [6, 8, 10]:
        if _check_deadline():
            break
        
        bounds = _angle_bounds(n)
        guesses = _initial_guesses(n)
        
        # Use differential evolution for global search
        try:
            result = differential_evolution(
                _neg_area,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                polish=True,
                init='sobol',
                workers=1
            )
            
            if result.success or result.fun < -best_area:
                waypoints = result.x.reshape(n, 3)
                area = -result.fun
                if area > best_area:
                    best_area = area
                    best_waypoints = waypoints.copy()
        except Exception:
            pass
        
        if _check_deadline():
            break
        
        # Local refinement with L-BFGS-B from best DE result and initial guesses
        for guess in guesses:
            if _check_deadline():
                break
            
            try:
                result = minimize(
                    _neg_area,
                    guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if result.success or result.fun < -best_area:
                    waypoints = result.x.reshape(n, 3)
                    area = -result.fun
                    if area > best_area:
                        best_area = area
                        best_waypoints = waypoints.copy()
            except Exception:
                pass
    
    # Fallback to baseline if nothing found
    if best_waypoints is None:
        best_waypoints = np.array(
            [
                [-0.1, 0.1, 15.0],
                [-0.2, 0.2, 20.0],
                [-0.3, 0.3, 34.0],
                [-0.3, 0.3, 40.0],
                [-0.4, 0.3, 50.0],
                [-0.5, 0.2, 55.0],
                [-0.6, 0.2, 75.0],
                [-0.7, 0.3, 86.0],
            ],
            dtype=np.float64,
        )
    
    return best_waypoints

# EVOLVE-BLOCK-END

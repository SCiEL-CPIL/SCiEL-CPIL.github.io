import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar

N = 14

# Pre-compute indices for all triangles i < j < k
TRI_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area using pre-computed indices."""
    if len(points) < 3:
        return 0.0
    
    i, j, k = TRI_INDICES.T
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _neg_score(points_flat):
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_heptagons(r, theta):
    """
    Generate two concentric heptagons.
    Outer ring: radius 1, 7 points starting at angle 0.
    Inner ring: radius r, 7 points rotated by theta.
    """
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + theta
    
    p_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    p_inner = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    return np.vstack([p_outer, p_inner])

def _score_two_heptagons(r, theta):
    """Score for two heptagons configuration."""
    pts = _two_heptagons(r, theta)
    return _score(pts)

def _neg_score_two_heptagons(params):
    """Negative score for optimization."""
    r, theta = params
    return -_score_two_heptagons(r, theta)

def _local_optimize(points, time_limit=20.0):
    """Nelder-Mead refinement with multiple restarts."""
    start_time = time.time()
    best_score = _score(points)
    best_pts = points.copy()
    
    # Try several perturbations to escape local minima
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
            
        if restart == 0:
            x0 = points.flatten()
        else:
            # Add small noise
            noise_scale = 0.01 * (restart ** 0.5)
            x0 = points.flatten() + rng.normal(0, noise_scale, size=points.shape)
        
        try:
            res = minimize(
                _neg_score, x0, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    return best_score, best_pts

def _optimize_structure(time_limit=30.0):
    """
    Optimize the two-heptagon structure (r, theta).
    Uses nested 1D optimization or 2D minimization.
    """
    start_time = time.time()
    
    # Initial guess: r=0.5, theta=pi/7 (half step)
    best_r = 0.5
    best_theta = np.pi / 7
    best_score = _score_two_heptagons(best_r, best_theta)
    
    def _neg_score_2d(x):
        if time.time() - start_time > time_limit:
            return 1000.0 # Penalty if timeout
        r, theta = x
        return -_score_two_heptagons(r, theta)
    
    # Use Nelder-Mead on the 2D parameter space
    x0 = np.array([best_r, best_theta])
    
    try:
        res = minimize(
            _neg_score_2d, x0, method='Nelder-Mead',
            bounds=[(0.1, 0.9), (0, 2*np.pi)],
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if time.time() - start_time < time_limit:
            r_opt, theta_opt = res.x
            s_opt = -res.fun
            if s_opt > best_score:
                best_score = s_opt
                best_r = r_opt
                best_theta = theta_opt
    except Exception:
        pass
    
    # Try a few discrete theta values to see if we missed a better basin
    for theta_init in [0, np.pi/7, 2*np.pi/7, np.pi/14, 3*np.pi/14]:
        if time.time() - start_time > time_limit - 5:
            break
        try:
            # Optimize r for fixed theta
            res_r = minimize_scalar(
                lambda r: -_score_two_heptagons(r, theta_init),
                bounds=(0.1, 0.9),
                method='bounded',
                options={'xatol': 1e-6}
            )
            r_opt = res_r.x
            s_opt = -res_r.fun
            if s_opt > best_score:
                best_score = s_opt
                best_r = r_opt
                best_theta = theta_init
        except Exception:
            continue

    return best_score, best_r, best_theta

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # 1. Try the structured Two-Heptagon optimization
    s_struct, r_opt, theta_opt = _optimize_structure(time_limit=30.0)
    if s_struct > best_overall_score:
        best_overall_score = s_struct
        best_overall_points = _two_heptagons(r_opt, theta_opt)
    
    # 2. Local optimization on the best structured points
    if time.time() < deadline - 20 and best_overall_points is not None:
        s, p = _local_optimize(best_overall_points, time_limit=20.0)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p
            
    # 3. Fallback: Regular polygon if structure failed (unlikely)
    if best_overall_points is None:
        init_pts = _regular_polygon(N, 1.0)
        s, p = _local_optimize(init_pts, time_limit=10.0)
        best_overall_score = s
        best_overall_points = p

    # 4. Final Polish with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 10 and best_overall_points is not None:
        s, p = _local_optimize(best_overall_points, time_limit=max(5, remaining_time - 5))
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p

    return best_overall_points

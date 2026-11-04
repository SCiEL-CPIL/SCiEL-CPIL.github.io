import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar, differential_evolution

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

def _two_heptagons(r, theta):
    """
    Generate 14 points: 7 on outer unit circle, 7 on inner circle of radius r,
    with inner ring rotated by theta relative to outer ring.
    """
    n_out = 7
    n_in = 7
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
    
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _score_two_heptagons(r, theta):
    """Score for two heptagons configuration."""
    pts = _two_heptagons(r, theta)
    return _score(pts)

def _neg_score_two_heptagons(params):
    """Negative score for optimization."""
    r, theta = params
    if r <= 0 or r > 1:
        return 1.0
    return -_score_two_heptagons(r, theta)

def _build_ring_config(radii, angles_per_ring, phase_offsets):
    """
    Build point set from concentric rings.
    radii: array of ring radii (outermost first)
    angles_per_ring: list of counts per ring
    phase_offsets: list of angular offsets per ring
    """
    pts = []
    for r, n_ang, phase in zip(radii, angles_per_ring, phase_offsets):
        if n_ang == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, n_ang, endpoint=False) + phase
        ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts.append(ring_pts)
    return np.vstack(pts)

def _optimize_ring_config(n1, n2, n3, r_init, phase_init, time_limit=15.0):
    """
    Optimize a 3-ring configuration with n1+n2+n3=N points.
    Parameters: [r1, r2, r3, phase1, phase2, phase3]
    r1 >= r2 >= r3 > 0, phases in [0, 2*pi)
    """
    start_time = time.time()
    
    def build_from_params(x):
        r1, r2, r3, p1, p2, p3 = x
        # Ensure ordering and positivity
        r1 = max(0.5, r1)
        r2 = max(0.1, min(r1, r2))
        r3 = max(0.01, min(r2, r3))
        p1 = p1 % (2 * np.pi)
        p2 = p2 % (2 * np.pi)
        p3 = p3 % (2 * np.pi)
        return _build_ring_config([r1, r2, r3], [n1, n2, n3], [p1, p2, p3])
    
    def neg_score_params(x):
        pts = build_from_params(x)
        return -_score(pts)
    
    x0 = np.array([r_init[0], r_init[1], r_init[2], phase_init[0], phase_init[1], phase_init[2]])
    best_score = _score(build_from_params(x0))
    best_x = x0.copy()
    
    bounds = [(0.5, 2.0), (0.1, 2.0), (0.01, 2.0), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    
    # Try multiple local optimizations from different starts
    starts = [x0]
    # Add some perturbed starts
    rng = np.random.RandomState(42 + n1*100 + n2*10 + n3)
    for _ in range(3):
        x_pert = x0 + rng.normal(0, 0.1, size=6)
        x_pert[0] = max(0.5, x_pert[0])
        x_pert[1] = max(0.1, min(x_pert[1], x_pert[0]))
        x_pert[2] = max(0.01, min(x_pert[2], x_pert[1]))
        x_pert[3:] = x_pert[3:] % (2*np.pi)
        starts.append(x_pert)
    
    for xi in starts:
        if time.time() > start_time + time_limit - 2:
            break
        try:
            res = minimize(
                neg_score_params, xi, method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    best_pts = build_from_params(best_x)
    return best_score, best_pts

def _optimize_two_ring_config(n1, n2, r_init, phase_init, time_limit=15.0):
    """
    Optimize a 2-ring configuration with n1+n2=N points.
    Parameters: [r1, r2, phase1, phase2]
    """
    start_time = time.time()
    
    def build_from_params(x):
        r1, r2, p1, p2 = x
        r1 = max(0.5, r1)
        r2 = max(0.05, min(r1, r2))
        p1 = p1 % (2 * np.pi)
        p2 = p2 % (2 * np.pi)
        return _build_ring_config([r1, r2], [n1, n2], [p1, p2])
    
    def neg_score_params(x):
        pts = build_from_params(x)
        return -_score(pts)
    
    x0 = np.array([r_init[0], r_init[1], phase_init[0], phase_init[1]])
    best_score = _score(build_from_params(x0))
    best_x = x0.copy()
    
    bounds = [(0.5, 2.0), (0.05, 2.0), (0, 2*np.pi), (0, 2*np.pi)]
    
    starts = [x0]
    rng = np.random.RandomState(42 + n1*100 + n2)
    for _ in range(4):
        x_pert = x0 + rng.normal(0, 0.1, size=4)
        x_pert[0] = max(0.5, x_pert[0])
        x_pert[1] = max(0.05, min(x_pert[1], x_pert[0]))
        x_pert[2:] = x_pert[2:] % (2*np.pi)
        starts.append(x_pert)
    
    for xi in starts:
        if time.time() > start_time + time_limit - 2:
            break
        try:
            res = minimize(
                neg_score_params, xi, method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    best_pts = build_from_params(best_x)
    return best_score, best_pts

def _local_optimize_free(points, time_limit=10.0):
    """Local optimization of all points freely using Powell's method."""
    start_time = time.time()
    
    def obj(x):
        if time.time() > start_time + time_limit:
            return 1e6  # Penalize timeout
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    
    try:
        res = minimize(
            obj, x0, method='Powell',
            options={
                'maxiter': 50, 
                'xtol': 1e-8, 
                'ftol': 1e-10
            }
        )
        
        if time.time() < start_time + time_limit:
            final_score = -res.fun
            if final_score > 0:
                return final_score, res.x.reshape(-1, 2).copy()
                
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial configuration: Two concentric heptagons
    best_points = _two_heptagons(0.6, np.pi / 7)
    best_score = _score(best_points)
    
    # Strategy 1: 2D grid search over (r, theta)
    r_values = np.linspace(0.3, 0.9, 13)
    theta_values = np.linspace(0, 2 * np.pi / 7, 14, endpoint=False)
    
    for r in r_values:
        if time.time() > deadline - 30:
            break
        for theta in theta_values:
            if time.time() > deadline - 30:
                break
            score = _score_two_heptagons(r, theta)
            if score > best_score:
                best_score = score
                best_points = _two_heptagons(r, theta)
    
    # Strategy 2: Local optimization of (r, theta) using Powell
    start_points = [
        (0.5, np.pi / 7),
        (0.6, np.pi / 7),
        (0.7, np.pi / 7),
        (0.5, 0),
        (0.6, 0),
        (0.7, 0),
    ]
    
    for r0, theta0 in start_points:
        if time.time() > deadline - 20:
            break
        
        x0 = np.array([r0, theta0])
        
        try:
            res = minimize(
                _neg_score_two_heptagons, x0, method='Powell',
                bounds=[(0.01, 0.99), (0, 2 * np.pi / 7)],
                options={
                    'maxiter': 50, 
                    'xtol': 1e-8, 
                    'ftol': 1e-10
                }
            )
            
            if time.time() < deadline - 10:
                r_opt, theta_opt = res.x
                if 0.01 < r_opt < 0.99:
                    score = _score_two_heptagons(r_opt, theta_opt)
                    if score > best_score:
                        best_score = score
                        best_points = _two_heptagons(r_opt, theta_opt)
                        
        except Exception:
            pass
    
    # Strategy 3: Explore symmetry-reduced ring configurations (from inspiration)
    ring_configs_3 = [
        (6, 5, 3),
        (5, 5, 4),
        (7, 4, 3),
        (6, 4, 4),
    ]
    
    for (n1, n2, n3) in ring_configs_3:
        if time.time() > deadline - 40:
            break
        
        # Try multiple initial radius ratios
        for r_ratio in [0.5, 0.6, 0.7]:
            if time.time() > deadline - 40:
                break
            r_init = [1.0, 0.5 + 0.3*r_ratio, 0.3 + 0.2*r_ratio]
            phase_init = [0.0, np.pi/n1, np.pi/n2]
            
            result = _optimize_ring_config(n1, n2, n3, r_init, phase_init, time_limit=10.0)
            if result is not None:
                score, points = result
                if score > best_score:
                    best_score = score
                    best_points = points
    
    # Strategy 4: Free local optimization of all points
    if time.time() < deadline - 10:
        result = _local_optimize_free(best_points, time_limit=8.0)
        if result is not None:
            score, points = result
            if score > best_score:
                best_score = score
                best_points = points
    
    # Strategy 5: Simulated Annealing for final polish
    if time.time() < deadline - 5:
        current_points = best_points.copy()
        current_score = best_score
        
        T_start = 0.01
        T_end = 1e-6
        n_steps = 500
        step_scale = 0.02
        
        rng = np.random.default_rng(42)
        
        for i in range(n_steps):
            if time.time() > deadline - 2:
                break
            
            noise = rng.normal(0, step_scale, size=(N, 2))
            new_points = current_points + noise
            
            new_score = _score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = new_points.copy()
            else:
                delta = current_score - new_score
                if delta > 0:
                    T = T_start * (T_end / T_start) ** (i / n_steps)
                    if T > 1e-10:
                        prob = np.exp(-delta / T)
                        if rng.random() < prob:
                            current_points = new_points
                            current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
    
    # Fallback if something went wrong
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

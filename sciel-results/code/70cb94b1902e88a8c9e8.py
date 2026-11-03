"""
Support-function (Minkowski-sum) parameterization + multi-start Nelder-Mead.
Convexity guaranteed by construction; no penalty terms needed.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
H = N // 2

# EVOLVE-BLOCK-START

def _polygon_from_support(h):
    """
    Convert support numbers h (shape N,) to polygon vertices (shape N, 2).
    
    The polygon is the Minkowski sum of N segments:
    P = sum_k h_k * e_k
    where e_k is the unit vector in direction theta_k = 2*pi*k/N.
    
    For a regular N-gon with support h_k = c for all k, the vertices are
    at radius c * cos(pi/N) / sin(pi/N) ... actually let's derive properly.
    
    For a regular N-gon, the support function in direction theta_k is:
    h_k = R * cos(pi/N) where R is circumradius.
    
    The vertex at angle phi_k = 2*pi*k/N has position:
    v_k = h_k * (cos(phi_k), sin(phi_k)) + h_{k+1} * (cos(phi_{k+1}), sin(phi_{k+1}))
    ... no, that's not right either.
    
    Actually, for the Minkowski sum of segments h_k * u_k where u_k = (cos(2*pi*k/N), sin(2*pi*k/N)):
    The resulting polygon has vertices that are the "corners" of the Minkowski sum.
    
    Let me think differently. The support function approach:
    If we have support numbers h_k in directions theta_k = 2*pi*k/N,
    the polygon is the intersection of half-planes:
    x * cos(theta_k) + y * sin(theta_k) <= h_k
    
    For a convex polygon with these support numbers, the vertices are the intersections
    of consecutive half-plane boundaries.
    
    Vertex k is the intersection of:
    x*cos(theta_k) + y*sin(theta_k) = h_k
    x*cos(theta_{k+1}) + y*sin(theta_{k+1}) = h_{k+1}
    
    Solving this 2x2 system for each k gives vertex k.
    """
    theta = 2 * np.pi * np.arange(N) / N
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    vertices = np.zeros((N, 2))
    for k in range(N):
        k1 = (k + 1) % N
        # Solve:
        # cos_t[k] * x + sin_t[k] * y = h[k]
        # cos_t[k1] * x + sin_t[k1] * y = h[k1]
        det = cos_t[k] * sin_t[k1] - sin_t[k] * cos_t[k1]
        if abs(det) < 1e-12:
            vertices[k] = [h[k] * cos_t[k], h[k] * sin_t[k]]
            continue
        x = (h[k] * sin_t[k1] - sin_t[k] * h[k1]) / det
        y = (cos_t[k] * h[k1] - h[k] * cos_t[k1]) / det
        vertices[k] = [x, y]
    
    return vertices

def _min_triangle_area(points):
    """Compute minimum area over all triples of points."""
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _score(points):
    """Minimum triangle area divided by convex hull area."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective_support(h):
    """
    Objective for optimization: minimize negative score.
    h: support numbers, shape (N,), must be positive for convexity.
    We use log-parameterization to ensure positivity: h_k = exp(p_k)
    """
    # Ensure h is positive
    if np.any(h <= 1e-6):
        return 1e10
    
    points = _polygon_from_support(h)
    s = _score(points)
    if s < 1e-12:
        return 1e10
    return -s

def _objective_log_support(p):
    """
    Objective in log-space: p_k = log(h_k), so h_k = exp(p_k).
    This ensures h_k > 0 automatically.
    """
    h = np.exp(p)
    return _objective_support(h)

def _generate_initial_log_configs(n_configs=20):
    """Generate diverse initial configurations in log-support space."""
    configs = []
    
    # Config 1: Regular 14-gon (all support numbers equal)
    # For regular N-gon, h_k = R * cos(pi/N) where R is circumradius
    # We can just use all equal, say log(1.0) = 0
    configs.append(np.zeros(N))
    
    # Configs 2-5: Perturbed regular with different angular modes
    for k in range(1, 5):
        perturbation = 0.1 * np.sin(k * 2 * np.pi * np.arange(N) / N)
        configs.append(perturbation)
    
    # Configs 6-10: Random positive perturbations
    for i in range(5):
        rng = np.random.RandomState(42 + i)
        configs.append(rng.normal(0, 0.2, N))
    
    # Configs 11-15: Elliptic-like (alternate support numbers)
    for stretch in [0.5, 0.7, 1.0, 1.3, 1.5]:
        h = np.zeros(N)
        for i in range(N):
            if i % 2 == 0:
                h[i] = np.log(stretch)
            else:
                h[i] = 0.0
        configs.append(h)
    
    # Configs 16-20: More structured random
    for i in range(5):
        rng = np.random.RandomState(100 + i)
        # Low-frequency perturbation
        freq = rng.choice([1, 2, 3])
        phase = rng.uniform(0, 2*np.pi)
        amp = rng.uniform(0.1, 0.4)
        configs.append(amp * np.sin(freq * 2 * np.pi * np.arange(N) / N + phase))
    
    return configs

def _refine_local_log(p, time_limit):
    """Refine a configuration using Nelder-Mead in log-support space."""
    start_time = time.time()
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e10
        return _objective_log_support(p)
    
    try:
        res = minimize(
            obj,
            p,
            method='Nelder-Mead',
            options={
                'maxiter': 300,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        return res.x
    except Exception:
        return p

def construct():
    """
    Generate 14 points to maximize min triangle area / hull area.
    
    Strategy:
    1. Use support-function parameterization (14D, positivity guaranteed by log-space).
    2. Multi-start Nelder-Mead local refinement from diverse initial configs.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = -1e10
    best_params = None
    best_points = None
    
    # Generate initial configs in log-support space
    initial_configs = _generate_initial_log_configs(20)
    
    # Score all initial configs
    config_scores = []
    for cfg in initial_configs:
        s = -_objective_log_support(cfg)
        config_scores.append((s, cfg))
    
    # Sort configs by initial score (best first)
    config_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top configs with local optimization
    n_to_refine = min(12, len(config_scores))
    for i in range(n_to_refine):
        if time.time() > deadline - 10:
            break
        
        s_init, cfg = config_scores[i]
        time_left = deadline - time.time() - 5
        if time_left < 3:
            break
        
        refined = _refine_local_log(cfg, time_left)
        s_ref = -_objective_log_support(refined)
        
        if s_ref > best_score:
            best_score = s_ref
            best_params = refined.copy()
            best_points = _polygon_from_support(np.exp(refined))
    
    # If we have a good result, do a second round of refinement with tighter starts
    if best_params is not None and time.time() < deadline - 30:
        # Generate perturbations around the best solution
        rng = np.random.RandomState(42)
        for i in range(5):
            if time.time() > deadline - 10:
                break
            
            perturbation = rng.normal(0, 0.1, N)
            cfg = best_params + perturbation
            time_left = deadline - time.time() - 5
            if time_left < 3:
                break
            
            refined = _refine_local_log(cfg, time_left)
            s_ref = -_objective_log_support(refined)
            
            if s_ref > best_score:
                best_score = s_ref
                best_params = refined.copy()
                best_points = _polygon_from_support(np.exp(refined))
    
    # Final: Normalize to unit area
    if best_points is not None:
        try:
            hull = ConvexHull(best_points)
            hull_area = hull.volume
            if hull_area > 1e-10:
                scale = np.sqrt(1.0 / hull_area)
                return best_points * scale
        except Exception:
            pass
    
    # Fallback: regular 14-gon normalized
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        return pts * scale
    except Exception:
        return pts

# EVOLVE-BLOCK-END

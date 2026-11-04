import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

# Base regular polygon angles
_BASE_ANGLES = 2 * np.pi * np.arange(N) / N
_BASE_COS = np.cos(_BASE_ANGLES)
_BASE_SIN = np.sin(_BASE_ANGLES)

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Calculate Minimum Triangle Area
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    min_area = 0.5 * np.min(np.abs(cross))

    return min_area / hull_area

def _soft_min_score(points, beta=100.0):
    """
    Soft-min surrogate for the minimum triangle area normalized by hull area.
    """
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    areas = 0.5 * np.abs(cross)

    # Soft-min via log-sum-exp
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_area
    soft_min = max_area - log_sum / beta

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    return soft_min / hull_area

def _generate_jittered_polygon(params, noise_seed=0):
    """
    Generate 14 points as a jittered regular polygon.
    
    params layout:
      params[0]: scale (radial multiplier)
      params[1]: rotation angle
      params[2]: shear x (affine transformation)
      params[3]: shear y (affine transformation)
      params[4]: noise amplitude
      params[5:]: 4 Gaussian random field coefficients (low-dimensional noise)
    
    The noise is generated from a fixed low-dimensional Gaussian random field:
    noise[i] = amplitude * (c0*cos(1*theta_i) + c1*sin(1*theta_i) + c2*cos(2*theta_i) + c3*sin(2*theta_i))
    
    This creates smooth, correlated radial perturbations that break the D14 symmetry
    of the regular polygon while maintaining a structured (not random) deformation.
    """
    scale = params[0]
    rotation = params[1]
    shear_x = params[2]
    shear_y = params[3]
    amplitude = params[4]
    c0, c1, c2, c3 = params[5], params[6], params[7], params[8]
    
    # Base regular polygon
    base_x = _BASE_COS
    base_y = _BASE_SIN
    
    # Radial noise from low-dimensional Gaussian random field
    theta = _BASE_ANGLES
    noise = amplitude * (c0 * np.cos(theta) + c1 * np.sin(theta) + 
                         c2 * np.cos(2*theta) + c3 * np.sin(2*theta))
    
    # Apply radial jitter
    radii = scale * (1.0 + noise)
    
    x = radii * base_x
    y = radii * base_y
    
    # Apply rotation
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    x_rot = x * cos_r - y * sin_r
    y_rot = x * sin_r + y * cos_r
    
    # Apply shear (affine transformation)
    x_final = x_rot + shear_x * y_rot
    y_final = y_rot + shear_y * x_rot
    
    return np.column_stack((x_final, y_final))

def _neg_jittered_score(params, beta=None):
    """
    Negative score for optimization.
    If beta is None, use exact score; otherwise use soft-min surrogate.
    """
    points = _generate_jittered_polygon(params)
    if beta is None:
        return -_calculate_score(points)
    else:
        return -_soft_min_score(points, beta)

def _optimize_jittered_polygon(deadline):
    """
    Optimize the jittered polygon parameters using Nelder-Mead.
    Multiple restarts with different noise seeds to explore the parameter space.
    """
    best_score = 0.0
    best_params = None
    
    # Initial parameter ranges
    scale_range = (0.3, 1.0)
    rotation_range = (0.0, 2 * np.pi / N)  # Exploit rotational symmetry
    shear_range = (-0.3, 0.3)
    amplitude_range = (0.0, 0.3)
    coeff_range = (-1.0, 1.0)
    
    # Multiple restarts with different random initializations
    rng = np.random.RandomState(42)
    
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Random initial parameters
        x0 = np.array([
            rng.uniform(*scale_range),
            rng.uniform(*rotation_range),
            rng.uniform(*shear_range),
            rng.uniform(*shear_range),
            rng.uniform(*amplitude_range),
            rng.uniform(*coeff_range),
            rng.uniform(*coeff_range),
            rng.uniform(*coeff_range),
            rng.uniform(*coeff_range)
        ])
        
        # Phase 1: Coarse Nelder-Mead with soft-min surrogate
        if time.time() > deadline - 10:
            break
        
        try:
            res_coarse = minimize(
                lambda p: _neg_jittered_score(p, beta=50.0),
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 300}
            )
            
            if -res_coarse.fun > best_score:
                best_score = -res_coarse.fun
                best_params = res_coarse.x.copy()
        except Exception:
            continue
        
        # Phase 2: Fine polish with higher beta
        if time.time() > deadline - 5:
            break
        
        try:
            res_fine = minimize(
                lambda p: _neg_jittered_score(p, beta=200.0),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500}
            )
            
            if -res_fine.fun > best_score:
                best_score = -res_fine.fun
                best_params = res_fine.x.copy()
        except Exception:
            pass
        
        # Phase 3: Exact score polish
        if time.time() > deadline - 2:
            break
        
        try:
            res_exact = minimize(
                lambda p: _neg_jittered_score(p, beta=None),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 300}
            )
            
            if -res_exact.fun > best_score:
                best_score = -res_exact.fun
                best_params = res_exact.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return 0.0, None
    
    return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    best_overall_score = 0.0
    best_overall_points = None

    # Primary strategy: Optimize jittered regular polygon
    jitter_score, jitter_params = _optimize_jittered_polygon(deadline)
    
    if jitter_params is not None:
        jitter_points = _generate_jittered_polygon(jitter_params)
        jitter_exact_score = _calculate_score(jitter_points)
        
        if jitter_exact_score > best_overall_score:
            best_overall_score = jitter_exact_score
            best_overall_points = jitter_points

    # Fallback: Regular 14-gon if jittered optimization failed
    if best_overall_points is None:
        best_overall_points = np.column_stack((_BASE_COS, _BASE_SIN))
        best_overall_score = _calculate_score(best_overall_points)

    return best_overall_points

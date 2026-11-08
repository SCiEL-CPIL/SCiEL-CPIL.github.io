"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _soft_min_area(points, beta=100.0):
    """
    Compute a smooth surrogate for the minimum triangle area.
    Uses log-sum-exp trick for numerical stability.
    """
    n = len(points)
    areas = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v1 = points[j] - points[i]
            # Compute cross product with all other points
            # Actually, we need all triples. Let's be more careful.
    
    # Better approach: compute all triangle areas
    # For 13 points, there are C(13,3) = 286 triangles
    # We can vectorize this
    
    # Get all indices
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    p1 = points[i_idx]
    p2 = points[j_idx]
    p3 = points[k_idx]
    
    # Triangle area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    # Soft min using log-sum-exp
    # soft_min = -1/beta * log(sum(exp(-beta * areas)))
    # For numerical stability, subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -log_sum_exp / beta
    
    return soft_min

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on sorted points
        # This is a rough fallback
        angles = np.arctan2(points[:, 1] - points[:, 1].mean(), 
                           points[:, 0] - points[:, 0].mean())
        order = np.argsort(angles)
        sorted_points = points[order]
        x = sorted_points[:, 0]
        y = sorted_points[:, 1]
        area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + 
                           x[-1] * y[0] - x[0] * y[-1])
        return area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    n = len(points)
    # Compute all triangle areas
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    p1 = points[i_idx]
    p2 = points[j_idx]
    p3 = points[k_idx]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    
    if hull_area < 1e-10:
        return 0.0
    
    return min_area / hull_area

def _fourier_to_points(params):
    """
    Convert Fourier parameters to point coordinates.
    params: array of shape (m,) where m is the number of Fourier coefficients
    We use a low-dimensional Fourier series to deform a regular 13-gon.
    
    The idea: start with a regular 13-gon inscribed in a circle, then add
    small perturbations using a few Fourier modes to preserve convexity.
    
    params[0] = radius scale
    params[1:] = Fourier coefficients for radial deformation
    """
    n = N
    theta = 2 * np.pi * np.arange(n) / n
    
    # Base: regular polygon
    r_base = 1.0
    
    # Add radial deformation using Fourier modes
    # Use only a few low-frequency modes to keep the space small
    # params[1] = a1, params[2] = b1, params[3] = a2, params[4] = b2, etc.
    # r(theta) = 1 + sum(a_k * cos(k*theta) + b_k * sin(k*theta))
    
    # Limit to first 3 harmonics (6 coefficients) + radius scale = 7 params
    # This is a low-dimensional parameterization
    
    r = r_base * (1.0 + params[0])  # Scale
    
    # Add Fourier deformation
    # Use up to 3 harmonics
    for k in range(1, 4):
        a_k = params[2 * k] if 2 * k < len(params) else 0.0
        b_k = params[2 * k + 1] if 2 * k + 1 < len(params) else 0.0
        r += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
    
    # Ensure r is positive
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = np.column_stack([x, y])
    
    # Center the points
    points -= points.mean(axis=0)
    
    return points

def construct():
    """
    Construct 13 points in a planar convex region to maximize the minimum
    triangle area divided by the convex hull area.
    
    Strategy:
    1. Use Fourier series parameterization of a deformed regular 13-gon
    2. Optimize using differential evolution
    3. Refine using Nelder-Mead on the soft-min objective
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Number of Fourier parameters
    # params[0] = radius scale, params[1:7] = Fourier coefficients (3 harmonics)
    n_params = 7
    
    # Bounds for the parameters
    # Radius scale: keep it reasonable
    # Fourier coefficients: small to preserve convexity
    bounds = [(-0.3, 0.3)] + [(-0.1, 0.1)] * 6
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Differential evolution
    if time.time() < deadline - 60:
        def objective(params):
            points = _fourier_to_points(params)
            # Use soft-min as a smooth surrogate for optimization
            return -_soft_min_area(points, beta=50.0)
        
        try:
            result = differential_evolution(
                objective, bounds,
                seed=N,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                mutation=(0.3, 0.7),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            best_params = result.x
            best_score = _score(_fourier_to_points(best_params))
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead refinement on the actual score
    if best_params is not None and time.time() < deadline - 20:
        def actual_objective(params):
            points = _fourier_to_points(params)
            return -_score(points)
        
        try:
            result = minimize(
                actual_objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            score_nm = -result.fun
            if score_nm > best_score:
                best_score = score_nm
                best_params = result.x
        except Exception:
            pass
    
    # Phase 3: Coordinate descent refinement
    if best_params is not None and time.time() < deadline - 5:
        current_params = best_params.copy()
        current_score = best_score
        
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(n_params):
                # Try small perturbations in each coordinate
                for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                    trial_params = current_params.copy()
                    trial_params[i] += delta
                    
                    # Check bounds
                    if trial_params[i] < bounds[i][0] or trial_params[i] > bounds[i][1]:
                        continue
                    
                    points = _fourier_to_points(trial_params)
                    score = _score(points)
                    
                    if score > current_score + 1e-10:
                        current_params = trial_params
                        current_score = score
                        improved = True
            
            if not improved:
                break
        
        best_score = current_score
        best_params = current_params
    
    # Convert best parameters to points
    if best_params is not None:
        points = _fourier_to_points(best_params)
    else:
        # Fallback: regular 13-gon
        theta = 2 * np.pi * np.arange(N) / N
        points = np.column_stack([np.cos(theta), np.sin(theta)])
    
    return points

# EVOLVE-BLOCK-END

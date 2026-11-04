import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

N = 14
SOFT_BETA = 50.0

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta=SOFT_BETA):
    """
    Smooth surrogate: soft_min of triangle areas / hull area.
    Returns a smooth value that approximates min_area/hull_area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft-min via log-sum-exp: soft_min(a) ≈ (1/beta) * log(sum(exp(-beta * a)))
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_min_val / hull_area

def _fourier_to_points(params):
    """
    Convert Fourier parameters to point coordinates.
    params: [a0, a1, b1, a2, b2, ...]
    Radius r_k = a0 + a1*cos(theta_k) + b1*sin(theta_k) + a2*cos(2*theta_k) + b2*sin(2*theta_k)
    """
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    a0 = params[0]
    a1 = params[1]
    b1 = params[2]
    a2 = params[3]
    b2 = params[4]
    
    # We can add higher modes if needed, but 2 modes (5 params) is a good start.
    # Let's stick to 5 params for now to keep DE efficient.
    
    radii = a0 + a1 * np.cos(angles) + b1 * np.sin(angles) + \
            a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)
    
    # Ensure radii are positive to avoid degenerate points
    # If any radius is negative, the point flips to the opposite side, which might be okay
    # but can cause issues with convex hull if points cross.
    # Clipping to a small positive value is safer.
    radii = np.maximum(radii, 0.01)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack((x, y))

def _fourier_score(params):
    """
    Objective function for Fourier parameterization.
    Returns negative soft-min score for minimization.
    """
    points = _fourier_to_points(params)
    return -_soft_min_score(points)

def _fourier_exact_score(params):
    """
    Objective function for Fourier parameterization using exact score.
    """
    points = _fourier_to_points(params)
    return -_calculate_score(points)

def _optimize_fourier(deadline):
    """
    Optimize the Fourier parameterization.
    """
    # Initial bounds
    # a0 (mean radius) ~ 1
    # a1, b1 (dipole) ~ 0.5
    # a2, b2 (quadrupole) ~ 0.3
    
    bounds = [
        (0.5, 1.5),  # a0
        (-0.8, 0.8), # a1
        (-0.8, 0.8), # b1
        (-0.5, 0.5), # a2
        (-0.5, 0.5)  # b2
    ]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Differential Evolution
    # Time check
    if time.time() > deadline - 60:
        return 0.0, None
    
    try:
        res = differential_evolution(
            _fourier_score,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-5,
            polish=True,
            # We use soft-min here for smoothness
        )
        
        if res.success:
            best_params = res.x
            best_score = -res.fun # This is soft-min score
            # Get exact score for comparison
            exact_score = _calculate_score(_fourier_to_points(best_params))
            best_score = exact_score
    except Exception:
        pass
    
    # Phase 2: Local Polish with L-BFGS-B on soft-min
    if best_params is None:
        # Fallback start
        best_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    if time.time() < deadline - 20:
        try:
            res_lbfgs = minimize(
                _fourier_score,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            polished_params = res_lbfgs.x
            polished_score = _calculate_score(_fourier_to_points(polished_params))
            
            if polished_score > best_score:
                best_score = polished_score
                best_params = polished_params
        except Exception:
            pass
    
    # Phase 3: Coordinate-level polish
    # We have the best points from Fourier params.
    # Now optimize the 28 coordinates directly to escape the Fourier constraint.
    best_points = _fourier_to_points(best_params)
    
    if time.time() < deadline - 10:
        def neg_soft_score_coords(x):
            pts = x.reshape(-1, 2)
            return _soft_min_score(pts, beta=SOFT_BETA)
        
        try:
            res_coords = minimize(
                neg_soft_score_coords,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            polished_coords = res_coords.x.reshape(-1, 2)
            polished_coords_score = _calculate_score(polished_coords)
            
            if polished_coords_score > best_score:
                best_score = polished_coords_score
                best_points = polished_coords
        except Exception:
            pass
        
        # Second polish with higher beta
        if time.time() < deadline - 5:
            def neg_soft_score_coords_high(x):
                pts = x.reshape(-1, 2)
                return _soft_min_score(pts, beta=SOFT_BETA * 5)
            
            try:
                res_coords2 = minimize(
                    neg_soft_score_coords_high,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                polished_coords2 = res_coords2.x.reshape(-1, 2)
                polished_coords2_score = _calculate_score(polished_coords2)
                
                if polished_coords2_score > best_score:
                    best_score = polished_coords2_score
                    best_points = polished_coords2
            except Exception:
                pass
    
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Try Fourier parameterization first
    score, points = _optimize_fourier(deadline)
    if points is not None:
        best_overall_score = score
        best_overall_points = points
    
    # If we have time, we could try the ring approach as a second candidate
    # but the Fourier approach is more general and likely better.
    # Let's stick to just Fourier to ensure it finishes comfortably and uses time for polishing.
    
    # Fallback if no valid points found
    if best_overall_points is None:
        # Return a regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_overall_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_overall_points

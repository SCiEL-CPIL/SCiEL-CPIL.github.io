"""Program for eight planar points maximizing the product of distances at fixed diameter."""
import numpy as np
from scipy.optimize import minimize
from itertools import product

# EVOLVE-BLOCK-START
def _score_from_points(points):
    """Compute Delta / n^n for a set of n points, scaled to diameter 2."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    
    # Compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    
    # Diameter is the max pairwise distance
    diameter = np.max(dists)
    if diameter < 1e-15:
        return 0.0
    
    # Scale so diameter is 2
    scale = 2.0 / diameter
    dists_scaled = dists * scale
    
    # Product over ordered pairs (i != j)
    # For n points, there are n*(n-1) ordered pairs
    # Delta = product of all ordered pair distances
    # log(Delta) = sum over i != j of log(d_ij)
    mask = ~np.eye(n, dtype=bool)
    log_delta = np.sum(np.log(dists_scaled[mask]))
    
    # Score = Delta / n^n = exp(log_delta) / n^n = exp(log_delta - n*log(n))
    score = np.exp(log_delta - n * np.log(n))
    
    return float(score)

def _generate_two_squares_x_theta(x, theta):
    """Generate 8 points: two concentric squares.
    Square 1: radius 1, angles [0, pi/2, pi, 3pi/2]
    Square 2: radius x, angles [theta, pi/2+theta, pi+theta, 3pi/2+theta]
    """
    angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    pts1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    
    angles2 = angles1 + theta
    pts2 = x * np.column_stack([np.cos(angles2), np.sin(angles2)])
    
    return np.vstack([pts1, pts2])

def _negative_score_2d(params):
    """Negative score for minimization, taking [x, theta]."""
    x, theta = params
    if x < 1e-6:
        return 1e10
    # Symmetry: theta in [0, pi/4]
    # We can map theta to [0, pi/4] if needed, but optimizer might go outside.
    # Let's just evaluate. The function is periodic/symmetric.
    points = _generate_two_squares_x_theta(x, theta)
    return -_score_from_points(points)

def construct():
    """Construct 8 points maximizing the product of distances at fixed diameter.
    
    Uses two concentric squares with ratio x = r2/r1 and relative rotation theta.
    Performs global grid search followed by local refinement.
    """
    best_score = -np.inf
    best_params = None
    
    # Global grid search over x and theta
    # x = r2/r1. By symmetry x and 1/x are equivalent (swapping squares).
    # So we can restrict x >= 1.
    # theta in [0, pi/4] due to square symmetry.
    
    # Coarse grid
    x_vals = np.linspace(0.5, 2.0, 50) # We will handle x<1 by symmetry or just search wide
    theta_vals = np.linspace(0, np.pi/4, 50)
    
    for x, theta in product(x_vals, theta_vals):
        # Ensure x >= 1 to reduce redundancy? 
        # Actually, _generate_two_squares_x_theta(x, theta) is not symmetric in x -> 1/x directly without swapping points.
        # But the set of points {Sq(1), Sq(x, theta)} is the same as {Sq(x), Sq(1, -theta/x?)}? No.
        # Let's just search x in [0.1, 5.0] and theta in [0, pi/4].
        points = _generate_two_squares_x_theta(x, theta)
        score = _score_from_points(points)
        if score > best_score:
            best_score = score
            best_params = (x, theta)
            
    # Also try the case where the "inner" square is the one with radius 1? 
    # The parameterization fixes Square 1 radius to 1. 
    # If the optimal solution has r2 < r1, x < 1.
    # My grid covers x from 0.5 to 2.0. This should be enough.
    # Let's extend x range slightly to be safe.
    x_vals_fine = np.linspace(0.2, 5.0, 100)
    theta_vals_fine = np.linspace(0, np.pi/4, 100)
    
    for x, theta in product(x_vals_fine, theta_vals_fine):
        points = _generate_two_squares_x_theta(x, theta)
        score = _score_from_points(points)
        if score > best_score:
            best_score = score
            best_params = (x, theta)

    # Local refinement using scipy.optimize
    if best_params is not None:
        initial = np.array(best_params)
        # Use Powell for robustness without gradients
        result = minimize(
            _negative_score_2d,
            initial,
            method='Powell',
            options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 1000, 'maxfev': 10000}
        )
        if result.fun < -best_score:
            best_params = result.x
            best_score = -result.fun
            
    # Generate final points
    x, theta = best_params
    points = _generate_two_squares_x_theta(x, theta)
    
    # Ensure distinct points
    # Check for duplicates and perturb if necessary
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                points[j][0] += 1e-8
    
    # Convert to list of lists of floats
    return [[float(p[0]), float(p[1])] for p in points]
# EVOLVE-BLOCK-END

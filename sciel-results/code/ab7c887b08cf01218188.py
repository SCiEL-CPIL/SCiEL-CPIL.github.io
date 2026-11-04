"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

def _triangle_area(p1, p2, p3):
    """Calculate area of triangle formed by p1, p2, p3."""
    # Using cross product magnitude / 2
    # p1, p2, p3 are (2,) arrays
    v1 = p2 - p1
    v2 = p3 - p1
    # In 2D, cross product is scalar: x1*y2 - y1*x2
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return 0.5 * abs(cross)

def _min_triangle_area(points):
    """Calculate minimum triangle area for all triples."""
    n = len(points)
    min_area = float('inf')
    # Pre-compute or just loop. N=14 is small.
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            for k in range(j + 1, n):
                p3 = points[k]
                area = _triangle_area(p1, p2, p3)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Calculate area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback if points are collinear or degenerate
        return 1.0

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _generate_points_from_params(params):
    """
    Generate 14 points with 7-fold rotational symmetry.
    Parameters: [r1, phi1]
    r0 is fixed to 1.0.
    P0 is at (1.0, 0).
    P1 is at (r1 * cos(phi1), r1 * sin(phi1)).
    The set consists of 7 rotations of P0 and 7 rotations of P1.
    """
    r1 = params[0]
    phi1 = params[1]
    
    r0 = 1.0
    phi0 = 0.0
    
    p0 = np.array([r0 * np.cos(phi0), r0 * np.sin(phi0)])
    p1 = np.array([r1 * np.cos(phi1), r1 * np.sin(phi1)])
    
    points = []
    step = 2 * np.pi / 7
    for k in range(7):
        angle = k * step
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        points.append(rot @ p0)
        points.append(rot @ p1)
        
    return np.array(points)

def _neg_score(params):
    """Negative score for minimization."""
    points = _generate_points_from_params(params)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for 120s limit
    
    # Define bounds for Differential Evolution
    # r1: radius of inner/second ring. Can be 0 to 2.
    # phi1: angle of second point in the fundamental sector.
    # Symmetry implies we can restrict phi1 to [0, 2*pi/7].
    # If phi1 is close to 0, P1 is close to P0.
    # If phi1 is pi/7, and r1=1, we get regular 14-gon.
    
    bounds = [(0.1, 2.0), (0.0, 2 * np.pi / 7)]
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    # 1. Try Regular 14-gon explicitly
    reg_params = [1.0, np.pi / 7]
    reg_points = _generate_points_from_params(reg_params)
    reg_score = _score(reg_points)
    best_score = reg_score
    best_params = reg_params
    best_points = reg_points
    
    # 2. Run Differential Evolution
    if time.time() < deadline:
        try:
            result = differential_evolution(
                _neg_score,
                bounds,
                maxiter=200,
                popsize=15,
                tol=0.001,
                seed=42,
                polish=True
            )
            
            if result.fun < 0:
                de_score = -result.fun
                if de_score > best_score:
                    best_score = de_score
                    best_params = result.x
                    best_points = _generate_points_from_params(best_params)
        except Exception:
            pass
            
    # 3. Polish with Nelder-Mead if time permits
    if time.time() < deadline and best_params is not None:
        try:
            nm_result = minimize(
                _neg_score,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            nm_score = -nm_result.fun
            if nm_score > best_score:
                best_score = nm_score
                best_params = nm_result.x
                best_points = _generate_points_from_params(best_params)
        except Exception:
            pass

    # 4. Final check and return
    if best_points is None:
        # Fallback to random if something went wrong
        rng = np.random.RandomState(N)
        best_points = rng.rand(N, 2)
        
    return best_points
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

def _min_triangle_area_normalized(points):
    """
    Compute the minimum area of any triangle formed by 3 points,
    normalized by the convex hull area.
    """
    n = len(points)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Find minimum triangle area
    # Generate all combinations of 3 indices
    # For N=14, this is 364 combinations
    idx = np.array(list(itertools.combinations(range(n), 3)))
    
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective(params):
    """
    Objective function for optimization: minimize negative of min normalized triangle area.
    """
    angles = params
    # Wrap angles to [0, 2*pi) to handle periodicity in evaluation
    # Note: Nelder-Mead operates on R^N, so we just evaluate the score
    # for the given angles.
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    score = _min_triangle_area_normalized(points)
    return -score

def construct():
    """
    Construct N points in a planar convex region to maximize the minimum
    triangle area normalized by convex hull area.
    """
    best_points = None
    best_score = -1
    
    # Multi-start strategy
    # 1. Regular polygon
    init1 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # 2. Perturbed regular polygon
    np.random.seed(42)
    init2 = init1 + np.random.uniform(-0.1, 0.1, N)
    
    # 3. Another perturbation
    init3 = init1 + np.random.uniform(-0.2, 0.2, N)
    
    # 4. A more aggressive perturbation
    init4 = init1 + np.random.uniform(-0.5, 0.5, N)
    
    inits = [init1, init2, init3, init4]
    
    for init in inits:
        # Ensure angles are in [0, 2*pi)
        init = np.mod(init, 2 * np.pi)
        
        result = minimize(
            _objective,
            init,
            method='Nelder-Mead',
            options={
                'maxiter': 10000,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True
            }
        )
        
        angles = result.x
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        score = _min_triangle_area_normalized(points)
        
        if score > best_score:
            best_score = score
            best_points = points
            
    return best_points
# EVOLVE-BLOCK-END

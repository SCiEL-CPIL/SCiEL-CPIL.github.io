import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all unique triples (i, j, k) with i < j < k
# There are C(14, 3) = 364 triangles
_i, _j, _k = np.indices((N, N, N))
_mask = (_i < _j) & (_j < _k)
_IDX_I = _i[_mask]
_IDX_J = _j[_mask]
_IDX_K = _k[_mask]

def _score_from_points(points):
    """
    Compute the score: min triangle area / convex hull area.
    points: (N, 2) array.
    """
    # Calculate areas of all triangles
    p_i = points[_IDX_I]
    p_j = points[_IDX_J]
    p_k = points[_IDX_K]
    
    # 2 * Area = |cross(P_j - P_i, P_k - P_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_area = np.min(areas)
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # If hull fails, points are likely collinear or degenerate
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit to be safe
    
    # We optimize 2N - 4 = 24 variables.
    # Fix P[0] = [0, 0] and P[1] = [1, 0] to remove translation, rotation, and scale invariance.
    # The remaining N-2 = 12 points have 24 coordinates.
    
    def objective(x_flat):
        # Reconstruct points
        points = np.zeros((N, 2))
        points[0] = [0.0, 0.0]
        points[1] = [1.0, 0.0]
        # x_flat has length 24, reshape to (12, 2)
        points[2:] = x_flat.reshape(-1, 2)
        
        # Calculate score
        return -_score_from_points(points)
    
    best_score = 0.0
    best_points = None
    
    # Generate initial guesses for the 12 free points
    # We want to spread them out.
    # Guess 1: Points on a circle
    # Guess 2: Random points in a square
    # Guess 3: Jittered grid
    
    rng = np.random.default_rng(42)
    
    initial_guesses = []
    
    # 1. Regular polygon-like distribution
    # Place points roughly on a circle of radius 1 centered at (0.5, 0)
    # Angles from 0 to 2*pi
    angles = np.linspace(0, 2*np.pi, N-2, endpoint=False)
    radii = np.ones(N-2)
    init1 = np.column_stack([0.5 + radii * np.cos(angles), radii * np.sin(angles)])
    initial_guesses.append(init1.flatten())
    
    # 2. Random points in a box
    init2 = rng.uniform(-1, 2, size=(N-2, 2))
    initial_guesses.append(init2.flatten())
    
    # 3. Jittered regular polygon
    init3 = init1 + rng.normal(0, 0.1, size=(N-2, 2))
    initial_guesses.append(init3.flatten())
    
    # 4. Another random set
    init4 = rng.uniform(-1, 2, size=(N-2, 2))
    initial_guesses.append(init4.flatten())
    
    # 5. Points on two lines? No, convex region implies spread.
    # Let's add a few more random restarts
    
    for _ in range(10):
        init = rng.uniform(-1, 2, size=(N-2, 2))
        initial_guesses.append(init.flatten())
        
    # Evaluate initial guesses to pick the best starting point for the first optimization
    # Or just run optimization on each.
    
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-8})
            
            if res.success or res.fun < 0: # fun is -score, so lower is better
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    # Reconstruct best points
                    best_points = np.zeros((N, 2))
                    best_points[0] = [0.0, 0.0]
                    best_points[1] = [1.0, 0.0]
                    best_points[2:] = res.x.reshape(-1, 2)
                    
        except Exception as e:
            pass
            
    if best_points is None:
        # Fallback to a simple regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # The evaluator expects the points.
    # The normalization (min_area / hull_area) is done by the evaluator.
    # Our optimization maximized this ratio.
    # We can return the points as is.
    return best_points

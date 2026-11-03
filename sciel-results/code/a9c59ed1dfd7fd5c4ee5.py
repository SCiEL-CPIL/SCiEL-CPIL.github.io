import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
SQRT3 = np.sqrt(3)
AREA_BIG = SQRT3 / 4.0  # area of unit equilateral triangle

# Precompute all triples for fast area computation
import itertools
_TRIPLES = list(itertools.combinations(range(N), 3))
_T_IDX = np.array(_TRIPLES)  # shape (165, 3)
_NUM_TRIPLES = len(_TRIPLES)

def _min_triangle_area_fast(pts):
    """
    pts: array of shape (N, 2) in Cartesian coords.
    Returns the minimum area of any triangle formed by 3 points.
    Area = 0.5 * |cross product|
    """
    # For each triple (i,j,k), compute area
    pi = pts[_T_IDX[:, 0]]  # (165, 2)
    pj = pts[_T_IDX[:, 1]]
    pk = pts[_T_IDX[:, 2]]
    
    # Cross product of (pj-pi) and (pk-pi)
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _objective_cartesian(x):
    """
    Objective for minimization: negative of min triangle area.
    x: flat array of shape (2*N,)
    """
    pts = x.reshape(N, 2)
    
    # Check if all points are inside the triangle
    # Triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # A point (x,y) is inside if:
    # y >= 0
    # y <= sqrt(3)*x  (below line AC: y = sqrt(3)*x)
    # y <= -sqrt(3)*(x-1) = sqrt(3)*(1-x)  (below line BC)
    
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    
    # Check boundaries
    if np.any(y_coords < -1e-10) or np.any(y_coords > SQRT3 * (1 - 1e-10)):
        return 1e10  # infeasible
    if np.any(y_coords > SQRT3 * x_coords + 1e-10):
        return 1e10
    if np.any(y_coords > SQRT3 * (1 - x_coords) + 1e-10):
        return 1e10
    
    # Also check x bounds (redundant for equilateral but safe)
    if np.any(x_coords < -1e-10) or np.any(x_coords > 1 + 1e-10):
        return 1e10
    
    min_area = _min_triangle_area_fast(pts)
    return -min_area

def _clip_to_triangle(pts):
    """Clip points to be inside the equilateral triangle."""
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    
    # y >= 0
    y = np.maximum(y, 0)
    
    # y <= sqrt(3) * x  =>  x >= y / sqrt(3)
    x = np.maximum(x, y / SQRT3)
    
    # y <= sqrt(3) * (1 - x)  =>  x <= 1 - y / sqrt(3)
    x = np.minimum(x, 1 - y / SQRT3)
    
    # x >= 0, x <= 1
    x = np.clip(x, 0, 1)
    
    return np.column_stack([x, y])

def _structured_init():
    """Generate a structured initial guess for 11 points."""
    pts = np.zeros((N, 2))
    
    # Three near vertices
    eps = 0.05
    pts[0] = [eps, eps * SQRT3]          # near (0,0)
    pts[1] = [1 - eps, eps * SQRT3]      # near (1,0)
    pts[2] = [0.5, SQRT3/2 - eps * SQRT3]  # near (0.5, sqrt(3)/2)
    
    # Three near edge midpoints
    pts[3] = [0.5, 0.02]                 # near bottom edge midpoint
    pts[4] = [0.25, SQRT3/4 - 0.02]      # near left edge midpoint
    pts[5] = [0.75, SQRT3/4 - 0.02]      # near right edge midpoint
    
    # Center
    pts[6] = [0.5, SQRT3/6]
    
    # Four more points in a ring around center
    r = 0.15
    angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    for i, a in enumerate(angles):
        pts[7 + i] = [0.5 + r * np.cos(a), SQRT3/6 + r * np.sin(a) / SQRT3]
    
    return _clip_to_triangle(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def _time_up():
        return time.time() > deadline - 5.0
    
    rng = np.random.default_rng(42)
    
    best_pts = _structured_init()
    best_area = _min_triangle_area_fast(best_pts)
    
    # Define bounds for differential evolution
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))       # x
        bounds.append((0, SQRT3/2))  # y
    
    # Phase 1: Differential Evolution for global search
    # Use a moderate number of iterations to stay within time budget
    de_result = None
    try:
        if not _time_up():
            de_result = differential_evolution(
                _objective_cartesian,
                bounds=bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=0.01,
                polish=False,
                init='latinhypercube',
                mutation=(0.3, 1.0),
                recombination=0.8
            )
            if de_result.success or de_result.fun < -best_area:
                candidate_pts = _clip_to_triangle(de_result.x.reshape(N, 2))
                candidate_area = _min_triangle_area_fast(candidate_pts)
                if candidate_area > best_area:
                    best_area = candidate_area
                    best_pts = candidate_pts
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B local refinement from multiple starting points
    starting_points = [best_pts]
    
    # Add some perturbed versions for diversity
    for _ in range(5):
        if _time_up():
            break
        perturbed = best_pts + rng.normal(0, 0.05, size=(N, 2))
        perturbed = _clip_to_triangle(perturbed)
        starting_points.append(perturbed)
    
    # Also add the structured init
    starting_points.append(_structured_init())
    
    # Run L-BFGS-B from each starting point
    for start_pts in starting_points:
        if _time_up():
            break
        
        x0 = start_pts.flatten()
        
        try:
            result = minimize(
                _objective_cartesian,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < -best_area:
                candidate_pts = _clip_to_triangle(result.x.reshape(N, 2))
                candidate_area = _min_triangle_area_fast(candidate_pts)
                if candidate_area > best_area:
                    best_area = candidate_area
                    best_pts = candidate_pts
        except Exception:
            continue
    
    # Phase 3: Final fine-tuning with smaller steps
    if not _time_up():
        x0 = best_pts.flatten()
        try:
            result = minimize(
                _objective_cartesian,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < -best_area:
                candidate_pts = _clip_to_triangle(result.x.reshape(N, 2))
                candidate_area = _min_triangle_area_fast(candidate_pts)
                if candidate_area > best_area:
                    best_area = candidate_area
                    best_pts = candidate_pts
        except Exception:
            pass
    
    return best_pts

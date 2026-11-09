import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute indices for speed
indices = [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]

def _min_area(points):
    """
    Calculate the minimum area of any triangle formed by the points.
    points: np.array of shape (N, 2)
    """
    min_a = float('inf')
    for i, j, k in indices:
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * |cross(p2-p1, p3-p1)|
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        if area < min_a:
            min_a = area
    return min_a

def _is_valid(points):
    """
    Check if all points are inside the triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    Inequalities:
    y >= 0
    y <= sqrt(3) * x
    y <= -sqrt(3) * (x - 1)
    """
    x = points[:, 0]
    y = points[:, 1]
    if np.any(y < -1e-9):
        return False
    if np.any(y > np.sqrt(3) * x + 1e-9):
        return False
    if np.any(y > -np.sqrt(3) * (x - 1) + 1e-9):
        return False
    return True

def _objective(x):
    """
    Objective function for differential_evolution.
    Returns negative of min area if valid, large penalty if invalid.
    """
    points = x.reshape(-1, 2)
    if not _is_valid(points):
        return 1e10
    return -_min_area(points)

def _objective_smooth(x):
    """
    Objective function for local refinement (SLSQP).
    We use the same logic but rely on constraints for validity.
    """
    points = x.reshape(-1, 2)
    return -_min_area(points)

def _constraints(x):
    """
    Constraints for SLSQP.
    Returns list of tuples (type, fun)
    """
    points = x.reshape(-1, 2)
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    cons = []
    # y >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        # y - sqrt(3)*x >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - np.sqrt(3)*x[2*i]})
        # -sqrt(3)*(x-1) - y >= 0  => sqrt(3)*(1-x) - y >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: np.sqrt(3)*(1 - x[2*i]) - x[2*i+1]})
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin for 120s limit (or 150s limit)
    
    # Bounds for DE
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))      # x
        bounds.append((0, H))      # y
    
    # Run Differential Evolution
    best_result = None
    try:
        # popsize and maxiter tuned for speed
        # We check time in the callback? DE doesn't support time-based callback easily.
        # We will run a limited number of iterations.
        # 22 dimensions. popsize=15 is default.
        # Let's try maxiter=50 first.
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            popsize=15,
            seed=42,
            tol=1e-4,
            polish=False # We will polish manually
        )
        
        if res.success or res.fun < 1e9:
            best_result = res.x
        else:
            # If DE failed, maybe try a few random restarts?
            # For now, take the best found even if not successful
            best_result = res.x
            
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random
        return _random_points()

    if time.time() > deadline:
        # Return best found so far
        return best_result.reshape(-1, 2)

    # Polish with SLSQP
    try:
        # SLSQP might get stuck in local minima, but starting from DE result is good.
        # Note: SLSQP requires smooth objective. Min area is non-smooth.
        # However, it often works well enough to push the boundaries.
        res2 = minimize(
            _objective_smooth, 
            best_result, 
            method='SLSQP', 
            bounds=bounds,
            constraints=_constraints(best_result),
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        
        if res2.success and res2.fun < best_result and _is_valid(res2.x.reshape(-1, 2)):
            best_result = res2.x
            
    except Exception as e:
        print(f"SLSQP failed: {e}")

    # Final check
    if best_result is None:
        return _random_points()
        
    return best_result.reshape(-1, 2)

def _random_points():
    import random
    rng = random.Random(7)
    points = []
    while len(points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 0.98 and u > 0.01 and v > 0.01:
            points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
    return np.array(points)

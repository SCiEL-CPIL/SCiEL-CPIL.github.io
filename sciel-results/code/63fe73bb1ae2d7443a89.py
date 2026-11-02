import numpy as np
from scipy.optimize import differential_evolution, minimize
import itertools

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def compute_areas(points):
    """Compute areas of all triangles formed by pairs of points."""
    areas = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def min_triangle_area(points):
    """Return the minimum area of any triangle formed by the points."""
    if len(points) < 3:
        return 0.0
    areas = compute_areas(points)
    return np.min(areas)

def is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the reference triangle."""
    x, y = p
    # y >= 0
    if y < -1e-9:
        return False
    # y <= sqrt(3) * x
    if y > np.sqrt(3) * x + 1e-9:
        return False
    # y <= -sqrt(3) * (x - 1)
    if y > -np.sqrt(3) * (x - 1) + 1e-9:
        return False
    return True

def penalty_objective(x):
    """
    Objective function for differential evolution.
    x is a flat array of length 2*N.
    Returns negative of min area + penalty if outside.
    """
    points = x.reshape(N, 2)
    
    # Check bounds and penalty
    penalty = 0.0
    for p in points:
        if not is_inside_triangle(p):
            # Calculate distance to nearest boundary roughly
            # Simpler: just a large penalty
            penalty += 1.0
            
    if penalty > 0:
        return 100.0 + penalty # Large value to discourage

    min_area = min_triangle_area(points)
    return -min_area

def construct():
    # 1. Global Search with Differential Evolution
    # Bounds: x in [-0.1, 1.1], y in [-0.1, 1.1]
    # We use loose bounds and rely on penalty to keep points inside.
    # This allows DE to explore freely.
    
    bounds = [(-0.1, 1.1)] * (2 * N)
    
    # DE parameters
    # popsize=15, maxiter=50 is a good balance for speed and quality
    # seed for reproducibility
    de_result = differential_evolution(
        penalty_objective, 
        bounds, 
        seed=42, 
        maxiter=50, 
        popsize=15,
        tol=1e-4,
        polish=False # We will polish manually
    )
    
    best_x = de_result.x
    best_points = best_x.reshape(N, 2)
    
    # Ensure points are strictly inside for the SLSQP start
    # If any point is on the boundary, nudge it slightly inwards
    for i in range(N):
        if not is_inside_triangle(best_points[i]):
             # Simple projection or nudge
             best_points[i] = best_points[i] * 0.9 + 0.1 * (0.5, np.sqrt(3)/6) # Nudge towards centroid
    
    # 2. Local Polish with SLSQP
    # Variables: x = [p1, ..., p11, t]
    # Minimize -t
    # Subject to:
    # 1. Area(i,j,k) - t >= 0 for all i,j,k
    # 2. Points inside triangle
    
    def slsqp_objective(x):
        points = x[:2*N].reshape(N, 2)
        t = x[2*N]
        return -t

    def slsqp_constraints(x):
        points = x[:2*N].reshape(N, 2)
        t = x[2*N]
        
        cons = []
        
        # Triangle area constraints
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1, p2, p3 = points[i], points[j], points[k]
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    # area - t >= 0
                    cons.append({'type': 'ineq', 'fun': lambda a=area, tt=t: a - tt})
                    
        # Point inside triangle constraints
        # y >= 0
        # y - sqrt(3)x <= 0  =>  sqrt(3)x - y >= 0
        # y + sqrt(3)(x-1) <= 0 => -y - sqrt(3)(x-1) >= 0 => sqrt(3)(1-x) - y >= 0
        
        for i in range(N):
            px, py = points[i]
            # y >= 0
            cons.append({'type': 'ineq', 'fun': lambda pyy=py: pyy})
            # sqrt(3)x - y >= 0
            cons.append({'type': 'ineq', 'fun': lambda px=px, pyy=py: np.sqrt(3)*px - pyy})
            # sqrt(3)(1-x) - y >= 0
            cons.append({'type': 'ineq', 'fun': lambda px=px, pyy=py: np.sqrt(3)*(1-px) - pyy})
            
        return cons

    # Initial guess for SLSQP
    # We need to estimate t. Use the min area from DE result.
    t_init = min_triangle_area(best_points)
    
    x0 = np.zeros(2*N + 1)
    x0[:2*N] = best_points.flatten()
    x0[2*N] = t_init
    
    # SLSQP bounds: points inside triangle, t > 0
    # x, y bounds:
    # x in [0, 1]
    # y in [0, sqrt(3)/2]
    # t in [0, 0.1] (area of unit triangle is sqrt(3)/4 ~ 0.433, min area for 11 pts will be small)
    
    bounds_slsqp = []
    for i in range(N):
        bounds_slsqp.append((0, 1)) # x
        bounds_slsqp.append((0, np.sqrt(3)/2)) # y
    bounds_slsqp.append((0, 0.1)) # t
    
    try:
        res = minimize(
            slsqp_objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds_slsqp, 
            constraints=slsqp_constraints,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if res.success:
            best_points = res.x[:2*N].reshape(N, 2)
        else:
            # Fallback to DE result if SLSQP fails
            pass
            
    except Exception:
        # Fallback to DE result
        pass

    return best_points

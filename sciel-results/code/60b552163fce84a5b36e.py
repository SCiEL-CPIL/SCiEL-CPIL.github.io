import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    pi = points[I]
    pj = points[J]
    pk = points[K]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _penalty(points):
    """
    points: (N, 2)
    Penalty for points outside the triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # y >= 0
    v1 = np.maximum(0, -y)
    # y <= sqrt(3) * x
    v2 = np.maximum(0, y - SQRT3 * x)
    # y <= sqrt(3) * (1 - x)
    v3 = np.maximum(0, y - SQRT3 * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _objective(flat_points):
    points = flat_points.reshape(-1, 2)
    area = _min_area(points)
    pen = _penalty(points)
    # We want to maximize area, minimize penalty
    # DE minimizes. So return -area + penalty
    return -area + 100.0 * pen

def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    clipped = points.copy()
    for i in range(len(clipped)):
        x, y = clipped[i]
        
        # Check and clip against each edge
        # Edge 1: y >= 0
        if y < 0:
            y = 0.0
        
        # Edge 2: y <= sqrt(3) * x
        if y > SQRT3 * x:
            # Project onto the line y = sqrt(3)*x
            y = SQRT3 * x
        
        # Edge 3: y <= sqrt(3) * (1 - x)
        if y > SQRT3 * (1 - x):
            y = SQRT3 * (1 - x)
        
        # After clipping y, we might need to adjust x too
        # Let's do a simple iterative projection
        for _ in range(10):
            x, y = clipped[i]
            changed = False
            
            if y < 0:
                y = 0.0
                changed = True
            
            if y > SQRT3 * x + 1e-12:
                # Need to increase x or decrease y
                # Move along the constraint
                if x < 1e-12:
                    x = 1e-12
                    changed = True
                y = SQRT3 * x
                changed = True
            
            if y > SQRT3 * (1 - x) + 1e-12:
                if 1 - x < 1e-12:
                    x = 1 - 1e-12
                    changed = True
                y = SQRT3 * (1 - x)
                changed = True
            
            if not changed:
                break
        
        clipped[i] = [x, y]
    
    return clipped

def _refine(points, time_budget=5.0):
    """Refine points using Nelder-Mead on the hard objective."""
    start = time.time()
    flat = points.flatten()
    
    try:
        result = minimize(
            _objective,
            flat,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        refined = result.x.reshape(N, 2)
        
        # Ensure points are inside triangle by clipping
        refined = _clip_to_triangle(refined)
        
        elapsed = time.time() - start
        if elapsed < time_budget:
            return refined
    except Exception:
        pass
    
    return points

def _initial_grid():
    """Generate an initial set of points using a grid-like pattern."""
    H = np.sqrt(3) / 2
    points = []
    
    # Use a triangular grid pattern
    # For 11 points, we can use a 3-4-4 pattern or similar
    
    # Let's use a simple approach: place points at vertices and centroid-like positions
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H / 2])
    points.append([0.75, H / 2])
    
    # Interior points - use a smaller triangular grid
    # Centroid
    points.append([0.5, H / 3])
    
    # Additional interior points
    points.append([0.33, H / 3])
    points.append([0.67, H / 3])
    points.append([0.42, 2 * H / 3])
    points.append([0.58, 2 * H / 3])
    
    return np.array(points[:N])

def _random_initial(seed=42):
    """Generate random initial points inside the triangle."""
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(N):
        # Generate random barycentric coordinates
        u = rng.rand()
        v = rng.rand()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        
        # Convert to Cartesian
        x = u + 0.5 * v
        y = np.sqrt(3) / 2 * v
        points.append([x, y])
    
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_points = None
    best_score = -np.inf
    
    def update_best(points):
        nonlocal best_points, best_score
        min_area = _min_area(points)
        if min_area > best_score:
            best_score = min_area
            best_points = points.copy()
    
    # Phase 1: Try multiple initializations with DE
    inits = [_initial_grid(), _random_initial(42), _random_initial(123)]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        
        bounds = [(0, 1), (0, H)] * N
        
        def callback(xk, convergence):
            if time.time() > deadline - 10:
                raise Exception("Timeout")

        try:
            res = differential_evolution(
                _objective, 
                bounds, 
                maxiter=50, 
                popsize=15, 
                tol=1e-4, 
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                polish=False,
                callback=callback
            )
            if res.success or res.fun < float('inf'):
                points = res.x.reshape(N, 2)
                points = _clip_to_triangle(points)
                update_best(points)
        except Exception:
            pass
    
    # Phase 2: Refine the best point found so far
    if best_points is not None:
        if time.time() < deadline - 10:
            refined = _refine(best_points, time_budget=10.0)
            update_best(refined)
    
    if best_points is None:
        # Fallback: simple random or grid
        best_points = _random_initial(42)
    
    return best_points

# EVOLVE-BLOCK-END

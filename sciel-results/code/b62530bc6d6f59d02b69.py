"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, margin=0.0):
    """Check if point p is inside the unit equilateral triangle with some margin."""
    # Barycentric coordinates relative to V0, V1, V2
    # Using the method of checking if all barycentric coordinates are >= margin
    # and <= 1-margin (for interior)
    
    # Area of reference triangle
    area_ref = 0.5 * np.sqrt(3)
    
    # Areas of sub-triangles
    def tri_area(a, b, c):
        return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
    
    a1 = tri_area(p, V1, V2)
    a2 = tri_area(V0, p, V2)
    a3 = tri_area(V0, V1, p)
    
    # Barycentric coordinates
    w0 = a1 / area_ref
    w1 = a2 / area_ref
    w2 = a3 / area_ref
    
    return (w0 >= -margin) and (w1 >= -margin) and (w2 >= -margin) and \
           (w0 <= 1+margin) and (w1 <= 1+margin) and (w2 <= 1+margin)

def _objective(params):
    """Objective function to minimize (negative of min area)."""
    # params: 22 values for 11 points
    points = params.reshape(N, 2)
    
    # Check if all points are inside the triangle
    for i in range(N):
        if not _is_inside_triangle(points[i], margin=0.01):
            return 100.0  # Penalty for being outside
    
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with a reasonable initial configuration
    # Use a regular grid-like pattern
    rng = np.random.RandomState(42)
    
    # Initial guess: spread points evenly
    # Use a hexagonal-like pattern
    points = []
    # 3 rows: 3, 4, 4 points? Or 2, 4, 5? Let's try a structured approach.
    # Actually, let's just use random initialization with DE.
    
    # Define bounds for each coordinate
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, np.sqrt(3)/2.0))  # y
    
    # Use differential evolution for global optimization
    # This is a non-convex problem, so DE is a good choice
    
    best_result = None
    best_score = -np.inf
    
    # First, try a few random initializations
    for trial in range(5):
        if time.time() > deadline - 30.0:
            break
            
        # Random initialization
        x_init = rng.uniform(0.05, 0.95, N)
        y_init = rng.uniform(0.05, np.sqrt(3)/2.0 * 0.95, N)
        params_init = np.column_stack([x_init, y_init]).flatten()
        
        # Run differential evolution
        try:
            result = differential_evolution(
                _objective,
                bounds,
                x0=params_init,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42 + trial,
                polish=True
            )
            if time.time() > deadline - 10.0:
                break
                
            score = -result.fun
            if score > best_score and result.fun < 50.0:  # Valid result
                best_score = score
                best_result = result.x.reshape(N, 2)
                
        except Exception:
            continue
    
    # If DE didn't find a good solution, try a simpler approach
    if best_result is None:
        # Use a structured grid
        # 11 points: try to place them in a triangular lattice pattern
        # Row 0: 3 points, Row 1: 4 points, Row 2: 4 points
        points = []
        # Scale factor to fit within the triangle
        # The triangle has height sqrt(3)/2
        h = np.sqrt(3) / 2.0
        
        # Row 0 (bottom): y = h/4, x from 0.25 to 0.75
        for i in range(3):
            x = 0.25 + i * 0.25
            y = h / 4.0
            points.append([x, y])
        
        # Row 1: y = h/2, x from 0.125 to 0.875
        for i in range(4):
            x = 0.125 + i * 0.25
            y = h / 2.0
            points.append([x, y])
        
        # Row 2: y = 3*h/4, x from 0.25 to 0.75
        for i in range(4):
            x = 0.25 + i * 0.25
            y = 3 * h / 4.0
            points.append([x, y])
        
        best_result = np.array(points)
        best_score = _min_triangle_area(best_result)
    
    # Verify the result is inside the triangle
    # Clip points that might be slightly outside
    for i in range(N):
        if not _is_inside_triangle(best_result[i], margin=0.0):
            # Move towards centroid
            centroid = np.array([0.5, np.sqrt(3)/6.0])
            best_result[i] = 0.9 * best_result[i] + 0.1 * centroid
    
    return best_result
# EVOLVE-BLOCK-END

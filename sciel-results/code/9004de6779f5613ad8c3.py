"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

N = 11
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])

# EVOLVE-BLOCK-START
def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three points."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, v0, v1, v2):
    """Check if point p is inside or on the boundary of triangle v0-v1-v2."""
    # Using barycentric coordinates
    v0 = np.array(v0)
    v1 = np.array(v1)
    v2 = np.array(v2)
    p = np.array(p)
    
    # Compute barycentric coordinates
    v01 = v1 - v0
    v02 = v2 - v0
    vp = p - v0
    
    # Solve for barycentric coordinates
    # p = v0 + u * v01 + v * v02
    # [v01, v02] [u, v]^T = vp
    A = np.column_stack((v01, v02))
    try:
        bary = np.linalg.solve(A, vp)
    except np.linalg.LinAlgError:
        return False
    
    u, v = bary
    # Check if u, v are in [0, 1] and u + v <= 1
    return (u >= -1e-9) and (v >= -1e-9) and (u + v <= 1 + 1e-9)

def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle (approximate)."""
    # For simplicity, we'll just check and project if needed
    # This is a rough projection; for better results, use proper projection
    clamped = []
    for p in points:
        if _is_inside_triangle(p, V0, V1, V2):
            clamped.append(p)
        else:
            # Simple projection: move towards centroid
            centroid = (V0 + V1 + V2) / 3.0
            # Move 50% towards centroid
            new_p = 0.5 * p + 0.5 * centroid
            clamped.append(new_p)
    return np.array(clamped)

def _objective(x):
    """Objective function for optimization (minimize negative min area)."""
    points = x.reshape(-1, 2)
    # Clamp points to triangle
    points = _clamp_to_triangle(points)
    # Compute min triangle area
    min_area = _min_triangle_area(points)
    # We want to maximize min_area, so minimize -min_area
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Initial guess: use a grid-like pattern or random points
    def _initial_guess():
        # Use a simple grid-like pattern inside the triangle
        points = []
        # Add vertices
        points.append(V0.copy())
        points.append(V1.copy())
        points.append(V2.copy())
        # Add some interior points
        for _ in range(N - 3):
            u, v = np.random.random(2)
            while u + v > 1.0:
                u, v = np.random.random(2)
            p = u * V0 + v * V1 + (1 - u - v) * V2
            points.append(p)
        return np.array(points)
    
    # Run differential evolution for global search
    try:
        # Define bounds for each coordinate
        # x in [0, 1], y in [0, sqrt(3)/2]
        bounds = [(0.0, 1.0), (0.0, np.sqrt(3) / 2.0)] * (N)
        
        # Check time
        if time.time() > deadline - 30.0:
            # Not enough time for DE, just use initial guess
            points = _initial_guess()
            return points
        
        # Run differential evolution
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            popsize=15,
            init='latinhypercube',
            polish=True
        )
        
        if result.success or result.fun < 0:  # result.fun is -min_area, so < 0 means positive area
            best_points = result.x.reshape(-1, 2)
            best_score = -result.fun
        
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline - 10.0:
        if best_points is not None:
            return best_points
        return _initial_guess()
    
    # Local refinement using L-BFGS-B
    if best_points is not None:
        try:
            result_local = minimize(
                _objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0), (0.0, np.sqrt(3) / 2.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if result_local.fun < best_score * -1:  # result_local.fun is -min_area
                best_points = result_local.x.reshape(-1, 2)
                best_score = -result_local.fun
        except Exception as e:
            pass
    
    # Final check
    if best_points is not None:
        return best_points
    else:
        return _initial_guess()

# EVOLVE-BLOCK-END

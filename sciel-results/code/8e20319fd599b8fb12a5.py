"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by AREA."""
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for speed
    # Area = 0.5 * | (pj - pi) x (pk - pi) |
    # We can compute all areas for a fixed i by vectorizing j and k
    for i in range(n):
        pi = points[i]
        # Vectors from i to all other points
        diffs = points - pi
        # Cross product z-component for all pairs (j, k) where j != i, k != i
        # Actually, we only need j > i, k > j for unique triangles, but for min area
        # we can just check all pairs involving i and one other point? No, triangle area
        # involves 3 points.
        # Let's stick to the triple loop but vectorize the inner part if possible.
        # For n=11, C(11,3) = 165 triangles. This is small enough for Python loops.
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * | (pj - pi) x (pk - pi) |
                v1 = pj - pi
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area / AREA  # normalize by triangle area


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    # y <= sqrt(3)/2 * (1 - |x - 0.5|) ... actually the triangle is:
    # y >= 0, y <= sqrt(3)*x (for x in [0, 0.5]), y <= sqrt(3)*(1-x) (for x in [0.5, 1])
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9


def _points_to_params(points):
    """Convert points to a flat parameter vector."""
    return points.flatten()


def _params_to_points(params):
    """Convert flat parameter vector to points."""
    return params.reshape(N, 2)


def _objective(params):
    """
    Objective function for minimization.
    We want to MAXIMIZE the min triangle area.
    So we MINIMIZE -min_triangle_area.
    To handle the non-smooth min, we use a soft-min approximation if needed, 
    but L-BFGS-B can often handle the kink if initialized well.
    Alternatively, we can maximize the sum of the smallest few areas to smooth it out.
    Here, we use a simple negative min area.
    """
    points = _params_to_points(params)
    
    # Check constraints: all points must be inside the triangle
    # If a point is outside, we penalize heavily.
    for p in points:
        if not _is_inside_triangle(p):
            return 1e6  # Large penalty
    
    min_area = _min_triangle_area(points)
    return -min_area


def _soft_min_objective(params, temperature=0.01):
    """
    Objective using soft-min to make it differentiable.
    min(A) approx -1/T * log(sum(exp(-T*a_i)))
    """
    points = _params_to_points(params)
    
    for p in points:
        if not _is_inside_triangle(p):
            return 1e6
    
    # Calculate all triangle areas
    areas = []
    n = N
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                pi = points[i]
                pj = points[j]
                pk = points[k]
                v1 = pj - pi
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    
    areas = np.array(areas)
    # Soft min
    # To avoid overflow, subtract max
    max_area = np.max(areas)
    log_sum_exp = max_area + np.log(np.sum(np.exp(-areas + max_area)))
    soft_min = -log_sum_exp / temperature  # Wait, soft min of A is approx min(A)
    # Actually: min(A) ~ -1/T * log(sum(exp(-T*a_i)))
    # Let's use: -1/T * log(sum(exp(-a_i/T)))
    # This is the log-sum-exp trick.
    # Let's just use the standard negative min for simplicity first, 
    # but soft min is smoother.
    
    # Let's use the standard soft min formula:
    # m = -1/T * log( sum( exp(-a_i / T) ) )
    # If T is small, this approaches min(a_i).
    
    scaled_areas = areas / temperature
    # Shift for stability
    max_scaled = np.max(scaled_areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(-scaled_areas + max_scaled)))
    soft_min_val = -log_sum
    
    return -soft_min_val


def _generate_structured_initial():
    """
    Generate a structured initial configuration.
    For Heilbronn triangles, a good start is often a grid or hexagonal packing.
    Let's try a grid inside the triangle.
    """
    points = []
    # Add vertices
    points.append(V0.copy())
    points.append(V1.copy())
    points.append(V2.copy())
    
    # Add centroid
    centroid = (V0 + V1 + V2) / 3.0
    points.append(centroid)
    
    # Add points along the edges or in a grid
    # Let's try a simple grid of 3x3 points, filtered to be inside
    # And then pick the best 11? No, we need exactly 11.
    # Let's just add random points near the center for the rest.
    rng = random.Random(42)
    while len(points) < N:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            # Check if it's distinct enough from existing points
            distinct = True
            for existing in points:
                if np.linalg.norm(p - existing) < 0.05:
                    distinct = False
                    break
            if distinct:
                points.append(p)
    
    return np.array(points)


def _optimize_from_initial(initial_points, deadline):
    """
    Run scipy optimization from a given initial configuration.
    """
    x0 = _points_to_points(initial_points)
    
    # Define bounds for each coordinate
    # x in [0, 1], y in [0, sqrt(3)/2]
    # But we have the triangle constraint. 
    # L-BFGS-B handles box constraints. The triangle constraint is non-box.
    # We will rely on the penalty function in the objective if a point goes outside.
    # However, it's better to project points back into the triangle if they go out.
    # Or use a transformation. 
    # For now, let's use the penalty approach.
    
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    # Use L-BFGS-B
    try:
        result = minimize(
            _objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        # Check if it finished before deadline
        if time.time() > deadline:
            return None, None
            
        final_points = _params_to_points(result.x)
        
        # Verify all points are inside
        valid = True
        for p in final_points:
            if not _is_inside_triangle(p):
                valid = False
                break
        
        if not valid:
            return None, None
            
        min_area = _min_triangle_area(final_points)
        return final_points, min_area
        
    except Exception as e:
        return None, None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    best_points = None
    best_min = 0.0
    
    # Strategy 1: Structured initial guess + Gradient Descent
    initial = _generate_structured_initial()
    opt_points, opt_min = _optimize_from_initial(initial, deadline)
    
    if opt_points is not None and opt_min > best_min:
        best_min = opt_min
        best_points = opt_points.copy()
        
    # Strategy 2: Random restarts with Gradient Descent
    rng = random.Random(123)
    
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Generate random initial points
        points = []
        while len(points) < N:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                p = V0 * (1 - u - v) + V1 * u + V2 * v
                points.append(p)
        
        points = np.array(points)
        
        opt_points, opt_min = _optimize_from_initial(points, deadline)
        
        if opt_points is not None and opt_min > best_min:
            best_min = opt_min
            best_points = opt_points.copy()
            
    # Strategy 3: If we have time, do a local refinement of the best solution
    # using a more precise local search (like the hill climbing from the original, 
    # but starting from the gradient-optimized solution)
    if best_points is not None and time.time() < deadline - 10:
        # Simple local refinement
        step_sizes = [0.01, 0.005, 0.001]
        for ss in step_sizes:
            if time.time() > deadline:
                break
            # Try moving each point slightly
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(N):
                    orig = best_points[i].copy()
                    for angle in [a * 2 * np.pi / 8 for a in range(8)]:
                        for dist in [ss, ss*0.5]:
                            new_p = best_points[i] + np.array([dist * np.cos(angle), dist * np.sin(angle)])
                            if _is_inside_triangle(new_p):
                                candidate = best_points.copy()
                                candidate[i] = new_p
                                new_min = _min_triangle_area(candidate)
                                if new_min > best_min:
                                    best_min = new_min
                                    best_points = candidate
                                    improved = True
                                    break
                            # Break inner loop if improved? No, check all
                    if improved:
                        break
    
    # Final check
    if best_points is None:
        # Fallback to a fixed configuration
        best_points = _generate_structured_initial()
        
    return best_points

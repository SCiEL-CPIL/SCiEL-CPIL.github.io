import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2.0])

# Pre-compute triangle indices for efficiency
tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

def point_in_triangle(p, v0, v1, v2):
    """Check if point p is inside or on the boundary of triangle v0-v1-v2."""
    # Barycentric coordinates technique
    # Compute vectors
    v0 = v0
    v1 = v1
    v2 = v2
    p = p
    
    # Area of triangle
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    d1 = cross(p, v0, v1)
    d2 = cross(p, v1, v2)
    d3 = cross(p, v2, v0)
    
    has_neg = (d1 < -1e-10) or (d2 < -1e-10) or (d3 < -1e-10)
    has_pos = (d1 > 1e-10) or (d2 > 1e-10) or (d3 > 1e-10)
    
    return not (has_neg and has_pos)

def is_inside_triangle(points):
    """Vectorized check if all points are inside the triangle."""
    # For efficiency, we can use barycentric coordinates directly
    # But for 11 points, a simple loop or vectorized cross product is fine.
    # Let's use the barycentric coordinate method vectorized.
    
    # v1 - v0, v2 - v0, p - v0
    v1_minus_v0 = V1 - V0
    v2_minus_v0 = V2 - V0
    pts_minus_v0 = points - V0
    
    # Cross products
    # cross(a, b) = a_x*b_y - a_y*b_x
    # We need cross(p-v0, v1-v0), cross(p-v0, v2-v0), etc.
    
    # Actually, simpler:
    # Barycentric coords (u, v, w) for point p:
    # p = u*v0 + v*v1 + w*v2
    # u+v+w = 1
    # We can solve for v and w.
    
    # Let's stick to a robust vectorized check using cross products relative to edges.
    # Edge V0-V1: Normal (0, 1). Point must be above y=0.
    # Edge V1-V2: Normal pointing inwards.
    # Edge V2-V0: Normal pointing inwards.
    
    # Check 1: y >= 0
    if np.any(points[:, 1] < -1e-10):
        return False
    
    # Check 2: Below line V1-V2
    # Line V1(1,0) to V2(0.5, sqrt(3)/2)
    # Slope = (sqrt(3)/2 - 0) / (0.5 - 1) = (sqrt(3)/2) / (-0.5) = -sqrt(3)
    # Equation: y - 0 = -sqrt(3) (x - 1) => y = -sqrt(3)x + sqrt(3) => sqrt(3)x + y - sqrt(3) <= 0
    if np.any(np.sqrt(3)*points[:, 0] + points[:, 1] - np.sqrt(3) > 1e-10):
        return False
        
    # Check 3: Below line V2-V0
    # Line V2(0.5, sqrt(3)/2) to V0(0,0)
    # Slope = (sqrt(3)/2) / 0.5 = sqrt(3)
    # Equation: y = sqrt(3)x => y - sqrt(3)x >= 0
    if np.any(points[:, 1] - np.sqrt(3)*points[:, 0] < -1e-10):
        return False
        
    return True

def calculate_min_area(points):
    """
    points: (N, 2) array of Cartesian coordinates
    Returns: minimum area of any triangle formed by 3 points
    """
    # Get coordinates for all triangles
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
    # Vectorized calculation
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    
    return np.min(area)

def objective(params):
    """
    Objective function for optimization.
    params: (22,) array of flattened point coordinates.
    Returns: -min_area if valid, large penalty if invalid.
    """
    points = params.reshape(N, 2)
    
    # Check if points are inside the triangle
    if not is_inside_triangle(points):
        return 1e10 # Penalty
    
    min_area = calculate_min_area(points)
    return -min_area

def construct():
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Initial guess: Regular grid or known 10-point config + 1
    # Let's use a simple grid as a starting point for DE
    # Or better, use a few random valid configurations to seed DE
    
    # Bounds for DE
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    # DE parameters
    # maxiter and popsize control runtime.
    # We can adjust these based on remaining time.
    
    # Strategy: Run DE for a fixed number of iterations, then polish.
    # If time runs out, return best found so far.
    
    # To speed up, we can use a smaller population and fewer iterations,
    # then refine with local optimization.
    
    # Let's try DE first.
    # workers=-1 uses all cores.
    
    de_result = None
    try:
        de_result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6,
            seed=42,
            workers=-1
        )
    except Exception as e:
        pass
        
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        # If DE took too long, we might not have a result or it's bad.
        # We should have checked time during DE, but DE doesn't easily support callbacks for time.
        # We'll just proceed to polish if we have a result.
        pass

    if de_result is not None and de_result.success:
        best_params = de_result.x
    else:
        # Fallback: Random valid point
        rng = np.random.RandomState(42)
        best_params = None
        for _ in range(1000):
            u, v = rng.random(2)
            if u + v < 1:
                # Map to triangle
                p = u * V1 + v * V2
                # This maps to a triangle with vertices 0, V1, V2? No.
                # Standard mapping: P = (1-u-v)V0 + u V1 + v V2
                # Since V0 is 0, P = u V1 + v V2
                # Condition u+v <= 1 ensures inside.
                # But we need 11 points.
                pass
        
        # Just use the center or a simple valid config if DE failed
        # For now, let's assume DE works or we use a default.
        # A safe default: 11 points on a grid.
        # Let's just pick the centroid repeated? No, area 0.
        # Let's pick a regular 11-point pattern.
        # For robustness, if DE fails, we return a valid random set.
        # But DE usually succeeds.
        
        # Let's generate a valid random set as fallback
        valid_points = []
        while len(valid_points) < N:
            u, v = rng.random(2)
            if u + v < 1:
                p = u * V1 + v * V2
                valid_points.append(p)
        best_params = np.array(valid_points).flatten()
        
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    
    if remaining_time > 5:
        # Local polish with Nelder-Mead
        try:
            # Nelder-Mead doesn't handle bounds well, but we can use it on the interior.
            # We need to ensure the polished points stay inside.
            # We can use a penalty function again.
            
            def objective_polish(params):
                points = params.reshape(N, 2)
                if not is_inside_triangle(points):
                    return 1e10
                return -calculate_min_area(points)
                
            res = minimize(
                objective_polish, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.success:
                best_params = res.x
        except Exception:
            pass

    final_points = best_params.reshape(N, 2)
    
    # Final safety check: ensure points are inside
    if not is_inside_triangle(final_points):
        # If polish pushed them out, revert to DE result or generate new
        # For simplicity, we'll just return the DE result if it was valid
        if de_result is not None and de_result.success:
            final_points = de_result.x.reshape(N, 2)
            if not is_inside_triangle(final_points):
                # Last resort: random valid
                rng = np.random.RandomState(0)
                valid_points = []
                while len(valid_points) < N:
                    u, v = rng.random(2)
                    if u + v < 1:
                        p = u * V1 + v * V2
                        valid_points.append(p)
                final_points = np.array(valid_points)
        else:
            # Generate valid random
            rng = np.random.RandomState(0)
            valid_points = []
            while len(valid_points) < N:
                u, v = rng.random(2)
                if u + v < 1:
                    p = u * V1 + v * V2
                    valid_points.append(p)
            final_points = np.array(valid_points)

    return final_points

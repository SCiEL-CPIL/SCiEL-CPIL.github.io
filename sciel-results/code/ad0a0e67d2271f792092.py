import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3) / 2.0])
    
    # Centroid
    G = (A + B + C) / 3.0

    # Rotation matrix for 120 degrees
    theta = 2 * np.pi / 3.0
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Function to rotate a point around the centroid
    def rotate_point(p):
        p_rel = p - G
        p_rot = R @ p_rel
        return p_rot + G

    def rotate_point2(p):
        p_rel = p - G
        p_rot = R @ (R @ p_rel)
        return p_rot + G

    def is_inside_triangle(p):
        # Check if point p is inside the triangle ABC
        # Using cross product / sign method
        # Or simple linear inequalities
        x, y = p
        if y < 0: return False
        if y > np.sqrt(3) * x: return False
        if y > -np.sqrt(3) * x + np.sqrt(3): return False
        return True

    def get_all_points(params):
        """
        params: array of 6 floats [x1, y1, x2, y2, x3, y3]
        Returns: list of 11 points
        """
        p1 = np.array([params[0], params[1]])
        p2 = np.array([params[2], params[3]])
        p3 = np.array([params[4], params[5]])
        
        points = [G]
        for p in [p1, p2, p3]:
            points.append(p)
            points.append(rotate_point(p))
            points.append(rotate_point2(p))
            
        return np.array(points)

    def min_triangle_area(points):
        """
        Calculate the minimum area of any triangle formed by 3 points.
        """
        n = len(points)
        min_area = float('inf')
        # Iterate over all combinations of 3 points
        # For n=11, this is 165 combinations.
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * | (p2-p1) x (p3-p1) |
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if area < min_area:
                        min_area = area
        return min_area

    def objective(params):
        """
        Objective function for optimization.
        Returns negative of the minimum area (to be minimized).
        """
        # Check time
        if time.time() > deadline:
            return 1e10  # Penalty
        
        p1 = np.array([params[0], params[1]])
        p2 = np.array([params[2], params[3]])
        p3 = np.array([params[4], params[5]])
        
        # Check if representative points are inside the triangle
        # Note: Rotated points will also be inside if the original is, due to symmetry
        if not is_inside_triangle(p1):
            return 1e10
        if not is_inside_triangle(p2):
            return 1e10
        if not is_inside_triangle(p3):
            return 1e10
            
        points = get_all_points(params)
        min_area = min_triangle_area(points)
        
        return -min_area

    # Bounds for the 3 representative points
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [
        (0.0, 1.0), (0.0, np.sqrt(3) / 2.0),
        (0.0, 1.0), (0.0, np.sqrt(3) / 2.0),
        (0.0, 1.0), (0.0, np.sqrt(3) / 2.0)
    ]

    # Run Differential Evolution
    # DE is good for global search
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15,
            tol=1e-6,
            polish=False # We will polish manually
        )
    except Exception:
        # Fallback if DE fails
        result = None

    best_params = result.x if result is not None else None
    best_score = -result.fun if result is not None else 0.0

    # Refine with Nelder-Mead from the best DE solution
    if best_params is not None and time.time() < deadline:
        try:
            res_nm = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_nm.fun < result.fun:
                best_params = res_nm.x
                best_score = -res_nm.fun
        except Exception:
            pass

    # If DE didn't run or failed, try a random restart
    if best_params is None or best_score <= 0:
        # Random restart
        import random
        rng = random.Random(123)
        best_params = None
        best_score = -1e10
        
        for _ in range(20):
            if time.time() > deadline:
                break
            # Random points inside triangle
            # Use barycentric coords for uniform sampling
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            p1 = u * B + v * C
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            p2 = u * B + v * C
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            p3 = u * B + v * C
            
            params = np.array([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]])
            score = -objective(params)
            if score > best_score:
                best_score = score
                best_params = params

    # Construct final points
    if best_params is not None:
        final_points = get_all_points(best_params)
    else:
        # Fallback to seed-like behavior if all else fails
        final_points = np.array([[0.5, 0.5]]) * N # Invalid, just to satisfy shape
        # Better fallback: random points
        import random
        rng = random.Random(7)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                pts.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        final_points = np.array(pts)

    return final_points


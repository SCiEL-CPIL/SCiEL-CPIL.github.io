import numpy as np
import itertools
import time

N = 11

def construct():
    # Vertices of the equilateral triangle
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    # Helper to project a point into the triangle
    # We use barycentric coordinates or simple clipping
    # For an equilateral triangle, we can use the half-plane constraints:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x
    # 3. y <= -sqrt(3) * (x - 1)
    
    def project_to_triangle(pts):
        pts = np.array(pts)
        # Iterate a few times to handle corners
        for _ in range(5):
            # Constraint 1: y >= 0
            mask = pts[:, 1] < 0
            if np.any(mask):
                pts[mask, 1] = 0.0
                # If x was also out of range, it will be fixed by other constraints
                # Actually, if y < 0, we set y=0. x must be in [0,1] roughly.
                
            # Constraint 2: y <= sqrt(3) * x  => x >= y / sqrt(3)
            mask = pts[:, 1] > np.sqrt(3) * pts[:, 0]
            if np.any(mask):
                # Project onto line y = sqrt(3)x
                # Closest point on line y = sqrt(3)x to (x,y)
                # Line: sqrt(3)x - y = 0. Normal: (sqrt(3), -1). Norm^2: 4.
                # d = (sqrt(3)x - y) / 4
                # x_new = x - sqrt(3)*d = x - 3(x-y/sqrt(3))/4 = (x + 3y/sqrt(3))/4 = (x + sqrt(3)y)/4
                # y_new = y - (-1)*d = y + (sqrt(3)x - y)/4 = (3y + sqrt(3)x)/4
                x_old = pts[mask, 0]
                y_old = pts[mask, 1]
                pts[mask, 0] = (x_old + np.sqrt(3)*y_old) / 4.0
                pts[mask, 1] = (3*y_old + np.sqrt(3)*x_old) / 4.0

            # Constraint 3: y <= -sqrt(3) * (x - 1) => y + sqrt(3)x <= sqrt(3) => x <= (sqrt(3)-y)/sqrt(3) = 1 - y/sqrt(3)
            mask = pts[:, 1] > -np.sqrt(3) * (pts[:, 0] - 1)
            if np.any(mask):
                # Project onto line y = -sqrt(3)(x-1) => y = -sqrt(3)x + sqrt(3)
                # Line: sqrt(3)x + y - sqrt(3) = 0. Normal: (sqrt(3), 1). Norm^2: 4.
                # d = (sqrt(3)x + y - sqrt(3)) / 4
                # x_new = x - sqrt(3)*d = x - 3(sqrt(3)x + y - sqrt(3))/4 = (4x - 3sqrt(3)x - 3y + 3sqrt(3))/4
                # This is getting messy. Let's use a simpler geometric projection for the right edge.
                # Line passes through (1,0) and (0.5, sqrt(3)/2).
                # Vector along line: (-0.5, sqrt(3)/2). Unit vector: (-1/2, sqrt(3)/2).
                # Point P. Vector from B(1,0) to P: v = P - B.
                # Projection of v onto unit vector u: proj = (v dot u) u.
                # Closest point = B + proj.
                # But we only want to project if the point is "outside" to the right.
                # The normal pointing inward is (-sqrt(3), -1)? No, inward is towards origin.
                # Let's just clamp x and y based on the lines, it's safer and faster for SA.
                # If y > -sqrt(3)(x-1), then x < 1 - y/sqrt(3) is violated? No.
                # Inequality: y <= -sqrt(3)x + sqrt(3).
                # If violated, y is too high for given x, or x is too high for given y.
                # Since we want to stay in triangle, we can clamp x to max allowed x for that y.
                # x_max = (sqrt(3) - y) / sqrt(3) = 1 - y/sqrt(3).
                # Also x_min = y/sqrt(3).
                # So if x > x_max, set x = x_max.
                # If x < x_min, set x = x_min.
                # Let's re-evaluate the constraints with x-clamping which is often more stable.
                
                # Let's restart the projection logic with clamping x between the two side lines.
                # Left side: x >= y / sqrt(3)
                # Right side: x <= 1 - y / sqrt(3)
                # Bottom: y >= 0
                # Top: y <= sqrt(3)/2 (implied by x constraints if x in [0,1])
                
                # Actually, let's just do a robust iterative clipping.
                pass
        
        # Simpler and more robust projection:
        # 1. Clip y to [0, sqrt(3)/2]
        pts[:, 1] = np.clip(pts[:, 1], 0, np.sqrt(3)/2)
        
        # 2. Clip x to [y/sqrt(3), 1 - y/sqrt(3)]
        x_min = pts[:, 1] / np.sqrt(3)
        x_max = 1.0 - pts[:, 1] / np.sqrt(3)
        pts[:, 0] = np.clip(pts[:, 0], x_min, x_max)
        
        return pts

    def min_area_vectorized(pts):
        # Calculate area for all combinations of 3 points
        # pts shape (N, 2)
        # We can generate all combinations of indices
        # But N=11, C(11,3) = 165. This is small.
        # However, vectorizing the cross product for all 165 triangles at once is fast.
        
        # Get all pairs i, j, k with i < j < k
        # To vectorize, we can create arrays of coordinates.
        # p1 = pts[i], p2 = pts[j], p3 = pts[k]
        # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
        
        # Instead of 165 separate calculations, we can use broadcasting if we structure it right,
        # but with N=11, a simple loop over combinations is likely faster in Python overhead than complex NumPy broadcasting setup for such small numbers,
        # UNLESS we precompute indices.
        
        # Let's try a hybrid: precompute index triples.
        idx = np.array(list(itertools.combinations(range(N), 3)))
        # idx shape (165, 3)
        i_idx = idx[:, 0]
        j_idx = idx[:, 1]
        k_idx = idx[:, 2]
        
        p1 = pts[i_idx] # (165, 2)
        p2 = pts[j_idx] # (165, 2)
        p3 = pts[k_idx] # (165, 2)
        
        # Cross product
        # (p2 - p1) x (p3 - p1)
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        return np.min(areas)

    # Simulated Annealing
    np.random.seed(42)
    
    # Initial points: Random
    best_pts = np.random.rand(N, 2)
    best_pts[:, 1] *= np.sqrt(3)/2
    best_pts = project_to_triangle(best_pts)
    
    current_pts = best_pts.copy()
    current_score = min_area_vectorized(current_pts)
    
    # SA Parameters
    T_start = 0.1
    T_end = 1e-6
    alpha = 0.995 # Cooling rate
    max_iter = 50000
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for safety
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        T = T_start * (alpha ** i)
        if T < T_end:
            break
            
        # Generate new point
        # Move one random point by a small random vector
        idx = np.random.randint(N)
        # Step size decreases with temperature
        step_size = T * 0.5
        move = np.random.normal(0, step_size, 2)
        
        new_pts = current_pts.copy()
        new_pts[idx] += move
        new_pts = project_to_triangle(new_pts)
        
        new_score = min_area_vectorized(new_pts)
        
        # Acceptance criteria
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp((new - current) / T)
            # Since we maximize, delta = new - current is negative
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_pts = new_pts
                    current_score = new_score
            
    return best_pts

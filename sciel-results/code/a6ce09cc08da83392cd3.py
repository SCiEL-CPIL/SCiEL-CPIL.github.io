import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11

# Precompute triangle indices for vectorized area calculation
tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

def _calculate_min_area(points):
    """Compute minimum triangle area for 11 points."""
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5
    return np.min(areas)

def _clip_to_triangle(points):
    """Ensure points are within the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    points = points.copy()
    for i in range(len(points)):
        x, y = points[i]
        # Constraints:
        # 1. y >= 0
        # 2. y <= sqrt(3) * x  => y - sqrt(3)*x <= 0
        # 3. y <= sqrt(3) * (1-x) => y + sqrt(3)*x - sqrt(3) <= 0
        
        # Simple projection: if outside, find closest point on boundary?
        # Or just clamp? Clamping might break other constraints.
        # Better: If outside, project to the triangle.
        
        # Check if inside
        if y < 0 or y > np.sqrt(3) * x or y > np.sqrt(3) * (1 - x):
            # Simple heuristic: scale towards centroid (0.5, sqrt(3)/6)
            # Centroid is (0.5, sqrt(3)/6)
            cx, cy = 0.5, np.sqrt(3)/6.0
            # Find t such that (1-t)*p + t*c is inside.
            # This is a bit expensive.
            # Alternative: Just reject the move in SA if it goes out of bounds.
            # For local refinement, we need valid points.
            # Let's implement a proper projection.
            
            # Distance to edges
            # Edge 1: y=0 -> dist = y
            # Edge 2: y - sqrt(3)x = 0 -> dist = (y - sqrt(3)x)/2
            # Edge 3: y - sqrt(3)(1-x) = 0 -> dist = (y - sqrt(3) + sqrt(3)x)/2
            
            # If y < 0, project to y=0.
            if y < 0:
                points[i, 1] = 0
                x = points[i, 0]
                # Check other constraints
                if y > np.sqrt(3) * x: # 0 > sqrt(3)x is false for x>0
                    pass
                if y > np.sqrt(3) * (1-x): # 0 > sqrt(3)(1-x) is false for x<1
                    pass
            elif y > np.sqrt(3) * x:
                # Project to line y = sqrt(3)x
                # Closest point on line to (x,y)
                # Line: sqrt(3)x - y = 0
                # Normal: (sqrt(3), -1)
                # d = (sqrt(3)x - y) / (3 + 1) = (sqrt(3)x - y)/4
                # p_new = (x,y) - d * (sqrt(3), -1)
                d = (np.sqrt(3)*x - y) / 4.0
                new_x = x - d * np.sqrt(3)
                new_y = y + d
                points[i] = [new_x, new_y]
            elif y > np.sqrt(3) * (1 - x):
                # Project to line y = sqrt(3)(1-x) => sqrt(3)x + y - sqrt(3) = 0
                # Normal: (sqrt(3), 1)
                # d = (sqrt(3)x + y - sqrt(3)) / 4
                d = (np.sqrt(3)*x + y - np.sqrt(3)) / 4.0
                new_x = x - d * np.sqrt(3)
                new_y = y - d
                points[i] = [new_x, new_y]
            
            # Re-check after projection, might still be out due to corner cuts
            # If still out, project to centroid
            x, y = points[i]
            if y < 0 or y > np.sqrt(3) * x or y > np.sqrt(3) * (1 - x):
                points[i] = [0.5, np.sqrt(3)/6.0]
                
    return points

def _sa_objective(points):
    """Objective for SA: minimize -min_area."""
    return -_calculate_min_area(points)

def _local_refine(points):
    """Local refinement using Nelder-Mead."""
    def _obj_flat(flat_points):
        pts = flat_points.reshape(N, 2)
        pts = _clip_to_triangle(pts)
        return -_calculate_min_area(pts)
    
    initial_flat = points.flatten()
    result = minimize(_obj_flat, initial_flat, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    
    best_pts = result.x.reshape(N, 2)
    best_pts = _clip_to_triangle(best_pts)
    return best_pts

def construct():
    """Main function: Simulated Annealing + Local Refinement."""
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Initialize points randomly inside the triangle
    # Using rejection sampling or barycentric
    rng = np.random.RandomState(42)
    points = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2.0) * v
                points[i] = [x, y]
                break
    
    current_score = _calculate_min_area(points)
    best_points = points.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 0.005
    T_end = 1e-6
    alpha = 0.9995  # Cooling rate
    step_size_start = 0.05
    step_size_end = 0.001
    step_alpha = 0.9995
    
    T = T_start
    step_size = step_size_start
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        i = rng.randint(N)
        old_point = points[i].copy()
        
        # Generate new position
        new_point = old_point + rng.normal(0, step_size, size=2)
        
        # Clip to triangle
        temp_pts = points.copy()
        temp_pts[i] = new_point
        temp_pts = _clip_to_triangle(temp_pts)
        
        # Calculate new score
        new_score = _calculate_min_area(temp_pts)
        delta = new_score - current_score
        
        # Metropolis criterion
        if delta > 0:
            # Better
            points = temp_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Worse, accept with probability exp(-delta/T)
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                points = temp_pts
                current_score = new_score
        
        # Update temperature and step size
        T *= alpha
        step_size *= step_alpha
        
    # Final Local Refinement
    refined_points = _local_refine(best_points)
    final_score = _calculate_min_area(refined_points)
    
    # If refinement didn't help (shouldn't happen, but check)
    if final_score > best_score:
        best_points = refined_points
        best_score = final_score
        
    return best_points

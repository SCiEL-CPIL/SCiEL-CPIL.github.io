import numpy as np
import time
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Initialize with regular polygon angles
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_angles = angles.copy()
    
    def compute_score(angles):
        # Convert angles to points
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Calculate Convex Hull Area
        # Points on a circle are in convex position.
        # We need to sort them by angle to apply Shoelace formula correctly for the hull.
        # Since they are on the unit circle, sorting by angle gives the hull vertices in order.
        sorted_angles = np.sort(angles)
        sorted_points = np.column_stack([np.cos(sorted_angles), np.sin(sorted_angles)])
        
        # Shoelace formula for polygon area
        x = sorted_points[:, 0]
        y = sorted_points[:, 1]
        hull_area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        
        if hull_area <= 1e-10:
            return 0.0

        # Calculate all triangle areas
        min_area = float('inf')
        # N=14, C(14,3) = 364 triangles. Loop is fast.
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    p0 = points[i]
                    p1 = points[j]
                    p2 = points[k]
                    # Area = 0.5 * |cross(p1-p0, p2-p0)|
                    v1 = p1 - p0
                    v2 = p2 - p0
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    area = 0.5 * abs(cross)
                    if area < min_area:
                        min_area = area
                        
        return min_area / hull_area

    current_score = compute_score(best_angles)
    best_score = current_score
    
    # Local optimization on angles
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            original_angle = best_angles[i]
            # Try a few random perturbations
            for _ in range(5):
                # Random small change
                delta = np.random.uniform(-step_size, step_size)
                new_angle = original_angle + delta
                # Keep within [0, 2pi) for consistency, though cos/sin handle periodicity
                # Sorting handles the order, so we don't strictly need to wrap, 
                # but it keeps numbers small.
                new_angles = best_angles.copy()
                new_angles[i] = new_angle
                
                new_score = compute_score(new_angles)
                if new_score > best_score:
                    best_score = new_score
                    best_angles = new_angles
                    improved = True
                    # Optional: reduce step size slightly on improvement
                    # step_size *= 0.95
                    break # Move to next point if one improvement found? Or keep trying?
                # If not improved, revert (implicit by using copy)
        
        if not improved:
            no_improve_count += 1
            # Reduce step size if stuck
            step_size *= 0.9
            if step_size < 1e-6:
                # Restart with larger step or break
                step_size = 0.1
                no_improve_count = 0
                if no_improve_count > 20:
                    break
        else:
            no_improve_count = 0

    # Return the best points found
    final_points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    return final_points

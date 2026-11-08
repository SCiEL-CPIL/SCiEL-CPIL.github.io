import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

N = 13

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 distinct points."""
    min_area = np.inf
    # Vectorized calculation for speed
    # points shape (N, 2)
    # We need all combinations of 3 indices.
    # C(13, 3) = 286.
    
    # Precompute indices
    idx = np.arange(N)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i = i[i < j]
    j = j[i < j]
    k = k[i < j]
    j = j[i < k]
    k = k[i < k]
    i = i[i < k]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        return 1e-10

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return -1e10
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _generate_two_shell_config(n_inner, r_inner, r_outer, phase_inner, phase_outer, rotation_inner, rotation_outer):
    """
    Generate points for a two-shell configuration.
    n_inner: number of points on inner circle
    r_inner: radius of inner circle
    r_outer: radius of outer circle
    phase_inner: angular offset for inner points
    phase_outer: angular offset for outer points
    rotation_inner: rotation of the regular polygon
    rotation_outer: rotation of the regular polygon
    """
    points = []
    
    # Inner points
    if n_inner > 0:
        angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rotation_inner
        x_inner = r_inner * np.cos(angles_inner)
        y_inner = r_inner * np.sin(angles_inner)
        points.extend(zip(x_inner, y_inner))
        
    # Outer points
    n_outer = N - n_inner
    if n_outer > 0:
        angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rotation_outer
        x_outer = r_outer * np.cos(angles_outer)
        y_outer = r_outer * np.sin(angles_outer)
        points.extend(zip(x_outer, y_outer))
        
    return np.array(points)

def _refine_coordinate_descent(points, time_limit):
    """
    Refine points using coordinate descent.
    Only moves one point at a time in small steps.
    """
    current_points = points.copy()
    current_score = _score(current_points)
    start_time = time.time()
    
    # Step size decay
    step_size = 0.05
    
    while time.time() < time_limit:
        improved = False
        # Shuffle order to avoid bias
        indices = np.arange(N)
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > time_limit:
                break
            
            best_local_score = current_score
            best_local_point = current_points[idx].copy()
            
            # Try moving in 8 directions
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                direction = np.array([np.cos(angle), np.sin(angle)])
                new_point = current_points[idx] + direction * step_size
                
                new_points = current_points.copy()
                new_points[idx] = new_point
                
                new_score = _score(new_points)
                
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_point = new_point
            
            if best_local_score > current_score:
                current_points[idx] = best_local_point
                current_score = best_local_score
                improved = True
            else:
                # Try shrinking step size if no improvement in this round?
                # Simple approach: keep step size constant but decay globally if no improvement in a full pass
                pass
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-4:
                break

    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for 150s limit
    
    best_score = -np.inf
    best_points = None
    
    # Strategy: Search over two-shell configurations
    # Parameters:
    # n_inner: 0 to 13
    # r_inner: radius (0 to 1)
    # r_outer: radius (r_inner to 1.5)
    # rotation_inner, rotation_outer: angular offsets
    
    # We will do a coarse grid search on (n_inner, r_inner, r_outer) 
    # and a finer search on rotations.
    
    # Initial best: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(base_points)
    best_points = base_points.copy()
    
    # Grid search parameters
    # n_inner from 0 to 13
    # r_inner from 0.2 to 0.8 (step 0.1)
    # r_outer from 1.0 to 1.2 (step 0.05)
    # rotations: 0 to 2*pi (step pi/12)
    
    # To be efficient, we will iterate n_inner, and for each, do a small grid on radii and rotations.
    
    for n_inner in range(N + 1):
        if time.time() > deadline - 30:
            break
            
        # If n_inner is 0 or N, it's just a regular polygon (already covered by base case mostly, but radii scale doesn't matter for score)
        # However, mixing shells is the key.
        
        # Radii grid
        r_inner_vals = np.linspace(0.1, 0.7, 5)
        r_outer_vals = np.linspace(1.0, 1.3, 5)
        
        for r_inner in r_inner_vals:
            for r_outer in r_outer_vals:
                if r_inner >= r_outer:
                    continue
                
                # Rotations grid
                # We can fix rotation_inner to 0 WLOG due to rotational symmetry of the inner set?
                # Not exactly, because the outer set is distinct.
                # But we can fix rotation_inner = 0 and vary rotation_outer.
                # And we can vary the relative phase.
                
                rot_inner_vals = [0.0] # Fix inner rotation to 0
                rot_outer_vals = np.linspace(0, 2 * np.pi, 12, endpoint=False)
                
                for rot_in in rot_inner_vals:
                    for rot_out in rot_outer_vals:
                        if time.time() > deadline - 10:
                            break
                            
                        points = _generate_two_shell_config(n_inner, r_inner, r_outer, 0, 0, rot_in, rot_out)
                        
                        # Quick score check
                        s = _score(points)
                        
                        if s > best_score:
                            best_score = s
                            best_points = points.copy()
                            
                        # If this is a promising candidate, refine it
                        # Heuristic: if it's close to best or better, refine
                        if s > best_score * 0.95:
                            # Refine with coordinate descent
                            # Limit time per refinement
                            refine_deadline = time.time() + 5
                            refined_points, refined_score = _refine_coordinate_descent(points, refine_deadline)
                            
                            if refined_score > best_score:
                                best_score = refined_score
                                best_points = refined_points.copy()
                                
                            # Update global deadline check
                            if time.time() > deadline - 5:
                                break

    # Final refinement on the best found
    if time.time() < deadline - 2 and best_points is not None:
        final_deadline = deadline - 1
        refined_points, refined_score = _refine_coordinate_descent(best_points, final_deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points

    return best_points

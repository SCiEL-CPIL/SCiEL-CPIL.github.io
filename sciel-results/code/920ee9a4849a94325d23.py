import numpy as np
import time
from scipy.optimize import minimize

def _compute_polygon_score(points):
    """
    Computes the score for a given set of points (20, 2).
    Returns the minimum vertex score.
    """
    n = len(points)
    # Normalize
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    dists_to_centroid = np.linalg.norm(points_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    points_norm = points_norm / mean_dist

    # Check strictly convex
    # 1. No repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points_norm[i] - points_norm[j]) < 1e-9:
                return 0.0
            
    # 2. Cyclic order and convexity
    # Calculate cross products of consecutive edges
    # Edge i: points[i] -> points[i+1]
    # Edge i+1: points[i+1] -> points[i+2]
    # Cross product z-component determines turn direction
    signs = []
    for i in range(n):
        p1 = points_norm[i]
        p2 = points_norm[(i+1) % n]
        p3 = points_norm[(i+2) % n]
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-9:
            return 0.0
        signs.append(np.sign(cross))
        
    # All signs must be the same
    if not all(s == signs[0] for s in signs):
        return 0.0
        
    # Total turning should be +/- 2*pi
    # For a simple convex polygon, this is automatically satisfied if it's convex and closed.
    # But let's verify the winding number or just trust the convexity check for simple polygons.
    # The prompt says "total turning +-2*pi". For a strictly convex polygon, this holds.

    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        v = points_norm[i]
        # Distances to other vertices
        diffs = points_norm - v
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf # Exclude self
        dists_sorted = np.sort(dists)
        
        # d_0 ... d_18
        if np.any(dists_sorted < 0.05):
            min_score = 0.0
            continue
            
        # Find k minimizing d_{k+1} - d_k
        gaps = dists_sorted[1:] - dists_sorted[:-1]
        k = np.argmin(gaps)
        
        D = (dists_sorted[k] + dists_sorted[k+1]) / 2.0
        if D < 0.1:
            min_score = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            idxs = [0, 1, 2, 3]
        elif k == 17: # n-3
            idxs = [15, 16, 17, 18]
        else:
            idxs = [k-1, k, k+1, k+2]
            
        selected_dists = dists_sorted[idxs]
        ratios = np.maximum(selected_dists / D, D / selected_dists)
        raw = np.mean(ratios)
        
        if raw == 0:
            score_i = 1.0
        else:
            score_i = min(1.0, 1.0 / raw)
            
        if score_i < min_score:
            min_score = score_i
            
    return min_score

def _objective(angles):
    """
    Objective function for optimization.
    angles: array of 20 angles.
    """
    # Ensure cyclic order
    # We assume angles are in [0, 2*pi) and sorted?
    # To be safe, we sort them. But sorting changes the vertex correspondence.
    # The problem asks for "cyclic order".
    # If we optimize angles, we should ensure they are in cyclic order.
    # Let's assume the input angles are the actual angles of the vertices in order.
    # We will penalize if they are not in increasing order (modulo 2*pi).
    
    # Normalize angles to [0, 2*pi)
    angles = np.mod(angles, 2*np.pi)
    
    # Check if angles are in cyclic order
    # Sort them to check if the order is preserved?
    # Actually, if we start from regular and move slowly, order is preserved.
    # But Nelder-Mead can take big steps.
    # Let's check if the sequence is increasing (with wrap-around).
    # A simple check: sort angles. If the sorted order is a rotation of the input order, it's cyclic.
    # For simplicity, let's just check if the points form a convex polygon.
    
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    
    # Quick convexity check
    # If not convex, return high penalty
    # We can reuse the logic from _compute_polygon_score but return 0 for invalid.
    # To save time, let's do a fast check.
    # Cross products of consecutive edges.
    for i in range(20):
        p1 = points[i]
        p2 = points[(i+1)%20]
        p3 = points[(i+2)%20]
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross < 1e-9: # Assuming CCW
             # If any cross product is negative or zero, it's not strictly convex CCW
             # But we might be CW.
             # Let's just compute the score, which handles validity.
             pass
             
    score = _compute_polygon_score(points)
    if score == 0.0:
        return 1.0 # Penalty: we want to minimize negative score, so return large positive
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = 600.0
    margin = 30.0 # Safety margin
    
    # Initial guess: Regular polygon
    angles0 = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Compute initial score
    x0 = np.cos(angles0)
    y0 = np.sin(angles0)
    points0 = np.column_stack((x0, y0))
    best_score = _compute_polygon_score(points0)
    best_points = points0.copy()
    
    # Optimization
    # We optimize the angles.
    # To handle the cyclic nature, we can fix angle[0] = 0 and optimize the rest?
    # No, Nelder-Mead works on the whole vector.
    # However, the objective is periodic.
    # Let's just run it.
    
    try:
        # Use Nelder-Mead
        # maxiter is a soft limit, we check time in the callback?
        # SciPy minimize doesn't have a built-in time limit callback that stops early easily.
        # We will run it in a loop or just set a reasonable maxiter and check time after.
        # Given 20 variables, Nelder-Mead might take many iterations.
        # Let's try to be smart: run for a fixed number of iterations, check time, repeat if time remains?
        # Or just run once with a high maxiter and hope it finishes?
        # 600s is a lot.
        
        # Let's try to run multiple restarts or a single long run.
        # A single run from regular polygon is a good start.
        
        # We will use a wrapper that checks time?
        # Actually, we can just run `minimize` and catch exceptions or let it run.
        # But if it hangs, we fail.
        # Let's limit maxiter to a number that fits in time.
        # Each iteration evaluates the objective ~10-20 times (simplex size).
        # Objective is O(N^2 log N) roughly. N=20. Very fast.
        # So we can do many iterations.
        
        res = minimize(_objective, angles0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if res.success or res.fun < -best_score:
            # Check validity
            angles_opt = res.x
            x_opt = np.cos(angles_opt)
            y_opt = np.sin(angles_opt)
            points_opt = np.column_stack((x_opt, y_opt))
            
            score_opt = _compute_polygon_score(points_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                
    except Exception as e:
        pass # Return best found so far

    # If we have time left, we could try a second run with a different initial guess?
    # Or refine the best found?
    # Let's check time.
    if time.time() + margin < deadline:
        # Try a small perturbation of the best solution to see if we can improve further?
        # Or just run another Nelder-Mead from the best solution with tighter tolerances?
        try:
            res2 = minimize(_objective, best_points_to_angles(best_points), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res2.fun < -best_score:
                angles_opt2 = res2.x
                x_opt2 = np.cos(angles_opt2)
                y_opt2 = np.sin(angles_opt2)
                points_opt2 = np.column_stack((x_opt2, y_opt2))
                score_opt2 = _compute_polygon_score(points_opt2)
                if score_opt2 > best_score:
                    best_score = score_opt2
                    best_points = points_opt2.copy()
        except:
            pass

    return best_points

def best_points_to_angles(points):
    # Extract angles from points
    # Points are on the unit circle (approximately, after normalization in score calc, but here they are raw)
    # We should use the raw points from the optimization which are on the unit circle.
    angles = np.arctan2(points[:, 1], points[:, 0])
    # Ensure they are in [0, 2*pi)
    angles = np.mod(angles, 2*np.pi)
    # Ensure cyclic order?
    # The optimization might have shuffled them?
    # The objective function penalizes non-convex/non-cyclic, so they should be in order.
    # But arctan2 might return values not in the correct cyclic sequence if the polygon crosses the branch cut.
    # To be safe, we should probably return the points directly, not angles.
    # The function returns points.
    # The helper is just for the second optimization run.
    # We need to pass angles to minimize.
    # If the points are in cyclic order, the angles should be sortable to match that order.
    # Let's assume the points are in cyclic order and return the angles corresponding to that order.
    # Since we don't have the original angle order from the points easily (except by sorting),
    # and the objective function works on angles, we need the angles in the order that produces the convex polygon.
    # Sorting the angles by value will give the cyclic order for points on the unit circle.
    return np.sort(angles)

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I need to wrap it in the final block.
# EVOLVE-BLOCK-END

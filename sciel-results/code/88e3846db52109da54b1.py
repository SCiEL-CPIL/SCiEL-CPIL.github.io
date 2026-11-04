import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

def _compute_hull_area(pts):
    """Compute area of convex hull of points."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        # Fallback if points are collinear or degenerate
        return 1e-10

def _min_triangle_area_raw(pts):
    """Compute the minimum area of all triangles formed by pts."""
    n = len(pts)
    min_area = float('inf')
    # Vectorized computation of areas for all triples is possible but 
    # for n=14, a loop is fine and simpler to debug.
    # Area of triangle (p1, p2, p3) = 0.5 * |cross(p2-p1, p3-p1)|
    for i in range(n):
        p1 = pts[i]
        for j in range(i + 1, n):
            p2 = pts[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = pts[k]
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Initial guess: Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Center it
    init_pts -= init_pts.mean(axis=0)
    
    best_score = -1.0
    best_pts = init_pts.copy()
    
    # Define the objective function for optimization
    # We want to maximize the minimum triangle area relative to hull area.
    # We normalize the points to have hull area 1.
    # Objective to MINIMIZE: - (soft_min_triangle_area)
    
    beta = 50.0  # Inverse temperature for soft-min
    
    def objective(params):
        pts = params.reshape(N, 2)
        
        # Center
        pts = pts - pts.mean(axis=0)
        
        # Check for degeneracy
        if np.allclose(pts, 0):
            return 1e10
            
        # Scale to unit hull area
        area = _compute_hull_area(pts)
        if area < 1e-9:
            return 1e10
        pts = pts / np.sqrt(area)
        
        # Compute all triangle areas
        # Using vectorization for speed
        # i, j, k indices
        # We can compute cross products for all pairs first? 
        # No, just compute the 364 areas.
        
        # For speed, we can precompute differences? 
        # Let's stick to a reasonably fast loop or vectorized triple loop.
        # Vectorized triple loop in numpy is tricky without memory blowup.
        # n=14 is small. A pure python loop might be slow if called 1000s of times.
        # Let's try to vectorize the area calculation.
        
        # Get all unique pairs (i, j) with i < j
        i_idx, j_idx = np.triu_indices(N, 1)
        v1 = pts[j_idx] - pts[i_idx] # shape (364, 2) -- wait, 14*13/2 = 91 pairs
        
        # We need triples (i, j, k).
        # This is harder to vectorize cleanly without generating all 364 combos explicitly.
        # Given N=14, 364 is small. Let's just do the loop but optimize it slightly.
        
        min_area = float('inf')
        for i in range(N):
            p1 = pts[i]
            for j in range(i + 1, N):
                p2 = pts[j]
                v1 = p2 - p1
                for k in range(j + 1, N):
                    p3 = pts[k]
                    v2 = p3 - p1
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    area = 0.5 * abs(cross)
                    if area < min_area:
                        min_area = area
        
        # Soft min
        # If min_area is very small, exp(-beta * min_area) is close to 1.
        # Sum of exps is approx count.
        # -1/beta * log(sum)
        # This approximation is good when one area is much smaller than others.
        # For a more robust soft-min, we should sum exp(-beta * A_i) for all i.
        # But we only have min_area here. 
        # To get the true soft-min, we need all areas.
        # Let's compute all areas into an array.
        
        areas = np.empty(364)
        idx = 0
        for i in range(N):
            p1 = pts[i]
            for j in range(i + 1, N):
                p2 = pts[j]
                v1 = p2 - p1
                for k in range(j + 1, N):
                    p3 = pts[k]
                    v2 = p3 - p1
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    areas[idx] = 0.5 * abs(cross)
                    idx += 1
        
        # Soft min
        # To avoid overflow/underflow, shift by min
        a_min = np.min(areas)
        soft_min_val = -a_min - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (areas - a_min))))
        
        # We want to maximize soft_min_val, so minimize -soft_min_val
        return -soft_min_val

    # Run optimization
    # We can try a few different initial conditions or betas
    # Since N=14 is small, L-BFGS-B should be efficient.
    
    # To improve robustness, we might run Nelder-Mead if L-BFGS-B fails or gets stuck.
    # But L-BFGS-B with soft-min is usually good.
    
    try:
        res = minimize(
            objective, 
            init_pts.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() > deadline:
            # Return best so far (initial)
            pass
        else:
            if res.success:
                candidate_pts = res.x.reshape(N, 2)
                # Evaluate raw score to compare
                # Normalize candidate
                c_pts = candidate_pts - candidate_pts.mean(axis=0)
                c_area = _compute_hull_area(c_pts)
                if c_area > 1e-9:
                    c_pts = c_pts / np.sqrt(c_area)
                    c_min_area = _min_triangle_area_raw(c_pts)
                    c_score = c_min_area # Since hull area is 1, score is min_area
                    if c_score > best_score:
                        best_score = c_score
                        best_pts = c_pts.copy()
                        
    except Exception:
        pass

    # Try a second run with a different beta or method if time permits
    if time.time() < deadline - 20:
        beta2 = 100.0
        def objective2(params):
            pts = params.reshape(N, 2)
            pts = pts - pts.mean(axis=0)
            if np.allclose(pts, 0):
                return 1e10
            area = _compute_hull_area(pts)
            if area < 1e-9:
                return 1e10
            pts = pts / np.sqrt(area)
            
            areas = np.empty(364)
            idx = 0
            for i in range(N):
                p1 = pts[i]
                for j in range(i + 1, N):
                    p2 = pts[j]
                    v1 = p2 - p1
                    for k in range(j + 1, N):
                        p3 = pts[k]
                        v2 = p3 - p1
                        cross = v1[0] * v2[1] - v1[1] * v2[0]
                        areas[idx] = 0.5 * abs(cross)
                        idx += 1
            
            a_min = np.min(areas)
            soft_min_val = -a_min - (1.0 / beta2) * np.log(np.sum(np.exp(-beta2 * (areas - a_min))))
            return -soft_min_val

        try:
            res2 = minimize(
                objective2, 
                best_pts.flatten(), # Start from previous best
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.success:
                candidate_pts = res2.x.reshape(N, 2)
                c_pts = candidate_pts - candidate_pts.mean(axis=0)
                c_area = _compute_hull_area(c_pts)
                if c_area > 1e-9:
                    c_pts = c_pts / np.sqrt(c_area)
                    c_min_area = _min_triangle_area_raw(c_pts)
                    if c_min_area > best_score:
                        best_score = c_min_area
                        best_pts = c_pts.copy()
        except Exception:
            pass

    # Final check
    if time.time() > deadline:
        # Ensure we return something valid
        pass

    # Normalize best_pts to unit hull area for final return
    # The problem says "Place 14 points... normalized to unit area".
    # The score is min_area / hull_area.
    # If we return points with hull area 1, the score is just min_area.
    # If we return points with any scale, the evaluator computes the ratio.
    # It is safer to return points with a reasonable scale, e.g., hull area 1.
    
    final_pts = best_pts - best_pts.mean(axis=0)
    final_area = _compute_hull_area(final_pts)
    if final_area > 1e-9:
        final_pts = final_pts / np.sqrt(final_area)
    
    return final_pts

# EVOLVE-BLOCK-END

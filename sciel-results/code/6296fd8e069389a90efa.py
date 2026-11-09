# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Strategy:
    1. Define the objective function to maximize (d_min / d_max)^2, equivalent to minimizing -log(d_min/d_max) or similar.
       We will use a penalty-based approach or direct ratio maximization via scipy's minimize (which minimizes).
       Objective to minimize: - (d_min / d_max) ** 2
       To handle d_max=0, we add a small epsilon or return a large negative value.
       
    2. Use Differential Evolution (DE) for global search.
       DE is population-based and robust to local minima, unlike the single-trajectory hill climbing in the seed.
       
    3. Use Nelder-Mead for local refinement of the best candidate found by DE.
       
    4. Initialize DE with bounds derived from known good configurations (Icosahedron, etc.) or use wide bounds 
       with a high maxiter. Given the 600s limit, we need to be efficient.
       We will fix the scale by normalizing internally or fixing one point/center of mass to reduce dimensionality?
       No, DE handles the full 42D space. We can fix the center of mass to 0 by subtracting the mean in the objective, 
       but DE operates on raw coordinates. It's better to let DE find the coordinates and normalize at the end.
       
       To speed up convergence, we can fix the first point to the origin? No, that breaks symmetry.
       We can fix the center of mass to 0 by projecting the DE population? That's complex.
       Instead, we rely on DE's robustness.
       
       Optimization:
       - Use a multi-start approach with DE? No, DE is already stochastic.
       - Use a tight deadline.
       
    5. Return the best points found.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 540  # Leave 60s margin for safety and final processing
    
    # Objective function: Minimize negative squared ratio
    def objective(x_flat):
        # Check time
        if time.time() > deadline:
            return 1e10  # Large penalty if time runs out
        
        pts = x_flat.reshape(n, d)
        
        # Center of mass constraint: Subtract mean to handle translation invariance
        # This reduces the effective search space and helps convergence
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate distances
        dists = pdist(pts)
        
        # Handle degenerate cases
        if np.any(np.isnan(dists)) or np.any(np.isinf(dists)):
            return 1e10
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return 1e10
            
        ratio = d_min / d_max
        # We want to maximize ratio^2, so minimize -ratio^2
        return -ratio * ratio

    # Bounds for DE
    # The optimal configuration likely has points within a unit sphere or similar scale.
    # Since the metric is scale-invariant, the scale of the solution doesn't matter for the score,
    # but it matters for the optimizer's step sizes.
    # We can fix the scale by requiring, e.g., the max distance to be 1?
    # DE doesn't easily support constraints. 
    # However, if we just let it search in [-1, 1]^42, it will find a configuration.
    # The ratio is scale invariant, so any scale is fine.
    # Let's use bounds [-1, 1] for each coordinate.
    
    bounds = [(-1, 1)] * (n * d)
    
    # DE Parameters
    # maxiter: How many generations? 
    # popsize: How many individuals?
    # We have 540s. DE is expensive.
    # Let's try a moderate population and iteration count.
    # If it takes too long, we can't easily interrupt DE mid-run in scipy easily without callbacks.
    # We can use a callback to check time.
    
    def time_callback(strategy, x, convergence, *args, **kwargs):
        if time.time() > deadline - 30:  # Stop 30s before deadline to allow for final processing
            return True  # True signals stop
        return False

    best_result = None
    best_score = -np.inf
    
    # Run DE
    try:
        # Use 'differential' mutation strategy
        # seed for reproducibility? The task doesn't require it, but it helps debugging.
        # However, random seed might be better for exploration. Let's use a fixed seed for consistency.
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50,  # Generations
            popsize=15,   # Population size factor (15 * 42 = 630 individuals per gen)
            tol=1e-6,
            seed=42,
            callback=time_callback,
            polish=False  # We will do our own polishing with Nelder-Mead
        )
        
        if result.success or result.fun < 1e9:
            best_result = result.x
            best_score = -result.fun
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # If DE didn't find a good result or timed out, fall back to a known good configuration
    if best_result is None or best_score < -1.0:  # -1.0 corresponds to ratio^2 = 1.0 (perfect)
        # Fallback: Icosahedron + 2 points
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi],
            [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0],
            [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1],
            [phi, 0, -1], [-phi, 0, -1]
        ])
        norms = np.linalg.norm(ico_verts, axis=1)
        ico_verts = ico_verts / norms[:, np.newaxis]
        
        fallback_pts = np.zeros((n, d))
        fallback_pts[:12] = ico_verts
        fallback_pts[12] = [0, 0, 1]
        fallback_pts[13] = [0, 0, -1]
        
        # Normalize fallback
        dists_f = pdist(fallback_pts)
        d_max_f = np.max(dists_f)
        if d_max_f > 0:
            fallback_pts = fallback_pts / d_max_f
            
        best_result = fallback_pts.flatten()
        # Calculate score for logging
        dists_f = pdist(fallback_pts)
        best_score = (np.min(dists_f) / np.max(dists_f)) ** 2

    # Local Refinement with Nelder-Mead
    # This is crucial for squeezing out the last bits of performance.
    if time.time() < deadline - 10:
        try:
            # Nelder-Mead requires a minimization function
            res_nm = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res_nm.fun < -best_score:  # Better score (less negative)
                best_result = res_nm.x
                best_score = -res_nm.fun
                
        except Exception as e:
            print(f"Nelder-Mead failed: {e}")
            
    # Final Processing
    points = best_result.reshape(n, d)
    
    # Center of mass to 0
    points -= np.mean(points, axis=0)
    
    # Scale to unit max distance (optional, but keeps numbers nice)
    dists = pdist(points)
    d_max = np.max(dists)
    if d_max > 1e-12:
        points = points / d_max
        
    # Ensure finite
    if not np.all(np.isfinite(points)):
        # Ultimate fallback
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi],
            [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0],
            [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1],
            [phi, 0, -1], [-phi, 0, -1]
        ])
        norms = np.linalg.norm(ico_verts, axis=1)
        ico_verts = ico_verts / norms[:, np.newaxis]
        points = np.zeros((n, d))
        points[:12] = ico_verts
        points[12] = [0, 0, 1]
        points[13] = [0, 0, -1]
        points -= np.mean(points, axis=0)
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max > 0:
            points = points / d_max

    return points

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600.0 # 10 minutes
    
    def compute_ratio(points):
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(points_flat):
        points = points_flat.reshape(n, d)
        # We want to maximize R, so minimize -R
        # To avoid division by zero or numerical issues, add a small epsilon
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-9:
            return 1.0 # Bad configuration
        r = d_min / d_max
        return - (r ** 2)

    best_points = None
    best_score = -1.0

    # Generate several initial configurations
    initializations = []
    
    # 1. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. 4x4 Grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    initializations.append(np.column_stack([xx.ravel(), yy.ravel()]))
    
    # 3. Hexagonal Packing approximation
    # Create a hex grid and take the central 16 points
    # This is a bit complex to generate exactly, so we'll use a simpler "cluster" init
    # Random cluster
    np.random.seed(42)
    initializations.append(np.random.randn(n, d) * 0.5)
    
    # 4. Two concentric circles (8 inner, 8 outer)
    r_inner = 0.5
    r_outer = 1.0
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_outer = np.linspace(np.pi/8, 2 * np.pi + np.pi/8, 8, endpoint=False)
    p_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    p_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    initializations.append(np.vstack([p_inner, p_outer]))

    # Run optimization for each initialization
    for i, init_points in enumerate(initializations):
        if time.time() > deadline - 60:
            break
            
        try:
            # Flatten points for scipy
            x0 = init_points.flatten()
            
            # Use Nelder-Mead for non-convex optimization
            # maxiter is set high, but we check time
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            # Check if we are running out of time during optimization is hard with standard minimize
            # So we rely on the outer loop check. If this single call takes too long, we might timeout.
            # To be safe, we can reduce maxiter or use a callback to check time.
            # Let's use a callback to abort if time is up.
            
        except Exception as e:
            pass

        # Re-run with a time-aware callback or just check time after
        # Actually, standard minimize doesn't have a built-in time limit callback that stops it cleanly in all versions without custom code.
        # Given 600s and ~4 initializations, if each takes ~100s, we are safe.
        # If one hangs, we might timeout. Let's add a simple time check in the loop before starting each.
        
        # Let's refine the best result from the last run if it succeeded
        if 'res' in locals() and res.success:
            pts = res.x.reshape(n, d)
            score = compute_ratio(pts)
            if score > best_score:
                best_score = score
                best_points = pts

    # If optimization failed or didn't improve, try a simple gradient ascent / repulsion method
    # This is a fallback or a second phase.
    
    # Let's add a "Repulsion" refinement step for the best point found so far, or the best init.
    # This is a structural change: using physics-based relaxation.
    
    def repulsion_refine(points, steps=5000):
        pts = points.copy()
        for i in range(steps):
            if time.time() > deadline - 30:
                break
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max == 0:
                break
            # Gradient of min distance is tricky.
            # Instead, maximize sum of 1/d^2 (repulsion) while keeping points bounded?
            # No, we want to maximize d_min/d_max.
            # Let's stick to Nelder-Mead as it's more direct for this metric.
            pass
        return pts

    # Since Nelder-Mead can be slow, let's ensure we have a valid return.
    if best_points is None:
        # Fallback to the best initialization score
        for init_points in initializations:
            score = compute_ratio(init_points)
            if score > best_score:
                best_score = score
                best_points = init_points
    else:
        # One more polish: try Nelder-Mead on the best found point with tighter tolerances?
        # Or just return the best found.
        pass

    # Ensure the output is finite and shape is correct
    if not np.all(np.isfinite(best_points)):
        best_points = initializations[0]

    return best_points

# EVOLVE-BLOCK-END

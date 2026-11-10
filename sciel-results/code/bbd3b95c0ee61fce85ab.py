import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _log_score(pts):
    """
    Calculate the log of the score component: sum(log(d_ij)) - 28*log(D).
    pts: 8x2 array of coordinates.
    """
    # Calculate pairwise distances
    # pts shape (8, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(8, dtype=bool)
    d_vals = dists[mask]
    
    # Check for distinct points
    if np.any(d_vals < 1e-9):
        return -np.inf
        
    D = np.max(d_vals)
    if D < 1e-9:
        return -np.inf
        
    # log_score = sum(log(d_vals)) - 28 * log(D)
    # Note: d_vals contains each unordered pair once.
    # The score definition uses ordered pairs, which squares the product of unordered pairs.
    # log(Score) = const + 2 * sum(log(d_unordered)) - 56 * log(D)
    # Maximizing sum(log(d_unordered)) - 28 * log(D) is equivalent.
    return np.sum(np.log(d_vals)) - 28.0 * np.log(D)

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # Regular octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    reg_octagon = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_score = _log_score(reg_octagon)
    best_pts = reg_octagon.copy()
    
    # Objective function for minimization
    def objective(x):
        pts = x.reshape(8, 2)
        s = _log_score(pts)
        if np.isinf(s):
            return 1e10
        return -s
        
    # Bounds for coordinates
    # Since scale doesn't matter, we can bound them to a reasonable range
    # e.g., [-1, 1]
    bounds = [(-1.0, 1.0)] * 16
    
    # Differential Evolution
    try:
        if time.time() < deadline - 30:
            res_de = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=200, 
                popsize=15,
                tol=1e-6
            )
            if res_de.success or res_de.fun < -best_score:
                de_pts = res_de.x.reshape(8, 2)
                de_score = _log_score(de_pts)
                if de_score > best_score:
                    best_score = de_score
                    best_pts = de_pts
    except Exception:
        pass
        
    # Local refinement using Powell's method
    def refine(pts0):
        nonlocal best_score, best_pts
        try:
            res_min = minimize(
                objective, 
                pts0.flatten(), 
                method='Powell', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            ref_pts = res_min.x.reshape(8, 2)
            ref_score = _log_score(ref_pts)
            if ref_score > best_score:
                best_score = ref_score
                best_pts = ref_pts
        except Exception:
            pass

    # Refine from regular octagon
    if time.time() < deadline - 10:
        refine(reg_octagon)
        
    # Refine from DE result
    if time.time() < deadline - 10 and 'res_de' in locals():
        refine(res_de.x.reshape(8, 2))
        
    # Convert to list of lists of floats
    result = best_pts.tolist()
    
    # Ensure distinct points and finite values
    # If something went wrong, fall back to regular octagon
    try:
        if not all(np.isfinite(p) for p in result):
            return reg_octagon.tolist()
        # Check distinctness
        pts_arr = np.array(result)
        diff = pts_arr[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(8, dtype=bool)
        if np.min(dists[mask]) < 1e-6:
            return reg_octagon.tolist()
    except Exception:
        return reg_octagon.tolist()

    return result

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Solve the Heilbronn problem for N=13 using global optimization (Differential Evolution)
    followed by local refinement (L-BFGS-B).
    """
    N = 13

    def objective(flat_pts):
        """
        Objective function for scipy.optimize.
        Minimizes the negative ratio A_min / A_hull.
        """
        pts = flat_pts.reshape((N, 2))
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(pts)):
            return 1e9
            
        # Calculate min triangle area
        min_area = float('inf')
        for i in range(N):
            x1, y1 = pts[i]
            for j in range(i + 1, N):
                x2, y2 = pts[j]
                for k in range(j + 1, N):
                    x3, y3 = pts[k]
                    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                    if area < min_area:
                        min_area = area
                        if min_area < 1e-10: # Early exit if degenerate
                            return 1e9
        
        # Calculate hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e9
        except Exception:
            return 1e9
            
        ratio = min_area / hull_area
        return -ratio

    # Define bounds for each coordinate
    # The problem is scale-invariant, so we can bound to [-1, 1]
    bounds = [(-1, 1)] * (2 * N)
    
    # Run Differential Evolution
    # seed=42 for reproducibility
    # maxiter and popsize control the thoroughness
    # tol controls the stopping criteria
    result_de = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=1000, 
        popsize=15, 
        tol=1e-6,
        polish=False # We will polish separately
    )
    
    best_flat = result_de.x
    best_score = -result_de.fun
    
    # Local Refinement with L-BFGS-B
    # This helps to find the exact local optimum near the DE result
    result_local = minimize(
        objective, 
        best_flat, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if result_local.fun < result_de.fun:
        best_flat = result_local.x
        best_score = -result_local.fun

    # Multi-start strategy: Run DE with different seeds to ensure we don't miss the global optimum
    # Since DE is stochastic, running it a few times with different seeds is safer.
    # However, 600s limit. DE with 1000 iters might take time.
    # Let's estimate: 15 popsize * 1000 iters = 15000 evals.
    # Each eval is ~286 triangle calcs + 1 hull calc.
    # 286 ops is ~0.01ms. Hull is ~0.1ms. Total ~0.2ms per eval.
    # 15000 * 0.2ms = 3 seconds. This is very fast.
    # We can afford multiple runs.
    
    best_flat_global = best_flat
    best_score_global = best_score
    
    for seed in [1, 2, 3]:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=seed, 
            maxiter=500, 
            popsize=10, 
            tol=1e-5,
            polish=False
        )
        if -res.fun > best_score_global:
            best_score_global = -res.fun
            best_flat_global = res.x
            
            # Refine the new best
            res_local = minimize(
                objective, 
                best_flat_global, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if -res_local.fun > best_score_global:
                best_score_global = -res_local.fun
                best_flat_global = res_local.x

    final_points = best_flat_global.reshape((N, 2))
    
    # Final sanity check
    if not np.all(np.isfinite(final_points)):
        # Fallback to a known good configuration if something went wrong
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        final_points = np.array([[0,0]] + [[np.cos(a), np.sin(a)] for a in angles])
        
    return final_points

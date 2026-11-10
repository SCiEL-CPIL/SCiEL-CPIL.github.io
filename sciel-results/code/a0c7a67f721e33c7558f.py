import numpy as np
from scipy.optimize import differential_evolution
import time

def calculate_score(pts):
    """
    Calculate the minimum vertex score for a polygon given as (n, 2) array of points.
    Assumes points are in cyclic order.
    """
    n = len(pts)
    if n < 3:
        return 0.0

    # Centroid and normalization
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    
    # Mean distance to centroid
    dists_to_centroid = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
        
    pts_norm = pts_norm / mean_dist

    # Check for strict convexity
    # 1. No repeated points
    # 2. Every turn between consecutive edges is nonzero and of one sign
    # 3. Total turning +-2*pi
    
    # Check repeated points
    diff = pts_norm[:, np.newaxis, :] - pts_norm[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    
    if np.min(dist_matrix) < 1e-6:
        return 0.0

    # Calculate cross products to check turns
    # Vector v_i = p_{i+1} - p_i
    # Cross product z-component of v_i x v_{i+1}
    v = np.roll(pts_norm, -1, axis=0) - pts_norm
    cross_z = v[:, 0] * np.roll(v, -1, axis=0)[:, 1] - v[:, 1] * np.roll(v, -1, axis=0)[:, 0]
    
    # All cross products must be non-zero and same sign
    if np.any(np.abs(cross_z) < 1e-9):
        return 0.0
        
    if not (np.all(cross_z > 0) or np.all(cross_z < 0)):
        return 0.0

    # Calculate scores for each vertex
    min_score = np.inf
    
    for i in range(n):
        # Get distances from i to all other vertices
        d = dist_matrix[i, :].copy()
        
        # Sort distances
        d_sorted = np.sort(d)
        
        # Check min distance > 0.05
        if d_sorted[0] < 0.05:
            return 0.0
        
        # Find k: first index minimizing d[k+1] - d[k]
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        if D < 0.1:
            return 0.0
        
        # Determine the 4 distances
        if k == 0:
            d_four = d_sorted[0:4]
        elif k == n - 2: # k=17 for n=20
            d_four = d_sorted[15:19]
        else:
            # d_{k-1} .. d_{k+2}
            d_four = d_sorted[k-1:k+3]
            
        # Calculate raw score
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_four / D, D / d_four)
        raw = np.mean(ratios)
        
        score_i = min(1.0, 1.0 / raw)
        
        if score_i < min_score:
            min_score = score_i
            
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # Start time
    start_time = time.time()
    deadline = start_time + 600.0 # 600s limit, keep safety margin
    
    # Fourier series representation for radial perturbation
    # r(theta) = 1 + sum_{k=1}^{K} (a_k cos(k theta) + b_k sin(k theta))
    # We optimize coefficients a_k, b_k for k=1..K
    # To ensure convexity, we need to be careful.
    # However, the directive suggests using a derivative-free evolutionary algorithm
    # with a hard convexity check.
    
    # Let's use Differential Evolution
    # Parameters: a_1, b_1, ..., a_K, b_K
    # Let's try K=3 (6 parameters) or K=5 (10 parameters)
    K = 3
    num_params = 2 * K
    
    # Bounds for coefficients
    # Small perturbations to maintain convexity
    # If r is too small or varies too much, convexity might fail
    # Let's bound them to [-0.2, 0.2] initially
    bounds = [(-0.2, 0.2)] * num_params
    
    def params_to_polygon(params):
        # Unpack params
        coeffs = params
        a = coeffs[0::2]
        b = coeffs[1::2]
        
        # Angles for vertices
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # Calculate radii
        radii = np.ones(n)
        for k in range(1, K+1):
            radii += a[k-1] * np.cos(k * angles)
            radii += b[k-1] * np.sin(k * angles)
        
        # Ensure radii are positive
        if np.any(radii <= 0.1):
            return None
            
        # Convert to cartesian
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        pts = np.column_stack((x, y))
        
        return pts

    def objective(params):
        # Check time
        if time.time() > deadline:
            return 1.0 # Large penalty to stop
            
        pts = params_to_polygon(params)
        if pts is None:
            return 1.0 # Penalty for invalid radii
            
        score = calculate_score(pts)
        if score == 0.0:
            return 1.0 # Penalty for invalid polygon or bad score
        return -score

    # Initial best is regular polygon
    best_score = calculate_score(np.column_stack((np.cos(np.linspace(0, 2*np.pi, n, endpoint=False)), 
                                                  np.sin(np.linspace(0, 2*np.pi, n, endpoint=False)))))
    best_params = np.zeros(num_params)
    
    # Run Differential Evolution
    # maxiter and popsize need to be tuned
    # We have ~600s. DE can be slow.
    # Let's try to run it with a time limit inside the objective if possible,
    # or just rely on scipy's internal limits if we set them appropriately.
    # But scipy's DE doesn't have a time limit option directly.
    # We can use a wrapper that checks time.
    
    # Since we can't easily interrupt DE mid-run without a callback or global flag,
    # and the objective checks time, it will return 1.0 which is bad,
    # but the algorithm might continue.
    # Better to run multiple shorter optimizations or just one long one and hope it finishes.
    # Or use a smaller population and maxiter.
    
    # Let's estimate: 20 vertices, distance matrix 20x20, sorting 19 elements.
    # This is fast.
    # DE with popsize=20, maxiter=1000 might take a while.
    
    # Let's try a conservative setup
    try:
        res = differential_evolution(objective, bounds, maxiter=500, popsize=20, tol=1e-6, polish=True)
        
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x
            
    except Exception as e:
        pass

    # If we still have time, try a local refinement with Nelder-Mead on the best found
    if time.time() < deadline - 30.0:
        from scipy.optimize import minimize
        
        def objective_nm(params):
            if time.time() > deadline:
                return 1.0
            pts = params_to_polygon(params)
            if pts is None:
                return 1.0
            score = calculate_score(pts)
            if score == 0.0:
                return 1.0
            return -score

        try:
            res2 = minimize(objective_nm, best_params, method='Nelder-Mead', options={'maxiter': 1000})
            if res2.fun < -best_score:
                best_score = -res2.fun
                best_params = res2.x
        except Exception as e:
            pass

    # Construct final polygon
    pts = params_to_polygon(best_params)
    if pts is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return pts

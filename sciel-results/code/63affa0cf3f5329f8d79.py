import numpy as np
from scipy.optimize import minimize, differential_evolution

def _score_polygon(points):
    n = len(points)
    # Normalize
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    dists_to_centroid = np.linalg.norm(points_centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
        
    points_norm = points_centered / mean_dist
    
    # Pairwise distances
    # (N, N)
    diffs = points_norm[:, np.newaxis, :] - points_norm[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    
    # Set diagonal to inf to exclude self
    np.fill_diagonal(dists, np.inf)
    
    # Sort distances for each vertex
    # (N, N-1)
    d_sorted = np.sort(dists, axis=1)
    
    # Check for repeated points (distance < 0.05)
    if np.any(d_sorted[:, 0] < 0.05):
        return 0.0
        
    # Gaps
    gaps = np.diff(d_sorted, axis=1)
    
    # k: index of min gap for each vertex
    k = np.argmin(gaps, axis=1)
    
    # D
    # d_sorted[:, k] and d_sorted[:, k+1]
    # Need to handle k+1 index carefully if k is last?
    # d_sorted has N-1 columns. k is in 0..N-2.
    # k+1 is in 1..N-1.
    # So d_sorted[:, k+1] is valid.
    d_k = d_sorted[np.arange(n), k]
    d_k1 = d_sorted[np.arange(n), k + 1]
    D = (d_k + d_k1) / 2.0
    
    # Check D < 0.1
    if np.any(D < 0.1):
        return 0.0
        
    # Select 4 distances
    # Indices:
    # if k == 0: 0, 1, 2, 3
    # if k == n-2: n-4, n-3, n-2, n-1
    # else: k-1, k, k+1, k+2
    
    d_sel = np.zeros((n, 4))
    
    # Mask for k == 0
    mask0 = (k == 0)
    if np.any(mask0):
        d_sel[mask0] = d_sorted[mask0][:, 0:4]
        
    # Mask for k == n-2
    mask_last = (k == n - 2)
    if np.any(mask_last):
        d_sel[mask_last] = d_sorted[mask_last][:, n-4:n]
        
    # Mask for else
    mask_mid = ~(mask0 | mask_last)
    if np.any(mask_mid):
        k_mid = k[mask_mid]
        # d_sorted[:, k-1], d_sorted[:, k], d_sorted[:, k+1], d_sorted[:, k+2]
        # We need to gather these.
        # d_sorted is (N, N-1).
        # k_mid is array of indices.
        # d_k-1 = d_sorted[np.arange(n)[mask_mid], k_mid - 1]
        # d_k = d_sorted[np.arange(n)[mask_mid], k_mid]
        # d_k1 = d_sorted[np.arange(n)[mask_mid], k_mid + 1]
        # d_k2 = d_sorted[np.arange(n)[mask_mid], k_mid + 2]
        
        idx = np.arange(n)[mask_mid]
        d_sel[mask_mid, 0] = d_sorted[idx, k_mid - 1]
        d_sel[mask_mid, 1] = d_sorted[idx, k_mid]
        d_sel[mask_mid, 2] = d_sorted[idx, k_mid + 1]
        d_sel[mask_mid, 3] = d_sorted[idx, k_mid + 2]
        
    # Ratios
    ratios = np.maximum(d_sel / D[:, np.newaxis], D[:, np.newaxis] / d_sel)
    raw = np.mean(ratios, axis=1)
    
    scores = np.minimum(1.0, 1.0 / raw)
    
    return np.min(scores)

def _generate_polygon(r, theta):
    n = 20
    k = np.arange(1, n + 1)
    z = r * np.exp(1j * theta)
    points_complex = z ** k
    points = np.column_stack((points_complex.real, points_complex.imag))
    return points

def _objective(params):
    r, theta = params
    # Constraints: r > 0, theta > 0
    if r <= 0 or theta <= 0:
        return 1000.0
        
    points = _generate_polygon(r, theta)
    
    # Check convexity
    n = len(points)
    signs = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return 1000.0 # Collinear
        signs.append(np.sign(cross))
        
    if len(set(signs)) > 1:
        return 1000.0 # Not strictly convex
        
    score = _score_polygon(points)
    return -score # Minimize negative score

def equidistant_points_polygon20() -> np.ndarray:
    # Try multiple starting points for Nelder-Mead
    best_score = -1000.0
    best_params = None
    
    # Starting points around the regular polygon
    theta0 = 2.0 * np.pi / 20.0
    r0 = 1.0
    
    starts = [
        [1.0, theta0],
        [0.95, theta0],
        [1.05, theta0],
        [1.0, theta0 * 0.95],
        [1.0, theta0 * 1.05],
        [0.9, theta0 * 0.9],
        [1.1, theta0 * 1.1]
    ]
    
    bounds = [(0.5, 2.0), (0.05 * np.pi, 0.15 * np.pi)]
    
    for start in starts:
        # Check if start is within bounds
        if not (bounds[0][0] <= start[0] <= bounds[0][1] and bounds[1][0] <= start[1] <= bounds[1][1]):
            continue
            
        res = minimize(_objective, start, method='Nelder-Mead', bounds=bounds, options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000})
        
        if res.success or res.fun < -best_score:
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
                
    # If no success, try differential evolution
    if best_score < 0.9:
        res_de = differential_evolution(_objective, bounds, seed=42, maxiter=100, tol=1e-6)
        if res_de.fun < -best_score:
            best_score = -res_de.fun
            best_params = res_de.x
            
    r_opt, theta_opt = best_params
    points = _generate_polygon(r_opt, theta_opt)
    
    return points

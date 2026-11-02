import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def construct():
    # Precompute indices for all triples
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    def score(coords):
        pts = coords.reshape(N, 2)
        
        # Check for degeneracy
        if np.any(np.isinf(pts)) or np.any(np.isnan(pts)):
            return 1e10
            
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 1e10
            
        if hull_area < 1e-10:
            return 1e10

        # Normalize points to unit hull area
        # Scale factor s such that s^2 * hull_area = 1 => s = 1/sqrt(hull_area)
        scale = 1.0 / np.sqrt(hull_area)
        pts_norm = pts * scale
        
        # Vectorized triangle area calculation
        pi = pts_norm[i_flat]
        pj = pts_norm[j_flat]
        pk = pts_norm[k_flat]
        
        v1x = pj[:, 0] - pi[:, 0]
        v1y = pj[:, 1] - pi[:, 1]
        v2x = pk[:, 0] - pi[:, 0]
        v2y = pk[:, 1] - pi[:, 1]
        
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        
        min_area = np.min(areas)
        
        # Maximize min_area, so minimize -min_area
        return -min_area

    # Initial guess: Regular 13-gon
    theta0 = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.cos(theta0)
    y0 = np.sin(theta0)
    coords0 = np.column_stack((x0, y0)).flatten()
    
    best_score = -np.inf
    best_coords = coords0
    
    # Multi-start optimization
    np.random.seed(42) # For reproducibility in search, though construct should be deterministic? 
    # The task says "Each evaluation must finish within 120s". It doesn't forbid randomness, 
    # but deterministic is safer. I will use fixed seeds or no randomness if possible.
    # Let's use a few fixed perturbations.
    
    perturbations = [0.0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05]
    
    for p in perturbations:
        if p == 0.0:
            start = coords0
        else:
            # Deterministic perturbation
            noise = np.sin(np.arange(len(coords0)) * 123.45) * p
            start = coords0 + noise
            
        # Use Powell method, which is derivative-free and handles non-smooth functions well
        res = minimize(score, start, method='Powell', options={'maxiter': 500, 'ftol': 1e-10, 'xtol': 1e-10})
        
        if res.fun < -best_score:
            best_score = -res.fun
            best_coords = res.x
            
        # Refine with Nelder-Mead from Powell's result
        res2 = minimize(score, res.x, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res2.fun < -best_score:
            best_score = -res2.fun
            best_coords = res2.x

    pts = best_coords.reshape(N, 2)
    return pts

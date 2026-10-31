"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

START_TIME = time.time()
TIMEOUT = 110.0 # Safety margin from 120s

def compute_min_area_fast(pts):
    """
    Vectorized calculation of the minimum area of all triangles formed by N points.
    pts: numpy array of shape (N, 2)
    """
    n = len(pts)
    if n < 3:
        return 0.0
        
    # Generate indices for all unique triples i < j < k
    # Using np.triu_indices or similar logic
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = pts[i_flat]
    p2 = pts[j_flat]
    p3 = pts[k_flat]
    
    # Vectorized area calculation
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return np.min(areas)

def objective(params):
    """
    Objective function for differential evolution.
    Maximizes the minimum triangle area.
    Returns the area (DE minimizes, so we could return -area, but let's stick to standard convention 
    and just return the area, and tell DE to maximize? No, DE minimizes. 
    So we return -area or we just return area and use a wrapper. 
    Actually, scipy's DE minimizes. So we should return -area.
    """
    if time.time() - START_TIME > TIMEOUT:
        return 1e9 # Large value to discourage
    
    u = params[0::2]
    v = params[1::2]
    
    # Check validity: u, v >= 0 and u + v <= 1
    if np.any(u < 0) or np.any(v < 0) or np.any(u + v > 1):
        return 1e9
        
    x = u + 0.5 * v
    y = (np.sqrt(3) / 2) * v
    pts = np.column_stack((x, y))
    
    area = compute_min_area_fast(pts)
    return -area # Negate because DE minimizes

def construct():
    global START_TIME
    START_TIME = time.time()
    
    bounds = [(0, 1)] * (2 * N)
    
    # Differential Evolution
    # Use a good seed and reasonable parameters
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=100, 
        popsize=25, 
        tol=1e-8, 
        seed=42,
        polish=False
    )
    
    best_params = result.x
    best_u = best_params[0::2]
    best_v = best_params[1::2]
    
    # Local Refinement with Powell
    # Powell minimizes, so we minimize -area
    def neg_area(params):
        if time.time() - START_TIME > TIMEOUT:
            return 1e9
        u = params[0::2]
        v = params[1::2]
        if np.any(u < 0) or np.any(v < 0) or np.any(u + v > 1):
            return 1e9
        x = u + 0.5 * v
        y = (np.sqrt(3) / 2) * v
        pts = np.column_stack((x, y))
        return -compute_min_area_fast(pts)

    # Run Powell on the best DE result
    try:
        local_result = minimize(
            neg_area, 
            best_params, 
            method='Powell', 
            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if local_result.fun < 1e8: # Valid result
            best_params = local_result.x
    except Exception:
        pass

    # Final check and conversion
    u = best_params[0::2]
    v = best_params[1::2]
    # Clamp to valid range just in case
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    # Fix any u+v > 1 by scaling
    sums = u + v
    mask = sums > 1
    if np.any(mask):
        u[mask] /= sums[mask]
        v[mask] /= sums[mask]
        
    x = u + 0.5 * v
    y = (np.sqrt(3) / 2) * v
    pts = np.column_stack((x, y))
    
    return pts
# EVOLVE-BLOCK-END

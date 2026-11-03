"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        # hull.volume is the area in 2D
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_ratios(pts):
    """
    Compute the ratio of each triangle area to the hull area.
    Returns an array of ratios or None if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return None
    
    # Indices for all triples i < j < k
    # Using np.indices is slow for large N, but N=13 is small.
    # 13 choose 3 = 286 triangles.
    idx_i, idx_j, idx_k = np.indices((N, N, N)).reshape(3, -1)
    mask = (idx_i < idx_j) & (idx_j < idx_k)
    i = idx_i[mask]
    j = idx_j[mask]
    k = idx_k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    return ratios

def _logsumexp_min(ratios, tau):
    """
    Smooth approximation of the minimum of ratios using LogSumExp.
    min(x) = -lim_{tau->inf} (1/tau) log(sum(exp(-tau * x)))
    """
    # For numerical stability, subtract max exponent (which is -min(ratios)*tau)
    # exponents = -tau * ratios
    # max_exp = -tau * min(ratios)
    # sum(exp(exponents - max_exp)) * exp(max_exp)
    # log_sum = max_exp + log(sum(exp(exponents - max_exp)))
    # result = -1/tau * log_sum = min(ratios) - 1/tau * log(sum(exp(-tau*(ratios - min(ratios)))))
    
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    # shifted <= 0, max is 0
    log_sum = np.log(np.sum(np.exp(shifted)))
    # Approximation of min:
    # min_r - (1/tau) * log_sum
    return min_r - (1.0 / tau) * log_sum

def _objective(params, tau):
    """
    Objective function for optimization.
    params: 22 values representing p_3 ... p_13.
    p_1 is fixed at (0,0), p_2 is fixed at (1,0).
    """
    # Reconstruct points
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    
    ratios = _compute_ratios(pts)
    if ratios is None:
        return 1e9 # Penalty for degenerate hull
        
    # We want to maximize the min ratio.
    # So we minimize -min_ratio.
    # Using LogSumExp approximation.
    approx_min = _logsumexp_min(ratios, tau)
    return -approx_min

def _generate_initial_params(rng):
    """
    Generate initial parameters for the 22 free coordinates.
    Uses a regular 13-gon as a base, transformed to fit p1=(0,0), p2=(1,0).
    """
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Add some randomness to break symmetry
    angles += rng.uniform(-0.1, 0.1, N)
    radii = np.ones(N)
    radii += rng.normal(0, 0.05, N)
    
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # We need to transform pts such that:
    # new_pts[0] = (0,0)
    # new_pts[1] = (1,0)
    
    # Translation: subtract pts[0]
    pts = pts - pts[0]
    
    # Now pts[0] is (0,0).
    # Vector from 0 to 1: v = pts[1]
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        # Degenerate, try random
        return rng.uniform(-1, 1, 22)
    
    # Scale: divide by length
    pts = pts / length
    
    # Now pts[1] is on unit circle.
    # Rotate: we want pts[1] to be (1,0).
    # Current angle of pts[1] is theta.
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    
    # Rotation matrix R(-theta)
    c = np.cos(-theta)
    s = np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    
    pts = pts @ R.T
    
    # Now pts[0] should be (0,0) and pts[1] should be (1,0) approx.
    # Extract params for pts[2:]
    params = pts[2:].flatten()
    return params

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Annealing schedule for tau
    # Start with small tau (smooth) and increase (sharp)
    taus = [10, 50, 200, 1000]
    
    # Multiple restarts to escape local minima
    num_restarts = 5
    
    for restart in range(num_restarts):
        if (time.time() - start_time) > time_limit - 10.0:
            break
            
        # Generate initial guess
        if restart == 0:
            p0 = _generate_initial_params(rng)
        else:
            # Perturb the best found so far or generate new
            if best_params is not None and rng.random() < 0.5:
                p0 = best_params + rng.normal(0, 0.1, 22)
            else:
                p0 = _generate_initial_params(rng)
        
        current_params = p0
        
        for tau in taus:
            if (time.time() - start_time) > time_limit - 5.0:
                break
                
            try:
                # Use L-BFGS-B for efficient gradient-based optimization
                res = minimize(
                    _objective, 
                    current_params, 
                    args=(tau,),
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < _objective(current_params, tau):
                    current_params = res.x
                    
                # Update best
                # Compute true score (min ratio) for this configuration
                pts = np.zeros((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = current_params.reshape(N - 2, 2)
                
                ratios = _compute_ratios(pts)
                if ratios is not None:
                    score = np.min(ratios)
                    if score > best_score:
                        best_score = score
                        best_params = current_params.copy()
                        
            except Exception as e:
                pass

    # If no valid solution found, return a default regular polygon
    if best_params is None:
        best_params = _generate_initial_params(rng)
        
    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(N - 2, 2)
    
    return final_pts

# EVOLVE-BLOCK-END

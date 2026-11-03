"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Compute the packing radius for given centers."""
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r

def _rings_init(n):
    """
    Initialize with a concentric rings structure.
    For N=50, a good structure is 1, 6, 12, 18, 13 (or similar).
    We'll define radii and angles and optimize them.
    """
    # Known structure: 1 center, 6 ring, 12 ring, 18 ring, 13 ring?
    # Let's try: 1, 6, 12, 18, 13 = 50.
    # Or 1, 6, 12, 18, 13.
    # Let's define ring sizes.
    ring_sizes = [1, 6, 12, 18, 13]
    
    # Initial radii guesses (normalized)
    # The outermost ring should be near 1 - r.
    # Let's start with evenly spaced radii and optimize.
    num_rings = len(ring_sizes)
    # Radii for rings 1..num_rings-1 (ring 0 is center)
    # Let's guess radii: 0.3, 0.5, 0.7, 0.9
    radii = np.linspace(0.3, 0.9, num_rings - 1)
    
    # Angles: start with uniform distribution
    angles = []
    for size in ring_sizes:
        if size == 0:
            angles.append(np.array([]))
        elif size == 1:
            angles.append(np.array([0.0]))
        else:
            # Offset each ring slightly to interleave
            offset = (ring_sizes.index(size) * 0.1) 
            angles.append(np.linspace(0, 2*np.pi, size, endpoint=False) + offset)
    
    # Construct points
    points = []
    # Center
    points.append([0.0, 0.0])
    # Rings
    for i in range(1, num_rings):
        r = radii[i-1]
        for a in angles[i]:
            points.append([r * np.cos(a), r * np.sin(a)])
    
    points = np.array(points)
    return points

def _score_vec(points):
    """Vectorized score computation."""
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r

def _smooth_min(dists, b=50.0):
    """Compute a smooth approximation of the minimum of a set of distances."""
    # min(x) ≈ -1/b * log(sum(exp(-b*x)))
    # To avoid overflow, subtract max
    neg_dists = -dists
    max_neg = np.max(neg_dists)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_dists - max_neg)))
    return -log_sum / b

def _objective(params, b=50.0):
    """
    Objective function for L-BFGS-B.
    Minimizes the negative smooth minimum of all relevant distances.
    params: flattened array of 2*N coordinates.
    """
    pts = params.reshape(-1, 2)
    n = len(pts)
    
    # Boundary distances
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([boundary, pair_dists])
    
    # Smooth min
    smooth_min_val = _smooth_min(all_dists, b)
    
    # We want to maximize the min, so minimize -min
    return -smooth_min_val

def _gradient(params, b=50.0):
    """
    Numerical gradient (or analytic if possible, but numerical is safer for now).
    """
    # Using finite differences for simplicity and robustness
    eps = 1e-6
    grad = np.zeros_like(params)
    f0 = _objective(params, b)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus, b)
        grad[i] = (f_plus - f0) / eps
        params_plus[i] -= 2*eps
        f_minus = _objective(params_plus, b)
        grad[i] = (f_plus - f_minus) / (2*eps)
        params_plus[i] += eps # restore
        
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 20s safety margin
    
    # 1. Initialize with rings structure
    points = _rings_init(N)
    best_points = points.copy()
    best_score = _score_vec(best_points)
    
    # 2. Local optimization using L-BFGS-B
    # We will run multiple rounds with different b (sharpness) parameters
    # and perturbations.
    
    b_values = [10.0, 30.0, 100.0, 300.0]
    
    # First pass: optimize the ring structure directly?
    # Actually, let's just optimize the coordinates starting from the ring init.
    
    x0 = points.flatten()
    
    for b in b_values:
        if time.time() > deadline - 10.0:
            break
            
        # Objective function for this b
        def obj_func(params, b_val=b):
            return _objective(params, b_val)
        
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if time.time() > deadline - 10.0:
                break
                
            cand = res.x.reshape(-1, 2)
            if np.all(np.isfinite(cand)):
                cand_score = _score_vec(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
                    x0 = cand.flatten()
        except Exception as e:
            pass
    
    # 3. Multi-start local optimization with perturbations
    # Use the best point found so far as the center for perturbations
    n_restarts = 0
    max_restarts = 20
    
    while n_restarts < max_restarts and time.time() < deadline - 10.0:
        # Generate a new starting point
        if n_restarts == 0:
            # First restart: use current best
            x0 = best_points.flatten()
        else:
            # Perturb the best point
            # Scale perturbation down as we go
            perturbation_scale = 0.05 / (1.0 + 0.5 * n_restarts)
            x0 = best_points.flatten() + np.random.randn(2 * N) * perturbation_scale
        
        n_restarts += 1
        
        # Use a moderate b for these local refinements
        b = 100.0
        
        def obj_func(params, b_val=b):
            return _objective(params, b_val)
        
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', 
                           options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if time.time() > deadline - 10.0:
                break
                
            cand = res.x.reshape(-1, 2)
            if np.all(np.isfinite(cand)):
                cand_score = _score_vec(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
        except Exception:
            pass
    
    # 4. Final high-precision polish
    if time.time() < deadline - 5.0:
        x0 = best_points.flatten()
        b = 1000.0
        
        def obj_func(params, b_val=b):
            return _objective(params, b_val)
        
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
            
            cand = res.x.reshape(-1, 2)
            if np.all(np.isfinite(cand)):
                cand_score = _score_vec(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END

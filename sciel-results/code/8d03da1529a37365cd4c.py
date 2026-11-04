import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 13

# Precompute triangle indices
_TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))
_TRI_I = _TRI_IDX[:, 0]
_TRI_J = _TRI_IDX[:, 1]
_TRI_K = _TRI_IDX[:, 2]

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_tri_area(pts):
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    mta = _min_tri_area(pts)
    return mta / ha

def _soft_min(areas, tau):
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / tau)))
    return max_area - tau * log_sum_exp

def _objective_flat(flat_pts, temperature=0.01):
    pts = flat_pts.reshape(N, 2)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e10
    
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas < 1e-10):
        return 1e10
        
    smooth_min = _soft_min(areas, temperature)
    return -(smooth_min / ha)

def _gaussian_radius(angle, centers, weights, widths, base_radius):
    """
    Compute radius at a given angle using a sum of Gaussian bumps.
    angle: scalar or array of angles
    centers: array of center angles
    weights: array of weights for each center
    widths: array of widths for each center
    base_radius: constant base radius
    """
    # Handle periodicity by checking distance to center and center +/- 2pi
    # For simplicity in vectorized ops, we'll expand the centers
    # Actually, for 13 points, we can just loop or use a trick.
    # Let's use the trick: dist = min(|angle - center|, 2pi - |angle - center|)
    
    # Vectorized approach for a single angle or array of angles
    # angle: (M,)
    # centers: (K,)
    # diff: (M, K)
    diff = np.abs(angle[:, np.newaxis] - centers[np.newaxis, :])
    diff = np.minimum(diff, 2 * np.pi - diff)
    
    # weights: (K,), widths: (K,)
    # gaussians: (M, K)
    gaussians = weights[np.newaxis, :] * np.exp(-0.5 * (diff / widths[np.newaxis, :])**2)
    
    return base_radius + np.sum(gaussians, axis=1)

def _decode_basis(x, n_basis=5):
    """
    Decode parameter vector x into 13 points.
    x[0:5]: centers (angles)
    x[5:10]: weights (amplitudes)
    x[10:15]: widths (sigma)
    x[15]: base_radius
    x[16]: angular_offset (shifts all 13 points)
    
    The 13 points are placed at equally spaced angles (plus offset)
    with radii determined by the basis function.
    """
    centers = np.mod(x[:n_basis], 2 * np.pi)
    weights = x[n_basis:2*n_basis]
    widths = np.abs(x[2*n_basis:3*n_basis]) + 0.1 # Ensure positive widths
    base_radius = x[3*n_basis]
    angular_offset = x[3*n_basis + 1]
    
    # 13 equally spaced angles
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + angular_offset
    
    # Compute radii
    radii = _gaussian_radius(base_angles, centers, weights, widths, base_radius)
    
    # Ensure positive radii (just in case base_radius + weights goes negative, though unlikely with constraints)
    radii = np.maximum(radii, 0.1)
    
    pts = np.column_stack((radii * np.cos(base_angles), radii * np.sin(base_angles)))
    return pts

def _objective_basis(x, tau=0.05):
    pts = _decode_basis(x)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e10
        
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas < 1e-10):
        return 1e10
        
    smooth_min = _soft_min(areas, tau)
    return -(smooth_min / ha)

def _refine_2d(start_pts, deadline, max_iter=100):
    """
    Refine points in 2D using L-BFGS-B with soft min objective.
    """
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    temperatures = [0.1, 0.05, 0.02, 0.01]
    
    for temp in temperatures:
        if time.time() > deadline - 5:
            break
            
        def obj(x):
            return _objective_flat(x, temp)
            
        try:
            res = minimize(
                obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            opt_pts = res.x.reshape(N, 2)
            score = _score(opt_pts)
            
            if score > best_score:
                best_score = score
                best_pts = opt_pts
        except Exception:
            continue
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_pts = None
    best_score = 0.0
    
    # Strategy 1: Differential Evolution on Basis Parameters
    # This explores the space of smooth convex-like shapes
    n_basis = 5
    # Bounds:
    # Centers: [0, 2pi]
    # Weights: [-1, 1]
    # Widths: [0.1, 2] (will take abs + 0.1, so bound on raw is [-1.9, 1.9] effectively, or just [0, 2])
    # Base Radius: [0.5, 1.5]
    # Offset: [0, 2pi]
    
    bounds = [
        [(0, 2*np.pi)] * n_basis,       # Centers
        [(-1.0, 1.0)] * n_basis,        # Weights
        [(0.1, 2.0)] * n_basis,         # Widths (raw values, will be abs+0.1)
        [(0.5, 1.5)],                   # Base Radius
        [(0, 2*np.pi)]                  # Offset
    ]
    
    if time.time() < deadline - 30:
        try:
            de_res = differential_evolution(
                _objective_basis,
                bounds,
                seed=42,
                maxiter=30,
                tol=1e-6,
                polish=False, # We will polish manually
                init='sobol',
                popsize=15,
                updating='deferred'
            )
            
            if de_res.fun < 0: # Valid solution found
                x_opt = de_res.x
                pts_opt = _decode_basis(x_opt)
                score_opt = _score(pts_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
        except Exception:
            pass
    
    # Strategy 2: Refine the best basis result (or a regular polygon if DE failed) in 2D
    if best_pts is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_pts)
        
    if time.time() < deadline - 10:
        best_pts, best_score = _refine_2d(best_pts, deadline)
        
    # Strategy 3: Try a few random 2D restarts and refine
    # If time permits, explore the 2D space more broadly
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for i in range(5):
            if time.time() > deadline - 10:
                break
            # Random points in unit disk
            r = np.sqrt(rng.random(N))
            theta = rng.random(N) * 2 * np.pi
            pts_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
            
            # Quick score check
            s = _score(pts_rand)
            if s > best_score * 0.8: # Only refine if promising
                pts_ref, s_ref = _refine_2d(pts_rand, deadline, max_iter=50)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = pts_ref

    # Final sanity check
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_pts

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_score(points):
    # points: 8x2 array
    # 1. Scale to diameter 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Diameter is max distance
    np.fill_diagonal(dists, 0)
    D = np.max(dists)
    
    if D < 1e-9:
        return 1e10 # Degenerate, high penalty
        
    # Scale
    scale = 2.0 / D
    scaled_points = points * scale
    
    # Recalculate distances on scaled points
    diff_s = scaled_points[:, np.newaxis, :] - scaled_points[np.newaxis, :, :]
    dists_s = np.sqrt(np.sum(diff_s**2, axis=-1))
    np.fill_diagonal(dists_s, 0)
    
    # Product of ordered pairs
    # We want to maximize log(prod).
    # log(prod) = sum(log(d_ij)) for i != j
    mask = ~np.eye(8, dtype=bool)
    # Handle zeros if any points are identical (should not happen with distinct params)
    # But if r1=r2 and delta=0, points overlap.
    # We should avoid such params or handle log(0).
    # If any dist is 0, log is -inf. Score is 0.
    # We return -log_score, so if log_score is -inf, we return inf.
    
    # Check for zeros in dists_s[mask]
    min_d = np.min(dists_s[mask])
    if min_d < 1e-12:
        return 1e10
        
    log_prod = np.sum(np.log(dists_s[mask]))
    
    # Score = exp(log_prod) / 8^8
    # -log(Score) = -log_prod + 8*log(8)
    
    return -log_prod + 8 * np.log(8)

def _get_points(r1, r2, delta):
    # r1, r2 radii. delta relative rotation.
    # Square 1 at angle 0.
    # Square 2 at angle delta.
    p1 = np.array([
        [r1 * np.cos(0), r1 * np.sin(0)],
        [r1 * np.cos(np.pi/2), r1 * np.sin(np.pi/2)],
        [r1 * np.cos(np.pi), r1 * np.sin(np.pi)],
        [r1 * np.cos(3*np.pi/2), r1 * np.sin(3*np.pi/2)]
    ])
    
    p2 = np.array([
        [r2 * np.cos(delta), r2 * np.sin(delta)],
        [r2 * np.cos(delta + np.pi/2), r2 * np.sin(delta + np.pi/2)],
        [r2 * np.cos(delta + np.pi), r2 * np.sin(delta + np.pi)],
        [r2 * np.cos(delta + 3*np.pi/2), r2 * np.sin(delta + 3*np.pi/2)]
    ])
    
    return np.vstack([p1, p2])

def _compute_score_wrapper(params):
    try:
        r1, r2, delta = params
        if r1 < 0.1 or r2 < 0.1:
            return 1e10
        # Symmetry: delta is periodic with pi/2?
        # Rotating square 2 by pi/2 gives same set of points relative to square 1?
        # No, square 2 is a set of 4 points. Rotation by pi/2 maps the set to itself.
        # So delta and delta + pi/2 are equivalent.
        # We can restrict delta to [0, pi/2].
        # Also r1, r2 > 0.
        
        points = _get_points(r1, r2, delta)
        return _compute_score(points)
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_score = 1e10
    best_params = None
    
    # Initial guesses
    # r1, r2, delta
    guesses = [
        (1.0, 1.0, np.pi/4), # Regular Octagon
        (1.0, 0.5, np.pi/4),
        (1.0, 0.5, 0.0),
        (1.0, 0.8, np.pi/8),
        (1.0, 0.8, 3*np.pi/8),
        (1.0, 0.2, np.pi/4),
        (1.2, 0.8, np.pi/4),
        (0.8, 1.2, np.pi/4),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, np.pi/8),
        (1.0, 1.0, 3*np.pi/8),
    ]
    
    # Add random guesses
    np.random.seed(42)
    for _ in range(50):
        r1 = np.random.uniform(0.5, 1.5)
        r2 = np.random.uniform(0.5, 1.5)
        delta = np.random.uniform(0, np.pi/2)
        guesses.append((r1, r2, delta))
        
    for g in guesses:
        if time.time() > deadline:
            break
            
        bounds = [(0.1, 2.0), (0.1, 2.0), (0.0, np.pi/2)]
        
        try:
            # Use Powell for robustness
            res = minimize(_compute_score_wrapper, g, method='Powell', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-10, 'xtol': 1e-10})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            continue

    # If no optimizer succeeded or all failed, fall back to best initial guess
    if best_params is None:
        for g in guesses:
            s = _compute_score_wrapper(g)
            if s < best_score:
                best_score = s
                best_params = g

    r1, r2, delta = best_params
    points = _get_points(r1, r2, delta)
    
    # Final sanity check: scale to diameter 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, 0)
    D = np.max(dists)
    if D > 0:
        points = points * (2.0 / D)
        
    return [[float(p[0]), float(p[1])] for p in points]
# EVOLVE-BLOCK-END

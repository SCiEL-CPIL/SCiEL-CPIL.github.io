"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from math import cos, sin, pi

def construct():
    t_start = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n = 8

    # Strategy: Model 8 points as two concentric squares.
    # Square 1: radius r1, angle offset 0
    # Square 2: radius r2, angle offset theta
    # Global rotation is irrelevant due to scale/rotation invariance of the score.
    # We optimize r1, r2, theta.
    
    # Regular octagon corresponds to:
    # r1 = cos(pi/8), r2 = sin(pi/8) (or vice versa), theta = pi/4
    # Let's check:
    # Points at angles 0, pi/2, pi, 3pi/2 with radius R1
    # Points at angles pi/4, 3pi/4, 5pi/4, 7pi/4 with radius R2
    # For regular octagon on unit circle (diam 2), R1=R2=1.
    # Wait, regular octagon vertices are at angles k*pi/4.
    # If we split into two squares:
    # Square A: 0, pi/2, pi, 3pi/2
    # Square B: pi/4, 3pi/4, 5pi/4, 7pi/4
    # If R1=R2=1, this is a regular octagon.
    
    def get_points(params):
        r1, r2, theta = params
        # Ensure radii are positive
        r1 = max(r1, 1e-9)
        r2 = max(r2, 1e-9)
        
        pts = np.zeros((8, 2))
        # Square 1
        for k in range(4):
            ang = k * pi / 2
            pts[k] = [r1 * cos(ang), r1 * sin(ang)]
        
        # Square 2
        for k in range(4):
            ang = theta + k * pi / 2
            pts[4 + k] = [r2 * cos(ang), r2 * sin(ang)]
        
        return pts

    def score_from_points(pts):
        # Calculate product of distances for ordered pairs
        # Delta = prod_{i != j} dist(i, j)
        # Score = Delta / n^n
        # It is better to maximize log(Delta) / n - log(n) 
        # Or simply maximize log(Delta) since n is fixed.
        # Actually, the score definition is Delta / n^n.
        # Let's compute log_score = sum(log(dist)) - n*log(n).
        # Maximizing log_score is equivalent to maximizing score.
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, 1.0) # Avoid log(0)
        
        # Check for very small distances (degenerate)
        if np.min(d) < 1e-12:
            return -np.inf
            
        log_delta = np.sum(np.log(d))
        # Scale invariance check:
        # The problem statement says "after scaling the set to diameter 2".
        # So we must scale pts such that max_pairwise_dist == 2.
        
        diam = np.max(d)
        if diam < 1e-12:
            return -np.inf
            
        scale_factor = 2.0 / diam
        # Distances scale linearly with scale_factor
        # d_scaled = d * scale_factor
        # log(d_scaled) = log(d) + log(scale_factor)
        # Sum over n*(n-1) terms:
        # log_delta_scaled = log_delta + n*(n-1)*log(scale_factor)
        
        log_delta_scaled = log_delta + n * (n - 1) * np.log(scale_factor)
        log_score = log_delta_scaled - n * np.log(n)
        
        return log_score

    def objective(params):
        # We want to maximize score, so minimize -score
        try:
            pts = get_points(params)
            # Check finite
            if not np.all(np.isfinite(pts)):
                return 1e10
            s = score_from_points(pts)
            if not np.isfinite(s):
                return 1e10
            return -s
        except Exception:
            return 1e10

    # Bounds for r1, r2, theta
    # Since we scale to diameter 2, the absolute scale doesn't matter, 
    # but the ratio r1/r2 and theta do.
    # We can fix one radius or just let DE find the ratio.
    # Let's bound r1, r2 in [0.1, 2.0] and theta in [0, pi/2] (symmetry)
    bounds = [(0.1, 2.0), (0.1, 2.0), (0.0, pi/2)]
    
    best_result = None
    best_score_val = -np.inf
    best_params = None

    # Run differential evolution
    # Use multiple seeds for robustness
    seeds = [42, 123, 456, 789]
    
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        try:
            res = differential_evolution(
                objective,
                bounds,
                seed=seed,
                maxiter=300,
                popsize=30,
                tol=1e-12,
                polish=True,
                init='sobol',
                updating='deferred',
                workers=1
            )
            if res.fun < best_score_val:
                best_score_val = res.fun
                best_params = res.x
                best_result = res.x
        except Exception:
            continue

    if best_params is None:
        # Fallback: Regular Octagon
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    # Generate final points from best parameters
    pts = get_points(best_params)
    
    # Scale to diameter 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, 1.0)
    diam = np.max(d)
    if diam < 1e-10:
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    
    pts_scaled = pts * (2.0 / diam)
    
    # Verify distinctness
    diff_s = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :]
    d_s = np.sqrt(np.sum(diff_s * diff_s, axis=2))
    np.fill_diagonal(d_s, 1.0)
    if np.min(d_s) < 1e-10:
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        
    if np.max(np.abs(pts_scaled)) >= 1e6:
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    return pts_scaled.tolist()

# EVOLVE-BLOCK-END

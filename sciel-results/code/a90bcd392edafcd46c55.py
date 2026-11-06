"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi

def _score(points):
    """Compute Delta / n^n for 8 points."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    
    # Scale to diameter 2
    scaled_dists = dists * (2.0 / diam)
    
    # Mask out diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Delta is product of ordered pair distances
    # Score is Delta / n^n
    # log(Score) = sum_{i!=j} log(scaled_dists[i,j]) - n*log(n)
    log_delta = np.sum(np.log(scaled_dists[mask]))
    log_score = log_delta - n * np.log(n)
    
    return np.exp(log_score)

def _two_squares_points(ratio, angle_offset=0.0):
    """
    Generate 8 points forming two concentric squares.
    Outer square radius = 1.0
    Inner square radius = ratio
    Inner square rotated by 45 degrees (pi/4) relative to outer.
    angle_offset allows global rotation, which doesn't change score but helps if we 
    were to add more parameters, though for pure two-square symmetry it's irrelevant 
    to the score. We fix angle_offset=0 for simplicity.
    """
    # Outer square vertices
    angles_outer = np.array([0, pi/2, pi, 3*pi/2]) + angle_offset
    pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    # Inner square vertices, rotated by pi/4
    angles_inner = np.array([pi/4, 3*pi/4, 5*pi/4, 7*pi/4]) + angle_offset
    pts_inner = ratio * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([pts_outer, pts_inner])
    return pts

def _score_from_ratio(ratio):
    """Compute score for a given radius ratio of the two squares."""
    pts = _two_squares_points(ratio)
    return _score(pts)

def _neg_log_score_from_ratio(ratio):
    """
    Compute negative log-score for minimization.
    We need the gradient of log_score w.r.t ratio.
    
    Let points be p_k(r).
    Score S(r) = exp( sum_{i!=j} log(|p_i - p_j| * 2/D) - n log n )
    log S = sum_{i!=j} log|p_i - p_j| - n(n-1) log D + const
    
    d(log S)/dr = sum_{i!=j} ( (p_i - p_j) dot (dp_i/dr - dp_j/dr) ) / |p_i - p_j|^2 
                  - n(n-1) * (1/D) * dD/dr
    
    This is complex to derive analytically for the specific geometry.
    Instead, we can use finite differences or just use a scalar optimizer 
    on the 1D problem which is very fast and robust.
    """
    s = _score_from_ratio(ratio)
    if s <= 0:
        return 1e10
    return -np.log(s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_score = 0.0
    best_ratio = 1.0
    best_points = _two_squares_points(1.0) # Regular octagon is ratio=1
    
    # 1. Coarse Grid Search
    # The ratio r is the radius of the inner square relative to the outer.
    # r can range from 0 to 1 (if inner is smaller) or > 1.
    # By symmetry, swapping inner/outer and adjusting angles might yield similar structures,
    # but the specific "two squares" topology usually implies one is rotated.
    # Let's search r in [0.1, 1.5] to be safe.
    
    # Actually, if r > 1, we can just swap the roles. 
    # Let's define outer radius R_out and inner R_in.
    # My _two_squares_points fixes R_out=1 and R_in=ratio.
    # If the optimal has R_in > R_out, we can capture it by ratio > 1.
    # But wait, if ratio > 1, the "inner" square becomes the outer one.
    # The geometry is just two squares with radii r1, r2 and angle diff pi/4.
    # So searching r in (0, infinity) covers all two-square configs with pi/4 offset.
    
    # Let's do a coarse scan first
    r_coarse = np.linspace(0.1, 2.0, 100)
    for r in r_coarse:
        if time.time() > deadline - 5.0:
            break
        s = _score_from_ratio(r)
        if s > best_score:
            best_score = s
            best_ratio = r
            best_points = _two_squares_points(r)
    
    # 2. Fine Grid Search around the best coarse point
    if time.time() < deadline - 10.0:
        r_min = max(0.01, best_ratio - 0.1)
        r_max = min(2.0, best_ratio + 0.1)
        r_fine = np.linspace(r_min, r_max, 500)
        for r in r_fine:
            if time.time() > deadline - 5.0:
                break
            s = _score_from_ratio(r)
            if s > best_score:
                best_score = s
                best_ratio = r
                best_points = _two_squares_points(r)
        
        # 3. Local Optimization using BFGS on the 1D variable
        if time.time() < deadline - 5.0:
            try:
                # Minimize negative log score
                result = minimize(_neg_log_score_from_ratio, best_ratio, method='BFGS',
                                  options={'gtol': 1e-10, 'maxiter': 1000})
                if result.success:
                    r_opt = result.x[0]
                    # Clamp to valid range
                    if r_opt > 0.01 and r_opt < 5.0:
                        s_opt = _score_from_ratio(r_opt)
                        if s_opt > best_score:
                            best_score = s_opt
                            best_ratio = r_opt
                            best_points = _two_squares_points(r_opt)
            except Exception:
                pass

    # 4. Check if the regular octagon (ratio=1, but actually ratio=1 with pi/4 offset IS the regular octagon)
    # Wait, if ratio=1 and offset=pi/4, the points are at angles 0, pi/4, pi/2... all radius 1.
    # That is exactly the regular octagon.
    # So we have already evaluated it.
    
    # 5. Consider other topologies?
    # The prompt mentions "paper's constructions show this is possible".
    # Often these are based on roots of unity or specific algebraic structures.
    # For n=8, the "two squares" is the most prominent symmetric competitor to the octagon.
    # Another possibility: 4 points on a circle, 4 on another, not necessarily squares.
    # But squares are the "most spread out" regular polygons for 4 points.
    
    # Let's also try a generic 2-parameter search: 
    # 4 points at radius R1, angles 0, pi/2, pi, 3pi/2
    # 4 points at radius R2, angles phi, phi+pi/2, phi+pi, phi+3pi/2
    # This generalizes the two-squares model by allowing the inner square to rotate arbitrarily.
    # Due to symmetry of the score (rotation invariant), phi should not matter if R1 != R2?
    # Actually, the relative angle matters.
    # Let's define:
    # Group 1: radius r1, angles 0, pi/2, pi, 3pi/2
    # Group 2: radius r2, angles theta, theta+pi/2, theta+pi, theta+3pi/2
    # We can fix theta=0 by rotating the whole system? No, if we rotate group 2, relative angle changes.
    # But we can fix the whole system rotation.
    # Let's search (r2, theta) with r1=1.
    
    def _two_squares_general(r2, theta):
        angles1 = np.array([0, pi/2, pi, 3*pi/2])
        pts1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
        
        angles2 = angles1 + theta
        pts2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        
        return np.vstack([pts1, pts2])
    
    def _score_general(params):
        r2, theta = params
        if r2 < 0.01:
            return 0.0
        pts = _two_squares_general(r2, theta)
        return _score(pts)
    
    def _neg_log_score_general(params):
        s = _score_general(params)
        if s <= 0:
            return 1e10
        return -np.log(s)
    
    # Use the previous best as a starting point for the general search
    # Previous best was theta=pi/4 (since _two_squares_points used pi/4 offset)
    # Wait, _two_squares_points used:
    # Outer: 0, pi/2, pi, 3pi/2
    # Inner: pi/4, 3pi/4...
    # So theta = pi/4.
    
    if time.time() < deadline - 10.0:
        # Start from the best 1D result
        x0 = [best_ratio, pi/4]
        
        # Also try a few other initial guesses for theta
        inits = [
            [best_ratio, pi/4],
            [best_ratio, 0.0],
            [1.0, pi/4],
            [0.5, pi/4],
            [1.5, pi/4]
        ]
        
        for init in inits:
            if time.time() > deadline - 5.0:
                break
            try:
                result = minimize(_neg_log_score_general, init, method='Nelder-Mead',
                                  options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000})
                if result.success:
                    r2_opt, theta_opt = result.x
                    if r2_opt > 0.01 and r2_opt < 5.0:
                        s_opt = _score_general([r2_opt, theta_opt])
                        if s_opt > best_score:
                            best_score = s_opt
                            best_points = _two_squares_general(r2_opt, theta_opt)
            except Exception:
                pass

    # Final check: ensure points are distinct and valid
    # The score function returns 0 if invalid, so best_points should be valid if best_score > 0
    
    # If we didn't find anything better than 1.0, return the regular octagon
    if best_score <= 1.0 + 1e-9:
        octagon = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        return octagon

    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END

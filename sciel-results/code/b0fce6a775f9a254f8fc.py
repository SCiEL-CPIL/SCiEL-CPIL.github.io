"""Optimized 8-point configuration maximizing product of distances at fixed diameter."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _score(points):
    """Compute Delta / 8^8 where Delta is product of all ordered pair distances after scaling to diameter 2."""
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    # Compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Diameter is max pairwise distance
    diam = np.max(dists)
    if diam < 1e-15:
        return -1e18
    # Scale to diameter 2
    scaled_dists = dists / diam * 2.0
    # Product over ordered pairs (i != j)
    # Use log to avoid overflow
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(scaled_dists[mask] + 1e-300))
    score = log_prod - n * np.log(n)  # log(Delta) - n*log(n) where Delta = prod, score = Delta/n^n
    return score

def _points_from_params(params):
    """Convert parameters to 8 points.
    
    We use a two-square manifold with 3 parameters:
    - The 8 points are arranged as two squares.
    - Square 1: 4 points at angles theta, theta+pi/2, theta+pi, theta+3pi/2 with radius r1
    - Square 2: 4 points at angles theta+pi/4, theta+3pi/4, theta+5pi/4, theta+7pi/4 with radius r2
    - We can also offset them or use different angles.
    
    Actually, let's use a more general 3-param family:
    Two squares, one rotated by pi/4 relative to the other, with different radii.
    But we need to fix the diameter to 2 for scoring.
    
    Let's parameterize:
    - r1, r2: radii of the two squares
    - angle_offset: rotation of the second square relative to the first
    
    Points:
    Square 1: (r1*cos(a), r1*sin(a)) for a in [0, pi/2, pi, 3pi/2]
    Square 2: (r2*cos(a+offset), r2*sin(a+offset)) for a in [pi/4, 3pi/4, 5pi/4, 7pi/4]
    """
    r1, r2, offset = params
    angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    angles2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]) + offset
    
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    return np.vstack([pts1, pts2])

def _neg_score(params):
    """Negative score for minimization."""
    try:
        pts = _points_from_params(params)
        return -_score(pts)
    except:
        return 1e18

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with regular octagon
    from math import cos, sin, pi
    regular = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    best_score = _score(regular)
    best_points = np.array(regular)
    
    # Phase 1: Differential evolution on 3-parameter two-square manifold
    # Bounds: r1, r2 in [0.1, 5], offset in [0, pi/2]
    bounds = [(0.1, 5.0), (0.1, 5.0), (0.0, np.pi/2)]
    
    def check_deadline():
        return time.time() > deadline
    
    if not check_deadline():
        try:
            result = differential_evolution(
                _neg_score, bounds,
                maxiter=200, popsize=20, tol=1e-10,
                seed=42, polish=True,
                callback=lambda x, convergence: convergence or check_deadline()
            )
            if result.success and not check_deadline():
                s = -result.fun
                if s > best_score:
                    best_score = s
                    best_points = _points_from_params(result.x)
        except Exception:
            pass
    
    # Phase 2: Full 16-parameter local optimization from best point
    if not check_deadline():
        # Flatten best_points to 16 params
        x0 = best_points.flatten()
        
        def _neg_score_full(x):
            pts = x.reshape(8, 2)
            return -_score(pts)
        
        def _grad_full(x):
            pts = x.reshape(8, 2)
            n = 8
            # Compute gradient numerically
            eps = 1e-8
            grad = np.zeros_like(x)
            for i in range(16):
                xp = x.copy()
                xp[i] += eps
                xm = x.copy()
                xm[i] -= eps
                grad[i] = (_neg_score_full(xp) - _neg_score_full(xm)) / (2 * eps)
            return grad
        
        try:
            res = minimize(
                _neg_score_full, x0,
                method='L-BFGS-B',
                jac=_grad_full,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if not check_deadline():
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_points = res.x.reshape(8, 2)
        except Exception:
            pass
    
    # Phase 3: High-precision polish with tighter tolerance
    if not check_deadline():
        x0 = best_points.flatten()
        
        def _neg_score_full(x):
            pts = x.reshape(8, 2)
            return -_score(pts)
        
        try:
            res = minimize(
                _neg_score_full, x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = res.x.reshape(8, 2)
        except Exception:
            pass
    
    # Return the best points found
    # Ensure coordinates are within bounds
    result = np.clip(best_points, -1e6, 1e6)
    return result.tolist()
# EVOLVE-BLOCK-END

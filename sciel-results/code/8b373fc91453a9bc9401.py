"""Optimized 8-point set maximizing product of distances at fixed diameter."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

def _score(points):
    """Compute Delta/n^n where Delta is product of all ordered pair distances,
    after scaling so diameter = 2."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    # Compute all pairwise distances
    # Use broadcasting for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    # Scale so diameter = 2
    scale = 2.0 / diam
    dists_scaled = dists * scale
    # Product over ordered pairs (i,j) where i != j
    # That's 8*7 = 56 terms
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(dists_scaled[mask] + 1e-300))
    score = math.exp(log_prod / (n ** n))
    return score

def _log_score(points):
    """Log version for optimization (more stable)."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam < 1e-15:
        return -1e10
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(dists_scaled[mask] + 1e-300))
    return log_prod / (n ** n)

def construct():
    """Return 8 distinct points maximizing the product of distances at fixed diameter."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin
    n = 8
    
    best_score = -1e10
    best_points = None
    
    # Initial: regular octagon
    angles_reg = 2 * np.pi * np.arange(n) / n
    pts_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    s_reg = _score(pts_reg)
    best_score = s_reg
    best_points = pts_reg.copy()
    
    # Strategy: optimize angular gaps on a circle, then also try general 2D optimization
    # Phase 1: Angular optimization on unit circle
    # Represent as 8 angles, optimize to maximize log score
    def neg_log_score_angles(theta):
        if time.time() > deadline - 5:
            return 1e10
        pts = np.column_stack([np.cos(theta), np.sin(theta)])
        return -_log_score(pts)
    
    # Try multiple random initializations for angular optimization
    rng = np.random.default_rng(42)
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        if attempt == 0:
            theta0 = angles_reg.copy()
        else:
            # Random perturbation of regular octagon
            theta0 = angles_reg + rng.normal(0, 0.1, n)
            theta0 = np.sort(theta0)
        
        try:
            res = minimize(neg_log_score_angles, theta0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            if res.success or res.fun < -best_score + 1e-6:
                pts = np.column_stack([np.cos(res.x), np.sin(res.x)])
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 2: General 2D optimization
    # Flatten 8 points into 16D vector
    def neg_log_score_2d(x):
        if time.time() > deadline - 5:
            return 1e10
        pts = x.reshape(n, 2)
        return -_log_score(pts)
    
    # Start from best angular solution
    x0 = best_points.flatten()
    
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
        
        if attempt == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + rng.normal(0, 0.05, 16)
        
        try:
            res = minimize(neg_log_score_2d, x_init, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            pts = res.x.reshape(n, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 3: Try asymmetric configurations
    # Some known constructions use non-circular arrangements
    # Try configurations with some points clustered and others spread out
    
    # Pattern: 4 points near one end, 4 near other end (like two clusters)
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        
        # Two-cluster configuration
        if attempt == 0:
            # Square-like: 4 at corners of a square, 4 slightly offset
            base = np.array([
                [1, 0], [0, 1], [-1, 0], [0, -1],
                [0.7, 0.7], [0.7, -0.7], [-0.7, -0.7], [-0.7, 0.7]
            ])
        elif attempt == 1:
            # Regular heptagon + center-ish point
            angles = 2 * np.pi * np.arange(7) / 7
            pts7 = np.column_stack([np.cos(angles), np.sin(angles)])
            base = np.vstack([pts7, [[0.2, 0.1]]])
        elif attempt == 2:
            # Two squares rotated
            a1 = 2 * np.pi * np.arange(4) / 4
            a2 = a1 + np.pi / 4
            base = np.vstack([
                np.column_stack([np.cos(a1), np.sin(a1)]),
                np.column_stack([np.cos(a2), np.sin(a2)])
            ])
        else:
            base = rng.normal(0, 0.5, (n, 2))
        
        x_init = base.flatten()
        try:
            res = minimize(neg_log_score_2d, x_init, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            pts = res.x.reshape(n, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        x_init = best_points.flatten()
        try:
            res = minimize(neg_log_score_2d, x_init, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
            pts = res.x.reshape(n, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Ensure distinct points and valid coordinates
    pts = best_points
    # Check distinctness
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                # Nudge apart
                pts[j] += [1e-4, 0]
    
    # Ensure |coordinate| < 1e6
    pts = np.clip(pts, -1e5, 1e5)
    
    return pts.tolist()
# EVOLVE-BLOCK-END

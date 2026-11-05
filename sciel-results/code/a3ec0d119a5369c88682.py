"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import cos, sin, pi, log, exp, sqrt
import time
from scipy.optimize import minimize, differential_evolution

def _score(points):
    """Compute Delta/n^n where Delta is product over ordered pairs of distances after scaling to diameter 2."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    # Compute all pairwise distances
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = np.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
    # Diameter is max distance
    diam = np.max(d)
    if diam < 1e-15:
        return 0.0
    # Scale to diameter 2
    scale = 2.0 / diam
    # Product over ordered pairs (i != j) of scaled distances
    # This is (scale)^{n*(n-1)} * product_{i!=j} d[i,j]
    # Use log to avoid overflow
    log_prod = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                if d[i,j] > 1e-300:
                    log_prod += log(d[i,j])
    log_delta = n*(n-1)*log(scale) + log_prod
    # Score = Delta / n^n
    score = exp(log_delta - n*log(n))
    return score

def _negative_score(params):
    """Objective for minimization: negative of the score."""
    n = 8
    points = params.reshape(n, 2)
    # Check distinctness
    for i in range(n):
        for j in range(i+1, n):
            if np.sqrt((points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2) < 1e-10:
                return 1e10
    s = _score(points)
    return -s

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    best_score = 1.0
    best_points = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    
    # Start with regular octagon
    octagon = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)], dtype=float)
    
    # Try multiple random initializations with local optimization
    np.random.seed(42)
    n_init = 20
    for init_idx in range(n_init):
        if time.time() > deadline - 20:
            break
        
        if init_idx == 0:
            pts = octagon.copy()
        else:
            # Random perturbation of octagon
            pts = octagon.copy()
            noise_scale = 0.1 + 0.3 * (init_idx / n_init)
            pts += np.random.randn(8, 2) * noise_scale
        
        # Local optimization using L-BFGS-B
        try:
            result = minimize(_negative_score, pts.flatten(), method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success or result.fun < -best_score:
                s = -result.fun
                if s > best_score:
                    best_score = s
                    best_points = result.x.reshape(8, 2).tolist()
        except Exception:
            pass
    
    # Try differential evolution for global search
    if time.time() < deadline - 40:
        try:
            bounds = [(-2, 2)] * 16
            de_result = differential_evolution(_negative_score, bounds, maxiter=100, 
                                              popsize=15, tol=1e-10, seed=42, 
                                              polish=True)
            if de_result.fun < -best_score:
                s = -de_result.fun
                if s > best_score:
                    best_score = s
                    best_points = de_result.x.reshape(8, 2).tolist()
        except Exception:
            pass
    
    # Final polish with high-precision local optimization
    if time.time() < deadline - 10:
        try:
            final_result = minimize(_negative_score, np.array(best_points).flatten(), 
                                  method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if final_result.fun < -best_score:
                s = -final_result.fun
                if s > best_score:
                    best_score = s
                    best_points = final_result.x.reshape(8, 2).tolist()
        except Exception:
            pass
    
    # Ensure coordinates are within bounds and points are distinct
    result_pts = []
    for p in best_points:
        x, y = float(p[0]), float(p[1])
        if abs(x) >= 1e6 or abs(y) >= 1e6:
            x = max(-1e6 + 1, min(1e6 - 1, x))
            y = max(-1e6 + 1, min(1e6 - 1, y))
        result_pts.append([x, y])
    
    # Verify distinctness
    for i in range(8):
        for j in range(i+1, 8):
            dist = sqrt((result_pts[i][0]-result_pts[j][0])**2 + (result_pts[i][1]-result_pts[j][1])**2)
            if dist < 1e-6:
                # Fallback to octagon if points are too close
                return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    
    return result_pts
# EVOLVE-BLOCK-END

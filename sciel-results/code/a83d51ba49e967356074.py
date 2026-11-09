import numpy as np
import time
from scipy.optimize import minimize
import sys

def _vertex_score(d):
    """Compute score for a single vertex given sorted distances to 19 other vertices."""
    if len(d) != 19:
        return 0.0
    if np.any(d < 0.05):
        return 0.0
    diffs = np.diff(d)
    k = int(np.argmin(diffs))
    D = 0.5 * (d[k] + d[k+1])
    if D < 0.1:
        return 0.0
    if k == 0:
        sel = d[:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k-1:k+3]
    ratios = np.maximum(sel / D, D / sel)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _polygon_score(points):
    """Compute the minimum vertex score for a convex 20-gon. Returns 0.0 if invalid."""
    n = len(points)
    centroid = np.mean(points, axis=0)
    p = points - centroid
    dists_c = np.linalg.norm(p, axis=1)
    mean_d = np.mean(dists_c)
    if mean_d <= 0.01:
        return 0.0
    p = p / mean_d
    
    # Check for duplicate points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-6:
                return 0.0
    
    # Check convexity: all cross products same sign, nonzero
    edges = np.roll(p, -1, axis=0) - p
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-8):
        return 0.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0
    
    # Compute all pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    
    min_score = 1.0
    for i in range(n):
        other_dists = np.delete(dist_mat[i], i)
        sorted_d = np.sort(other_dists)
        s = _vertex_score(sorted_d)
        if s < min_score:
            min_score = s
            if min_score == 0.0:
                return 0.0
    return min_score

def _params_to_points(theta, r):
    """Convert angles and radii to 2D points."""
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

def _objective(params):
    """Objective function: negative score (to minimize)."""
    n = 20
    theta = params[:n]
    r = params[n:]
    if np.any(r <= 0.05):
        return 10.0  # Large penalty
    points = _params_to_points(theta, r)
    score = _polygon_score(points)
    return -score

def _regular_polygon():
    n = 20
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    start = time.time()
    deadline = 620.0
    
    # Start with regular polygon
    best_points = _regular_polygon()
    best_score = _polygon_score(best_points)
    
    n = 20
    base_theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Strategy 1: Optimize radii only (fixed angles)
    for restart in range(30):
        if time.time() > deadline - 30:
            break
        r0 = 1.0 + 0.3 * np.random.randn(n)
        r0 = np.maximum(r0, 0.1)
        
        def obj_r(r):
            if np.any(r <= 0.05):
                return 10.0
            pts = _params_to_points(base_theta, r)
            return -_polygon_score(pts)
        
        try:
            res = minimize(obj_r, r0, method='L-BFGS-B', bounds=[(0.05, 3.0)]*n,
                          options={'maxiter': 100, 'ftol': 1e-8})
            if res.fun < -best_score:
                pts = _params_to_points(base_theta, res.x)
                s = _polygon_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass
    
    # Strategy 2: Optimize both angles and radii
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        theta0 = base_theta + 0.1 * np.random.randn(n)
        r0 = 1.0 + 0.2 * np.random.randn(n)
        r0 = np.maximum(r0, 0.1)
        params0 = np.concatenate([theta0, r0])
        
        try:
            res = minimize(_objective, params0, method='L-BFGS-B', 
                          bounds=[(-np.pi, np.pi)]*n + [(0.05, 3.0)]*n,
                          options={'maxiter': 80, 'ftol': 1e-8})
            if res.fun < -best_score:
                theta_opt = res.x[:n]
                r_opt = res.x[n:]
                pts = _params_to_points(theta_opt, r_opt)
                s = _polygon_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass
    
    # Strategy 3: Nelder-Mead on radii with perturbed angles
    for restart in range(15):
        if time.time() > deadline - 15:
            break
        theta_pert = base_theta + 0.05 * np.random.randn(n)
        r0 = 1.0 + 0.15 * np.random.randn(n)
        r0 = np.maximum(r0, 0.1)
        
        def obj_nm(r):
            if np.any(r <= 0.05):
                return 10.0
            pts = _params_to_points(theta_pert, r)
            return -_polygon_score(pts)
        
        try:
            res = minimize(obj_nm, r0, method='Nelder-Mead',
                          options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 300, 'maxfev': 1000})
            if res.fun < -best_score:
                pts = _params_to_points(theta_pert, res.x)
                s = _polygon_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass
    
    return best_points

if __name__ == "__main__":
    pts = equidistant_points_polygon20()
    print("Score:", _polygon_score(pts))

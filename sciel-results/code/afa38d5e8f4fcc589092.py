import numpy as np
from scipy.optimize import differential_evolution
import math

def construct():
    n = 8
    
    def compute_score(pts):
        """Compute the score for a set of points, scaling to diameter 2 first."""
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        max_d = np.max(dists)
        
        if max_d < 1e-10:
            return 0.0
        
        pts_scaled = pts * (2.0 / max_d)
        
        diff = pts_scaled[:, np.newaxis] - pts_scaled[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        
        dists = np.maximum(dists, 1e-10)
        
        log_delta = np.sum(np.log(dists[mask]))
        log_score = log_delta - n * math.log(n)
        score = math.exp(log_score)
        
        return score

    def objective_from_params(params):
        """
        params: 8 values = [r0, a0, r1, a1, r2, a2, r3, a3]
        where (r_k, a_k) defines one point of antipodal pair k.
        The 8 points are: (r_k cos a_k, r_k sin a_k) and (-r_k cos a_k, -r_k sin a_k)
        """
        pts = np.zeros((n, 2))
        for k in range(4):
            r = params[2*k]
            a = params[2*k + 1]
            x = r * math.cos(a)
            y = r * math.sin(a)
            pts[2*k] = [x, y]
            pts[2*k + 1] = [-x, -y]
        
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        max_dist = np.max(dists)
        
        if max_dist < 1e-10:
            return 1e10
        
        pts_unit = pts / max_dist
        
        diff = pts_unit[:, np.newaxis] - pts_unit[np.newaxis, :]
        dists_unit = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        
        dists_unit = np.maximum(dists_unit, 1e-10)
        
        log_prod = np.sum(np.log(dists_unit[mask]))
        
        return -log_prod

    # Bounds for differential evolution
    # r in [0, 1], a in [0, 2*pi)
    bounds = []
    for k in range(4):
        bounds.append((0.0, 1.0))    # radius
        bounds.append((0.0, 2*math.pi))  # angle

    best_score = -np.inf
    best_pts = None

    # Run differential evolution multiple times with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        try:
            result = differential_evolution(
                objective_from_params,
                bounds,
                seed=seed,
                maxiter=1000,
                popsize=30,
                tol=1e-10,
                atol=1e-10,
                init='sobol',
                polish=True
            )
            if result.fun < 1e10:
                params = result.x
                pts = np.zeros((n, 2))
                for k in range(4):
                    r = params[2*k]
                    a = params[2*k + 1]
                    x = r * math.cos(a)
                    y = r * math.sin(a)
                    pts[2*k] = [x, y]
                    pts[2*k + 1] = [-x, -y]
                
                score = compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts
        except Exception:
            pass

    # Also try without antipodal constraint - full 16-dim search with DE
    bounds_full = [(0.0, 1.0)] * (n * 2)
    
    def objective_full(params):
        pts = params.reshape(n, 2)
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        max_dist = np.max(dists)
        
        if max_dist < 1e-10:
            return 1e10
        
        pts_unit = pts / max_dist
        
        diff = pts_unit[:, np.newaxis] - pts_unit[np.newaxis, :]
        dists_unit = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        
        dists_unit = np.maximum(dists_unit, 1e-10)
        
        log_prod = np.sum(np.log(dists_unit[mask]))
        
        return -log_prod

    for seed in [42, 123, 456]:
        try:
            result = differential_evolution(
                objective_full,
                bounds_full,
                seed=seed,
                maxiter=800,
                popsize=25,
                tol=1e-10,
                atol=1e-10,
                init='sobol',
                polish=True
            )
            if result.fun < 1e10:
                pts = result.x.reshape(n, 2)
                score = compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts
        except Exception:
            pass

    # Fallback to regular octagon
    if best_pts is None or best_score <= 1.0:
        return [[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]
        
    return best_pts.tolist()

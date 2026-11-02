"""Heilbronn 11-point problem: maximize minimum triangle area in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = 0.5 * 1.0 * S3H  # area of the equilateral triangle with side 1

# Precompute all (i,j,k) triples
_TRIPLES = list(combinations(range(N), 3))
_N_TRIPLES = len(_TRIPLES)  # 165

# Convert triple indices to flat indices for vectorized area computation
_I_IDX = np.array([t[0] for t in _TRIPLES])
_J_IDX = np.array([t[1] for t in _TRIPLES])
_K_IDX = np.array([t[2] for t in _TRIPLES])


def _min_area(points):
    """Compute minimum triangle area among all triples. points: (N,2) array."""
    p_i = points[_I_IDX]
    p_j = points[_J_IDX]
    p_k = points[_K_IDX]
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return np.min(areas)


def _all_areas(points):
    """Compute all triangle areas. points: (N,2) array."""
    p_i = points[_I_IDX]
    p_j = points[_J_IDX]
    p_k = points[_K_IDX]
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)


def _soft_min_area(points, beta):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    Returns -soft_min so we can minimize it.
    soft_min = -(1/beta) * log(sum(exp(-beta * areas)))
    """
    areas = _all_areas(points)
    # Stable log-sum-exp for -beta * areas
    max_arg = np.max(-beta * areas)
    log_sum_exp = max_arg + np.log(np.sum(np.exp(-beta * areas - max_arg)))
    soft_min = -log_sum_exp / beta
    return -soft_min  # negate to minimize


def _penalized_soft_min(x, beta):
    """
    Objective for DE: x is 2N barycentric coords (u_i, v_i).
    Convert to Cartesian, compute penalized soft-min area.
    """
    u = x[0::2]
    v = x[1::2]
    
    # Penalty for being outside the triangle (barycentric: u>=0, v>=0, u+v<=1)
    pen = 0.0
    pen += np.sum(np.maximum(0, -u)**2)
    pen += np.sum(np.maximum(0, -v)**2)
    pen += np.sum(np.maximum(0, u + v - 1.0)**2)
    pen *= 1000.0
    
    # Convert to Cartesian
    pts = np.empty((N, 2))
    pts[:, 0] = u + 0.5 * v
    pts[:, 1] = S3H * v
    
    obj = _soft_min_area(pts, beta)
    return obj + pen


def _epigraph_objective(x, beta):
    """
    Epigraph objective: x = [u0,v0,...,u10,v10, t]
    Minimize: -t + penalty
    where penalty encourages areas >= t
    """
    u = x[0::2]
    v = x[1::2]
    t = x[-1]
    
    # Convert to Cartesian
    pts = np.empty((N, 2))
    pts[:, 0] = u + 0.5 * v
    pts[:, 1] = S3H * v
    
    areas = _all_areas(pts)
    
    # Smooth penalty for areas below t
    # Use a smooth max(0, t - area) approximation
    violations = np.maximum(0.0, t - areas)
    penalty = np.sum(violations**2)
    
    # Barycentric constraint penalty
    bary_pen = 1000.0 * (
        np.sum(np.maximum(0, -u)**2) +
        np.sum(np.maximum(0, -v)**2) +
        np.sum(np.maximum(0, u + v - 1.0)**2)
    )
    
    return -t + penalty + bary_pen


def _local_optimize(x0, beta, max_iter=200):
    """
    Local optimization using L-BFGS-B on the epigraph formulation.
    x0: 2N+1 array [u0,v0,...,u10,v10,t]
    Returns (points, min_area) or (None, 0)
    """
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)]
    
    try:
        res = minimize(
            _epigraph_objective,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if res.x is not None:
            u = res.x[0::2]
            v = res.x[1::2]
            
            # Clamp to triangle
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            mask = (u + v) > 1.0
            if np.any(mask):
                s = u[mask] + v[mask]
                u[mask] /= s
                v[mask] /= s
            
            pts = np.empty((N, 2))
            pts[:, 0] = u + 0.5 * v
            pts[:, 1] = S3H * v
            
            # Verify inside triangle
            for i in range(N):
                px, py = pts[i]
                if px < -1e-9 or py < -1e-9 or px > 1.0 + 1e-9 or py > S3H + 1e-9:
                    return None, 0.0
                # Check barycentric
                w2 = py / S3H
                w1 = (px - 0.5 * py) / 0.5  # not quite right, let me redo
                # Actually: x = u + 0.5*v, y = S3H*v
                # v = y/S3H, u = x - 0.5*v
                v_i = py / S3H
                u_i = px - 0.5 * v_i
                if u_i < -1e-9 or v_i < -1e-9 or u_i + v_i > 1.0 + 1e-9:
                    return None, 0.0
            
            min_a = _min_area(pts)
            return pts, min_a
    except Exception:
        pass
    
    return None, 0.0


def construct():
    start_time = time.time()
    TIME_LIMIT = 140  # safety margin
    
    def _time_up():
        return (time.time() - start_time) > TIME_LIMIT
    
    best_points = None
    best_score = -float('inf')
    
    # Stage 1: Differential Evolution for global search
    bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        res_de = differential_evolution(
            _penalized_soft_min,
            bounds,
            args=(30.0,),
            seed=42,
            popsize=25,
            maxiter=150,
            tol=1e-8,
            polish=False,
            init='sobol'
        )
        
        if _time_up():
            if res_de.x is not None:
                u = res_de.x[0::2]
                v = res_de.x[1::2]
                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1)
                mask = (u + v) > 1.0
                if np.any(mask):
                    s = u[mask] + v[mask]
                    u[mask] /= s
                    v[mask] /= s
                pts = np.empty((N, 2))
                pts[:, 0] = u + 0.5 * v
                pts[:, 1] = S3H * v
                return pts
            return np.zeros((N, 2))
        
        best_x_de = res_de.x
    except Exception:
        best_x_de = np.random.RandomState(42).random(2 * N)
    
    # Stage 2: Local polish from multiple starting points
    candidates = [best_x_de.copy()]
    
    # Add perturbed versions
    rng = np.random.RandomState(123)
    for _ in range(5):
        perturbed = best_x_de + rng.normal(0, 0.02, 2 * N)
        candidates.append(perturbed)
    
    # Add a few random starts
    for _ in range(3):
        candidates.append(rng.random(2 * N))
    
    for x0 in candidates:
        if _time_up():
            break
        
        # Estimate t from the candidate
        u = x0[0::2]
        v = x0[1::2]
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        mask = (u + v) > 1.0
        if np.any(mask):
            s = u[mask] + v[mask]
            u[mask] /= s
            v[mask] /= s
        
        pts_tmp = np.empty((N, 2))
        pts_tmp[:, 0] = u + 0.5 * v
        pts_tmp[:, 1] = S3H * v
        t_est = _min_area(pts_tmp) * 0.9
        
        x_full = np.concatenate([x0, [t_est]])
        
        # Try with increasing beta for sharper approximation
        for beta in [20.0, 50.0, 100.0]:
            if _time_up():
                break
            
            pts, score = _local_optimize(x_full, beta, max_iter=300)
            if pts is not None and score > best_score:
                best_score = score
                best_points = pts.copy()
    
    # Stage 3: Final high-precision polish
    if best_points is not None and not _time_up():
        # Convert back to barycentric
        u = np.zeros(N)
        v = np.zeros(N)
        for i in range(N):
            px, py = best_points[i]
            v[i] = py / S3H
            u[i] = px - 0.5 * v[i]
        
        x_init = np.concatenate([np.column_stack([u, v]).flatten(), [best_score * 0.99]])
        
        for beta in [100.0, 200.0, 500.0]:
            if _time_up():
                break
            
            pts, score = _local_optimize(x_init, beta, max_iter=500)
            if pts is not None and score > best_score:
                best_score = score
                best_points = pts.copy()
                # Update x_init for next iteration
                u = np.zeros(N)
                v = np.zeros(N)
                for i in range(N):
                    px, py = best_points[i]
                    v[i] = py / S3H
                    u[i] = px - 0.5 * v[i]
                x_init = np.concatenate([np.column_stack([u, v]).flatten(), [best_score * 0.99]])
    
    if best_points is None:
        # Fallback: use DE result
        u = best_x_de[0::2]
        v = best_x_de[1::2]
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        mask = (u + v) > 1.0
        if np.any(mask):
            s = u[mask] + v[mask]
            u[mask] /= s
            v[mask] /= s
        best_points = np.empty((N, 2))
        best_points[:, 0] = u + 0.5 * v
        best_points[:, 1] = S3H * v
    
    return best_points

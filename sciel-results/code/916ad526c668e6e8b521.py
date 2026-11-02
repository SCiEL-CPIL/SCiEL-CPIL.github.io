"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import random
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, LinearConstraint

S3H = np.sqrt(3) / 2
TIME_LIMIT = 115  # seconds, safety margin

def _min_area_raw(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if cross < min_a:
                    min_a = cross
    return min_a

def _soft_min_area(points, beta=50.0):
    """
    Smooth surrogate for the minimum triangle area using log-sum-exp.
    Returns a smooth approximation of -min_area (to be minimized).
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                areas.append(cross)
    
    areas = np.array(areas)
    # Soft min: -log-sum-exp(-beta * areas) / beta approximates min(areas)
    # We want to maximize min, so minimize -soft_min
    max_val = np.max(areas)
    # Stable log-sum-exp
    log_sum_exp = max_val + np.log(np.sum(np.exp(-beta * (areas - max_val))))
    soft_min = -log_sum_exp / beta
    return -soft_min  # negate for minimization

def _epigraph_objective(x):
    """
    Epigraph formulation with auxiliary variable t.
    x = [u_0, v_0, u_1, v_1, ..., u_{N-1}, v_{N-1}, t]
    Minimize: -t
    Subject to: area(i,j,k) >= t for all triples
    We use a smooth penalty for the constraints.
    """
    u = x[0::2]
    v = x[1::2]
    t = x[-1]
    
    # Convert to Cartesian
    pts = np.zeros((N, 2))
    pts[:, 0] = u + 0.5 * v
    pts[:, 1] = S3H * v
    
    # Compute all triangle areas
    n = N
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                areas.append(cross)
    
    areas = np.array(areas)
    
    # Smooth penalty for area < t
    beta = 30.0
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations**2) + 0.01 * np.sum(np.exp(-beta * np.maximum(0, areas - t)))
    
    # Objective: minimize -t + penalty
    obj = -t + penalty
    
    # Penalty for invalid barycentric coords
    invalid_u = np.maximum(0, -u)
    invalid_v = np.maximum(0, -v)
    invalid_w = np.maximum(0, u + v - 1.0)
    obj += 100.0 * (np.sum(invalid_u**2) + np.sum(invalid_v**2) + np.sum(invalid_w**2))
    
    return obj

def _clamp_to_triangle(u, v):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        u[mask] /= sum_uv[mask]
        v[mask] /= sum_uv[mask]
    return u, v

def _bary_to_cart(u, v):
    """Convert barycentric coords to Cartesian."""
    x = u + 0.5 * v
    y = S3H * v
    return np.column_stack((x, y))

def _local_polish_epigraph(x0, start_time):
    """Polish a solution using L-BFGS-B with epigraph formulation."""
    # x0 should be 2N+1 dimensional: [u0, v0, ..., uN-1, vN-1, t]
    # If x0 is 2N dimensional, add t estimate
    if len(x0) == 2 * N:
        u = x0[0::2]
        v = x0[1::2]
        u, v = _clamp_to_triangle(u, v)
        pts = _bary_to_cart(u, v)
        t_est = _min_area_raw(pts) * 0.95
        x0 = np.append(x0, t_est)
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)]
    
    try:
        res = minimize(
            _epigraph_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if res.x is not None:
            u = res.x[0::2]
            v = res.x[1::2]
            u, v = _clamp_to_triangle(u, v)
            pts = _bary_to_cart(u, v)
            
            # Verify points are inside
            px = pts[:, 0]
            py = pts[:, 1]
            w0 = 1.0 - px - py / S3H
            w1 = px - py / (2.0 * S3H)
            w2 = py / S3H
            if np.min(w0) < -1e-6 or np.min(w1) < -1e-6 or np.min(w2) < -1e-6:
                return None, 0.0
            
            min_a = _min_area_raw(pts)
            return pts, min_a
    except Exception:
        pass
    
    return None, 0.0

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_points = None
    best_score = -float('inf')
    
    bounds = [(0.0, 1.0) for _ in range(2 * N)]
    
    # Stage 1: Differential Evolution for global search
    # Use the soft-min objective for smoother landscape
    def _de_objective(x):
        u = x[0::2]
        v = x[1::2]
        
        # Penalty for invalid points
        invalid_u = np.maximum(0, -u)
        invalid_v = np.maximum(0, -v)
        invalid_w = np.maximum(0, u + v - 1.0)
        penalty = 100.0 * (np.sum(invalid_u**2) + np.sum(invalid_v**2) + np.sum(invalid_w**2))
        
        # Convert to Cartesian
        pts = np.zeros((N, 2))
        pts[:, 0] = u + 0.5 * v
        pts[:, 1] = S3H * v
        
        # Use soft-min for smoother objective
        obj = _soft_min_area(pts, beta=40.0)
        return obj + penalty
    
    try:
        res = differential_evolution(
            _de_objective, 
            bounds, 
            seed=42, 
            popsize=30, 
            maxiter=200, 
            tol=1e-10, 
            polish=False,
            init='sobol'
        )
        
        if time.time() > start_time + TIME_LIMIT - 10:
            # Out of time, return best from DE if valid
            if res.x is not None:
                u = res.x[0::2]
                v = res.x[1::2]
                u, v = _clamp_to_triangle(u, v)
                pts = _bary_to_cart(u, v)
                return pts
            return np.zeros((N, 2))
            
        if res.x is not None:
            best_x_de = res.x
        else:
            best_x_de = np.random.random(2*N)
            
    except Exception as e:
        best_x_de = np.random.random(2*N)

    # Stage 2: Local Polish with epigraph formulation
    candidates = [best_x_de]
    
    # Add some random perturbations to explore local neighborhood
    for _ in range(3):
        perturbed = best_x_de + np.random.normal(0, 0.03, 2*N)
        candidates.append(perturbed)
        
    # Add a few fully random starts in case DE failed
    for _ in range(2):
        candidates.append(np.random.random(2*N))

    for x0 in candidates:
        if time.time() > start_time + TIME_LIMIT - 10:
            break
            
        pts, score = _local_polish_epigraph(x0, start_time)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Stage 3: High-precision final polish
    if best_points is not None and time.time() < start_time + TIME_LIMIT - 5:
        u = np.zeros(N)
        v = np.zeros(N)
        for i in range(N):
            # Convert back to barycentric
            px = best_points[i, 0]
            py = best_points[i, 1]
            v[i] = py / S3H
            u[i] = px - 0.5 * v[i]
        
        x_init = np.zeros(2*N + 1)
        x_init[0::2] = u
        x_init[1::2] = v
        x_init[-1] = best_score * 0.99
        
        bounds_final = [(0.0, 1.0)] * (2*N) + [(best_score * 0.9, best_score * 1.1)]
        
        try:
            res_final = minimize(
                _epigraph_objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds_final,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            
            if res_final.x is not None:
                u_f = res_final.x[0::2]
                v_f = res_final.x[1::2]
                u_f, v_f = _clamp_to_triangle(u_f, v_f)
                pts_f = _bary_to_cart(u_f, v_f)
                
                px = pts_f[:, 0]
                py = pts_f[:, 1]
                w0 = 1.0 - px - py / S3H
                w1 = px - py / (2.0 * S3H)
                w2 = py / S3H
                if np.min(w0) >= -1e-6 and np.min(w1) >= -1e-6 and np.min(w2) >= -1e-6:
                    min_a_f = _min_area_raw(pts_f)
                    if min_a_f > best_score:
                        best_score = min_a_f
                        best_points = pts_f.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback: Use DE result directly if it was valid
        if 'best_x_de' in locals():
            u = best_x_de[0::2]
            v = best_x_de[1::2]
            u, v = _clamp_to_triangle(u, v)
            best_points = _bary_to_cart(u, v)
        else:
            # Absolute fallback
            best_points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        break
                best_points[i, 0] = u + 0.5 * v
                best_points[i, 1] = S3H * v

    return best_points

# EVOLVE-BLOCK-END

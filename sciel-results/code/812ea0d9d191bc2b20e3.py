import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]

def _min_triangle_area_vectorized(points):
    """
    Compute the smallest area among all triangles formed by triples of points, vectorized.
    points: (N, 2) array
    Returns: float min area
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # Cross product magnitude for 2D vectors
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
        
    return np.min(areas)

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    """
    s = params[0::2]
    t = params[1::2]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V1[1] # Note: V1[1] is 0, so y = lam2 * V2[1]
    # Correction: y = lam1 * 0 + lam2 * (sqrt(3)/2)
    y = lam2 * V2[1]
    
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Inverse of param_to_points:
    # x = lam1 * 1 + lam2 * 0.5 = s(1-t) + 0.5 s t = s - 0.5 s t
    # y = lam2 * sqrt(3)/2 = s t * sqrt(3)/2
    # Let u = s t. Then y = u * sqrt(3)/2 => u = 2y/sqrt(3)
    # x = s - 0.5 u => s = x + 0.5 u = x + y/sqrt(3)
    # t = u / s = (2y/sqrt(3)) / (x + y/sqrt(3)) = 2y / (x*sqrt(3) + y)
    
    u = 2.0 * y / np.sqrt(3)
    s = x + 0.5 * u
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = u[mask] / s[mask]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    initial_params = []
    
    # 1. FPS initializations
    for i in range(5):
        if time_left() < 50.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # 2. Random initializations
    for i in range(5):
        if time_left() < 50.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    # 3. Structured: Vertices + Center + Midpoints
    # 3 vertices
    # 1 center
    # 3 midpoints of edges
    # 4 points at 1/3 and 2/3 along medians?
    # Let's just use a few symmetric patterns
    center = (V0 + V1 + V2) / 3.0
    pts_sym = [center]
    # 5 points on a circle around center, 5 on another
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.25
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        # Clip to triangle roughly
        p = _points_to_param(np.array([pt]))
        if p[0] <= 1.0 and p[1] <= 1.0 and p[0] >= 0 and p[1] >= 0:
             pts_sym.append(pt)
        else:
             # Project to nearest edge? Or just skip?
             # For simplicity, add it and let clipping handle it, or skip.
             # Let's add a valid point instead
             r1 = rng.uniform(0, 1); r2 = rng.uniform(0, 1)
             if r1+r2 > 1: r1=1-r1; r2=1-r2
             pts_sym.append(r1*V1 + r2*V2)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.5
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        p = _points_to_param(np.array([pt]))
        if p[0] <= 1.0 and p[1] <= 1.0 and p[0] >= 0 and p[1] >= 0:
             pts_sym.append(pt)
        else:
             r1 = rng.uniform(0, 1); r2 = rng.uniform(0, 1)
             if r1+r2 > 1: r1=1-r1; r2=1-r2
             pts_sym.append(r1*V1 + r2*V2)
             
    if len(pts_sym) == N:
        params = _points_to_param(np.array(pts_sym))
        initial_params.append(params)

    # Simulated Annealing with Periodic Polishing
    # Parameters for SA
    n_iter_sa = 2000  # Iterations per restart
    t0 = 0.05         # Initial temperature (scale of objective change)
    tf = 1e-6         # Final temperature
    alpha = (tf / t0) ** (1.0 / n_iter_sa)
    
    for start_p in initial_params:
        if time_left() < 30.0:
            break
            
        current_params = start_p.copy()
        current_score = -_objective(current_params)
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
            
        T = t0
        
        for i in range(n_iter_sa):
            if time_left() < 15.0:
                break
                
            # Generate a move
            # Large steps initially, smaller later
            step_size = 0.1 * (T / t0) + 0.01
            noise = rng.normal(0, step_size, size=2*N)
            new_params = np.clip(current_params + noise, 0.0, 1.0)
            
            new_score = -_objective(new_params)
            delta = new_score - current_score
            
            # Accept if better or with probability exp(delta/T)
            if delta > 0:
                current_params = new_params
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_params = current_params.copy()
                    
                    # Periodic Polish: If we found a new best, refine it with NM
                    # This helps escape shallow local minima
                    if time_left() > 20.0:
                        try:
                            res = minimize(
                                _objective, best_params, method='Nelder-Mead',
                                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                            )
                            polish_score = -res.fun
                            if polish_score > best_score:
                                best_score = polish_score
                                best_params = res.x.copy()
                                # Update current state to the polished one to continue from there
                                current_params = best_params.copy()
                                current_score = best_score
                        except Exception:
                            pass
            else:
                # Metropolis criterion
                if T > 0:
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_params = new_params
                        current_score = new_score
            
            T *= alpha
    
    # Final Polish of the global best
    if best_params is not None and time_left() > 10.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = np.zeros(2 * N)
        best_params[0::2] = 0.5
        best_params[1::2] = 0.5

    points = _param_to_points(best_params)
    
    # Final safety check: ensure all points are strictly inside or on the boundary
    def _clip_to_triangle(pts):
        for i in range(len(pts)):
            x, y = pts[i]
            l2 = 2 * y / np.sqrt(3)
            l1 = x - y / np.sqrt(3)
            l0 = 1 - l1 - l2
            
            if l0 < 0 or l1 < 0 or l2 < 0:
                l0 = max(0, l0)
                l1 = max(0, l1)
                l2 = max(0, l2)
                s = l0 + l1 + l2
                if s > 0:
                    l0 /= s
                    l1 /= s
                    l2 /= s
                else:
                    l0 = 1
                    l1 = 0
                    l2 = 0
                
                pts[i] = l0 * V0 + l1 * V1 + l2 * V2
        
        return pts

    points = _clip_to_triangle(points)
    
    return np.array(points)

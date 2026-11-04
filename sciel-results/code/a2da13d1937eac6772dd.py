"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])

# Precompute indices for all C(N, 3) triples
_IDX_I = None
_IDX_J = None
_IDX_K = None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
    return _IDX_I, _IDX_J, _IDX_K

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Compute all triangle areas, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    For each point i, we have params s_i, t_i in [0, 1].
    Barycentric coordinates (relative to V0, V1, V2) are:
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    lambda_3 = 1 - s_i
    Point = lambda_1 * V1 + lambda_2 * V2 + lambda_3 * V0
    Since V0 is (0,0), Point = lambda_1 * V1 + lambda_2 * V2
    """
    s = params[0::2]
    t = params[1::2]
    # Clip to ensure strictly inside or on boundary to avoid numerical issues at vertices
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    # Inverse of barycentric mapping
    # y = (lam1 * 0 + lam2 * sqrt(3)/2) => lam2 = 2*y/sqrt(3)
    # x = lam1 * 1 + lam2 * 0.5 => lam1 = x - 0.5*lam2
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _soft_min_objective(params, beta):
    """
    Smooth surrogate: negative soft-min of triangle areas.
    soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    # Numerically stable log-sum-exp
    min_a = np.min(areas)
    # exp(-beta * (areas - min_a)) is safe as areas - min_a >= 0
    exp_terms = np.exp(-beta * (areas - min_a))
    log_sum = np.log(np.sum(exp_terms))
    soft_min = min_a - log_sum / beta
    return -soft_min

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    This ensures points are well-distributed, which is crucial for maximizing min area.
    """
    points = np.zeros((n_points, 2))
    # Start with a random point or centroid? Centroid is safe.
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        # Generate random candidates
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        # Map to triangle
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        # Barycentric to Cartesian
        # p = (1-r1-r2)*V0 + r1*V1 + r2*V2
        # Since V0=0, p = r1*V1 + r2*V2
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        # Find candidate farthest from existing points
        # dist^2 to each existing point
        # shape: (n_candidates, k)
        dists = np.linalg.norm(candidates[:, np.newaxis, :] - points[:k, np.newaxis, :], axis=2)
        # min distance to any existing point
        min_dists = np.min(dists, axis=1)
        # pick the one with max min_dist
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.RandomState(42)
    
    best_score = 0.0
    best_params = None
    
    # Number of restarts. We will adjust based on time.
    # Let's aim for ~5-10 restarts depending on speed.
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # 1. Initialization
        # Alternate between FPS and random perturbation of best solution
        if restart == 0:
            init_points = _generate_fps_initialization(rng)
        elif best_params is not None:
            # Perturb best solution
            noise = 0.05 * rng.randn(2 * N)
            init_params = best_params + noise
            init_params = np.clip(init_params, 0, 1)
            init_points = _param_to_points(init_params)
        else:
            init_points = _generate_fps_initialization(rng)
            
        init_params = _points_to_param(init_points)
        
        # 2. Soft-Min Optimization Phase
        # Use a sequence of increasing beta values
        # Low beta: smooth landscape, finds global basin
        # High beta: sharp landscape, approaches true min
        betas = [5.0, 10.0, 20.0, 50.0, 100.0]
        
        current_params = init_params.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            # Define objective for this beta
            def obj_func(p):
                return _soft_min_objective(p, beta)
            
            # Use L-BFGS-B which handles bounds well
            # Bounds: all params in [0, 1]
            bounds = [(0.0, 1.0)] * (2 * N)
            
            try:
                res = minimize(
                    obj_func,
                    current_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.success:
                    current_params = res.x
            except Exception:
                pass
        
        # 3. Polish Phase with Exact Nelder-Mead
        # The soft-min solution is a good starting point for the exact problem
        # Nelder-Mead is derivative-free and robust for the final nonsmooth step
        def exact_obj(p):
            pts = _param_to_points(p)
            return -_min_triangle_area_vectorized(pts)
            
        try:
            res_nm = minimize(
                exact_obj,
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.success or exact_obj(res_nm.x) < exact_obj(current_params):
                current_params = res_nm.x
        except Exception:
            pass
        
        # Evaluate final score
        final_points = _param_to_points(current_params)
        score = _min_triangle_area_vectorized(final_points)
        
        if score > best_score:
            best_score = score
            best_params = current_params.copy()
            
    # If no optimization ran or failed, fallback to FPS
    if best_params is None:
        final_points = _generate_fps_initialization(rng)
    else:
        final_points = _param_to_points(best_params)
        
    # Final sanity check: project to triangle just in case of numerical drift
    # Though _param_to_points with clipping should ensure this.
    # Let's do a hard clip to be safe.
    # Points are already in [0,1] params, so they are inside.
    
    return final_points

# Quick test
if __name__ == "__main__":
    pts = construct()
    print(f"Number of points: {len(pts)}")
    min_area = _min_triangle_area_vectorized(pts)
    tri_area = 0.5 * 1.0 * (np.sqrt(3) / 2.0)
    print(f"Minimum triangle area: {min_area}")
    print(f"Normalized score: {min_area / tri_area}")
    print(f"Points:")
    for p in pts:
        print(f"  ({p[0]:.4f}, {p[1]:.4f})")

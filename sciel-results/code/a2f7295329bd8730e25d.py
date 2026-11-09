"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_triangles = len(_idx_i)

def project_to_triangle(p):
    """
    p: array of shape (N, 2)
    Returns projected points inside the triangle (0,0), (1,0), (0.5, H)
    """
    p = p.copy()
    # Clip to bounding box first
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, H)
    
    # Iterate a few times to handle corners properly
    for _ in range(3):
        # Edge AB: y >= 0
        mask_ab = p[:, 1] < 0.0
        if np.any(mask_ab):
            p[mask_ab, 1] = 0.0
            
        # Edge AC: y <= sqrt(3) x  =>  sqrt(3)x - y >= 0
        # Violation: y > sqrt(3)x
        mask_ac = p[:, 1] > np.sqrt(3) * p[:, 0]
        if np.any(mask_ac):
            px = p[mask_ac, 0]
            py = p[mask_ac, 1]
            # Projection of (px, py) onto line through origin with direction (1, sqrt(3))
            dot = px + py * np.sqrt(3)
            px_proj = dot / 4.0
            py_proj = dot * np.sqrt(3) / 4.0
            # Clamp to segment AC: x in [0, 0.5]
            px_proj = np.clip(px_proj, 0.0, 0.5)
            py_proj = px_proj * np.sqrt(3)
            p[mask_ac, 0] = px_proj
            p[mask_ac, 1] = py_proj
            
        # Edge BC: y <= sqrt(3)(1-x) => y + sqrt(3)x <= sqrt(3)
        # Violation: y > sqrt(3)(1-x)
        mask_bc = p[:, 1] > np.sqrt(3) * (1.0 - p[:, 0])
        if np.any(mask_bc):
            px = p[mask_bc, 0]
            py = p[mask_bc, 1]
            # Line eq: sqrt(3)x + y - sqrt(3) = 0.
            # Normal pointing inward: (sqrt(3), 1).
            d = (np.sqrt(3) * px + py - np.sqrt(3)) / 4.0
            px_proj = px - d * np.sqrt(3)
            py_proj = py - d * 1.0
            # Clamp to segment BC: x in [0.5, 1.0]
            px_proj = np.clip(px_proj, 0.5, 1.0)
            py_proj = np.sqrt(3) * (1.0 - px_proj)
            p[mask_bc, 0] = px_proj
            p[mask_bc, 1] = py_proj
            
    return p

def min_triangle_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    Vectorized calculation.
    """
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()

def objective_flat(x):
    """
    x: flattened array of shape (2*N,)
    Returns negative min area (to be minimized).
    """
    points = x.reshape(N, 2)
    points = project_to_triangle(points)
    return -min_triangle_area(points)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s buffer for 150s limit
    
    best_points = None
    best_score = -np.inf

    bounds = [(0.0, 1.0), (0.0, H)] * N

    # Phase 1: Differential Evolution
    # Using a relatively small popsize and maxiter to fit within time, 
    # relying on local polish afterwards.
    try:
        result = differential_evolution(
            objective_flat,
            bounds=bounds,
            maxiter=200,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False,
            updating='immediate',
            init='sobol'
        )
        if time.time() > deadline:
             # Fallback if DE takes too long (unlikely with these settings, but safe)
             if best_points is not None:
                 return best_points
             rng = np.random.RandomState(42)
             pts = []
             for _ in range(N):
                 while True:
                     u, v = rng.random(), rng.random()
                     if u + v < 1.0:
                         pts.append([u + 0.5 * v, H * v])
                         break
             return np.array(pts)
        
        best_params = result.x
        best_points = project_to_triangle(best_params.reshape(N, 2))
        best_score = min_triangle_area(best_points)
    except Exception:
        pass

    # Phase 2: Polish with Nelder-Mead (Adaptive)
    if time.time() < deadline - 30.0:
        try:
            res2 = minimize(
                objective_flat,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            curr_points = project_to_triangle(res2.x.reshape(N, 2))
            curr_score = min_triangle_area(curr_points)
            if curr_score > best_score:
                best_score = curr_score
                best_points = curr_points
        except Exception:
            pass

    # Phase 3: Random restarts with Nelder-Mead polish
    # We use random restarts to explore different local minima.
    # The number of restarts is dynamic based on remaining time.
    if time.time() < deadline - 15.0:
        rng = np.random.RandomState(123)
        # Estimate time per restart (roughly 1-2s for NM with 11 points)
        # We'll try to fit as many as possible, but cap it to avoid infinite loops if NM is stuck.
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline - 15.0:
                break
            
            # Generate random start points
            start_params = np.zeros(2 * N)
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        start_params[2 * i] = u + 0.5 * v
                        start_params[2 * i + 1] = H * v
                        break
            
            try:
                # Use a moderate number of iterations for restarts
                res_r = minimize(
                    objective_flat,
                    start_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                curr_points = project_to_triangle(res_r.x.reshape(N, 2))
                curr_score = min_triangle_area(curr_points)
                if curr_score > best_score:
                    best_score = curr_score
                    best_points = curr_points
            except Exception:
                pass

    # Phase 4: Final tight polish on the best found
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            start_params = best_points.flatten()
            res_final = minimize(
                objective_flat,
                start_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            curr_points = project_to_triangle(res_final.x.reshape(N, 2))
            curr_score = min_triangle_area(curr_points)
            if curr_score > best_score:
                best_score = curr_score
                best_points = curr_points
        except Exception:
            pass

    if best_points is not None:
        # Final safety check: ensure all points are strictly inside (or on boundary)
        # The projection should handle this, but let's verify.
        for i in range(N):
            x, y = best_points[i]
            if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9 or y > np.sqrt(3) * x + 1e-9 or y > np.sqrt(3) * (1 - x) + 1e-9:
                # If somehow outside, project again
                best_points[i] = project_to_triangle(best_points[i:i+1])[0]
        return best_points
    else:
        # Fallback
        rng = np.random.RandomState(0)
        pts = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 0.99:
                    pts.append([u + 0.5 * v, H * v])
                    break
        return np.array(pts)

# EVOLVE-BLOCK-END

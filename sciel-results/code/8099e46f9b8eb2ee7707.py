import numpy as np
import scipy.optimize as opt
import time

N = 11
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])

# Precompute indices for all triangles
i, j, k = np.indices((N, N, N)).reshape(3, -1).T
mask = (i < j) & (j < k)
idx_i = i[mask]
idx_j = j[mask]
idx_k = k[mask]

def min_triangle_area(points):
    p_i = points[idx_i]
    p_j = points[idx_j]
    p_k = points[idx_k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(params):
    uv = params.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Penalty for being outside the triangle (u,v,w >= 0)
    # Using a quadratic penalty to be smooth
    neg_u = np.maximum(0.0, -u)
    neg_v = np.maximum(0.0, -v)
    neg_w = np.maximum(0.0, -w)
    penalty = 1000.0 * (np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(neg_w**2))
    
    points = u[:, np.newaxis] * V[0] + v[:, np.newaxis] * V[1] + w[:, np.newaxis] * V[2]
    
    area = min_triangle_area(points)
    return -area + penalty

def gen_lattice(rows):
    pts = []
    total_rows = len(rows)
    for r_idx, count in enumerate(rows):
        w_val = (total_rows - 1 - r_idx) / total_rows
        rem = 1.0 - w_val
        for c in range(count):
            if count > 1:
                t = c / (count - 1)
            else:
                t = 0.5
            u_val = rem * (1.0 - t)
            v_val = rem * t
            pts.append([u_val, v_val])
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_score = -np.inf
    
    configs = [
        [1, 2, 3, 5],
        [1, 2, 4, 4],
        [2, 3, 3, 3],
        [1, 3, 3, 4]
    ]
    
    initial_guesses = []
    for cfg in configs:
        if sum(cfg) == N:
            initial_guesses.append(gen_lattice(cfg))
            
    # Add a slightly perturbed version of the best lattice
    if initial_guesses:
        base = initial_guesses[0]
        perturbed = base + np.random.default_rng(1).normal(0, 0.01, base.shape)
        initial_guesses.append(perturbed)

    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            res = opt.minimize(objective, guess.flatten(), method='Nelder-Mead', 
                               options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-12})
            
            uv = res.x.reshape(N, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1.0 - u - v
            
            # Check if valid (inside triangle)
            if np.all(u >= -1e-6) and np.all(v >= -1e-6) and np.all(w >= -1e-6):
                # Clip
                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1)
                w = 1.0 - u - v
                if np.all(w >= -1e-6):
                    points = u[:, np.newaxis] * V[0] + v[:, np.newaxis] * V[1] + w[:, np.newaxis] * V[2]
                    raw_score = min_triangle_area(points)
                    if raw_score > best_score:
                        best_score = raw_score
                        best_points = points
        except Exception:
            pass

    if best_points is None:
        # Fallback
        uv = initial_guesses[0]
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        best_points = u[:, np.newaxis] * V[0] + v[:, np.newaxis] * V[1] + w[:, np.newaxis] * V[2]
        
    return best_points

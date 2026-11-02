import numpy as np
from scipy.optimize import minimize
import time

N = 12

def compute_min_outer_side(inner_positions, inner_rotations, outer_center, outer_rotation, n_angles=360):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    # Support function of the union of inner hexagons (side=1)
    max_support = np.full(n_angles, -np.inf)
    for i in range(N):
        px, py = inner_positions[i]
        rot = inner_rotations[i]
        # For each theta, the support of unit hexagon centered at (px,py) with rotation rot:
        # h(theta) = px*cos(theta) + py*sin(theta) + max_k cos(theta - rot - k*pi/3)
        k = np.round((thetas - rot) / (np.pi / 3))
        vertex_angles = rot + k * (np.pi / 3)
        cos_vals = np.cos(thetas - vertex_angles)
        h = px * cos_t + py * sin_t + cos_vals
        max_support = np.maximum(max_support, h)
    
    # Outer hexagon: support in direction theta is:
    # h_outer(theta) = cx*cos(theta) + cy*sin(theta) + L * max_k cos(theta - alpha - k*pi/3)
    k_outer = np.round((thetas - outer_rotation) / (np.pi / 3))
    vertex_angles_outer = outer_rotation + k_outer * (np.pi / 3)
    cos_outer = np.cos(thetas - vertex_angles_outer)
    center_dot_outer = outer_center[0] * cos_t + outer_center[1] * sin_t
    
    # Need: center_dot_outer + L * cos_outer >= max_support for all theta
    # L >= (max_support - center_dot_outer) / cos_outer
    # cos_outer >= cos(pi/6) = sqrt(3)/2 > 0, so division is safe
    ratios = (max_support - center_dot_outer) / cos_outer
    return np.max(ratios)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial arrangement: hexagonal lattice pattern
    # For 12 unit hexagons, use a compact arrangement
    # 4-3-3-2 rows in hexagonal lattice
    e1 = np.array([2.0, 0.0])
    e2 = np.array([1.0, np.sqrt(3)])
    
    positions = []
    rows = [4, 3, 3, 2]
    for row_idx, n_in_row in enumerate(rows):
        for col in range(n_in_row):
            offset = 1.0 if row_idx % 2 == 1 else 0.0
            p = col * e1 + row_idx * e2 + np.array([offset, 0.0])
            positions.append(p)
    
    positions = np.array(positions)
    # Center
    positions -= positions.mean(axis=0)
    
    inner_positions = positions.copy()
    inner_rotations = np.zeros(N)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        inner_pos = params[:N*3].reshape(N, 3)[:, :2]
        inner_rot = params[:N*3].reshape(N, 3)[:, 2]
        outer_c = params[N*3:N*3+2]
        outer_r = params[N*3+2]
        
        L = compute_min_outer_side(inner_pos, inner_rot, outer_c, outer_r, n_angles=180)
        
        # Disjointness penalty: center distance >= 2.0
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(inner_pos[i] - inner_pos[j])
                if d < 2.0:
                    penalty += (2.0 - d) ** 2 * 50.0
        
        return L + penalty
    
    init_params = []
    for i in range(N):
        init_params.extend([inner_positions[i][0], inner_positions[i][1], 0.0])
    init_params.extend([0.0, 0.0, 0.0])
    init_params = np.array(init_params)
    
    best_params = init_params.copy()
    best_L = compute_min_outer_side(inner_positions, inner_rotations, np.zeros(2), 0.0, n_angles=180)
    
    result = minimize(objective, init_params, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if result.fun < best_L:
        best_L = result.fun
        best_params = result.x.copy()
    
    # Additional restarts
    if time.time() < deadline - 40:
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            np.random.seed(seed)
            perturbed = inner_positions + np.random.normal(0, 0.3, size=(N, 2))
            perts_rot = np.random.uniform(0, np.pi/3, size=N)
            
            p = []
            for i in range(N):
                p.extend([perturbed[i][0], perturbed[i][1], perts_rot[i]])
            p.extend([0.0, 0.0, 0.0])
            p = np.array(p)
            
            r = minimize(objective, p, method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if r.fun < best_L:
                best_L = r.fun
                best_params = r.x.copy()
    
    inner_pos = best_params[:N*3].reshape(N, 3)[:, :2]
    inner_rot = best_params[:N*3].reshape(N, 3)[:, 2]
    outer_c = best_params[N*3:N*3+2]
    outer_r = best_params[N*3+2]
    
    final_L = compute_min_outer_side(inner_pos, inner_rot, outer_c, outer_r, n_angles=360)
    
    inner = []
    for i in range(N):
        inner.append([float(inner_pos[i][0]), float(inner_pos[i][1]), float(np.degrees(inner_rot[i]))])
    
    return inner, (float(outer_c[0]), float(outer_c[1])), float(final_L), float(np.degrees(outer_r))


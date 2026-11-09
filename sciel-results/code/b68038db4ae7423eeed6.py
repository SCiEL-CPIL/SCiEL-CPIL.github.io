import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT. Returns 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth

def _compute_objective(params):
    """
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
    Returns: minimal outer side length + penalty for overlaps
    """
    positions = params[0:2*N].reshape(N, 2)
    angles = params[2*N:3*N]
    
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i]))
    
    # Normals for outer hexagon (angle 0)
    normals = np.array([
        [np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))]
        for i in range(6)
    ])
    
    max_proj = 0.0
    for verts in verts_list:
        projs = np.dot(verts, normals.T)
        max_proj = max(max_proj, np.max(projs))
    
    S = max_proj * 2.0 / np.sqrt(3.0)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(verts_list[i], verts_list[j])
            if depth > 0:
                penalty += depth * 10.0
    
    return S + penalty

def _get_final_S_and_validity(params):
    positions = params[0:2*N].reshape(N, 2)
    angles = params[2*N:3*N]
    
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i]))
    
    normals = np.array([
        [np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))]
        for i in range(6)
    ])
    
    max_proj = 0.0
    for verts in verts_list:
        projs = np.dot(verts, normals.T)
        max_proj = max(max_proj, np.max(projs))
    
    S = max_proj * 2.0 / np.sqrt(3.0)
    
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(verts_list[i], verts_list[j])
            if depth > 1e-6:
                valid = False
    
    return S, valid

def construct():
    start_time = time.time()
    
    # Initial configuration: 3x4 hexagonal lattice
    positions = []
    for r in range(3):
        for c in range(4):
            x = (c + r / 2.0) * np.sqrt(3)
            y = r * 1.5
            positions.append([x, y])
    
    positions = np.array(positions)
    centroid = positions.mean(axis=0)
    positions = positions - centroid
    angles = np.zeros(N)
    params0 = np.concatenate([positions.flatten(), angles])
    
    best_params = params0
    best_S, best_valid = _get_final_S_and_validity(params0)
    if not best_valid:
        # If initial is invalid (shouldn't be for lattice), try to fix
        pass
    
    # Try optimization
    try:
        result = minimize(
            _compute_objective,
            params0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success:
            opt_params = result.x
            S, valid = _get_final_S_and_validity(opt_params)
            if valid and S < best_S:
                best_S = S
                best_params = opt_params
        else:
            # Even if not successful, check if it's better
            opt_params = result.x
            S, valid = _get_final_S_and_validity(opt_params)
            if valid and S < best_S:
                best_S = S
                best_params = opt_params
    except Exception:
        pass
    
    # If time permits, try a few perturbations
    if time.time() - start_time < 10.0:
        for _ in range(5):
            if time.time() - start_time > 120.0:
                break
            noise = np.random.normal(0, 0.1, size=params0.shape)
            perturbed = params0 + noise
            try:
                res = minimize(
                    _compute_objective,
                    perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success:
                    S, valid = _get_final_S_and_validity(res.x)
                    if valid and S < best_S:
                        best_S = S
                        best_params = res.x
            except Exception:
                pass
    
    positions = best_params[0:2*N].reshape(N, 2)
    angles = best_params[2*N:3*N]
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

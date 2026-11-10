# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize, differential_evolution
from itertools import product


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _face_normals(a_deg):
    a = np.deg2rad(a_deg)
    mid_angles = a + np.deg2rad(30 + 60 * np.arange(6))
    return np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])


def _pair_gap(verts_a, verts_b, norms_a, norms_b):
    proj_a = verts_a @ norms_a.T
    proj_b = verts_b @ norms_a.T
    gap_a = np.maximum(
        proj_b.min(axis=0) - proj_a.max(axis=0),
        proj_a.min(axis=0) - proj_b.max(axis=0)
    )
    proj_a2 = verts_a @ norms_b.T
    proj_b2 = verts_b @ norms_b.T
    gap_b = np.maximum(
        proj_b2.min(axis=0) - proj_a2.max(axis=0),
        proj_a2.min(axis=0) - proj_b2.max(axis=0)
    )
    return max(gap_a.max(), gap_b.max())


def _min_gap_all(verts, norms):
    min_gap = np.inf
    n = len(verts)
    for i in range(n):
        for j in range(i + 1, n):
            g = _pair_gap(verts[i], verts[j], norms[i], norms[j])
            if g < min_gap:
                min_gap = g
    return min_gap


def _enclosing_side_and_angle(all_verts, centroid, n_samples=361):
    shifted = all_verts - centroid
    best_side = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0, 60, n_samples):
        n = _face_normals(a_deg)
        dots = shifted @ n.T
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side:
            best_side = side
            best_a = a_deg
    return best_side, best_a


def _objective(params, deadline):
    if time.time() > deadline:
        return 1e10
    
    verts = np.empty((11, 6, 2))
    norms = np.empty((11, 6, 2))
    
    for i in range(11):
        cx, cy, ad = params[3*i], params[3*i+1], params[3*i+2]
        verts[i] = _hex_vertices(cx, cy, 1.0, ad)
        norms[i] = _face_normals(ad)
    
    min_gap = _min_gap_all(verts, norms)
    if min_gap < 1e-6:
        return 1e6 * (1e-6 - min_gap) + 100.0
    
    all_verts = verts.reshape(-1, 2)
    centroid = all_verts.mean(axis=0)
    shifted = all_verts - centroid
    
    best_side = np.inf
    for a_deg in np.linspace(0, 60, 121):
        n = _face_normals(a_deg)
        dots = shifted @ n.T
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side:
            best_side = side
    
    return best_side


def _local_refine(params, deadline, max_iter=300):
    if time.time() > deadline:
        return params
    
    def obj(x):
        if time.time() > deadline:
            return 1e10
        return _objective(x, deadline)
    
    try:
        res = minimize(
            obj, params,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'fatol': 1e-14, 'xatol': 1e-14, 'adaptive': True}
        )
        if res.fun < _objective(params, deadline):
            return res.x
    except Exception:
        pass
    
    return params


def _gen_config_flex():
    """
    Generate initial config: center + 6 ring + 4 flexible outer.
    The 4 outer hexagons have adjustable angles and radii.
    """
    cfg = np.zeros(33)
    # Center
    cfg[0:3] = [0.0, 0.0, 0.0]
    # 6 ring
    for k in range(6):
        ang = 60.0 * k
        cfg[3*(k+1):3*(k+1)+3] = [2.0*math.cos(math.radians(ang)), 2.0*math.sin(math.radians(ang)), 0.0]
    # 4 flexible: start at 30, 120, 210, 300 degrees, radius ~3.46
    for k, base_ang in enumerate([30, 120, 210, 300]):
        idx = 3*(7+k)
        cfg[idx] = 3.464 * math.cos(math.radians(base_ang))
        cfg[idx+1] = 3.464 * math.sin(math.radians(base_ang))
        cfg[idx+2] = 0.0
    return cfg


def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Start with flexible config
    best_params = _gen_config_flex()
    best_side = _objective(best_params, deadline)
    
    # Local refinement
    if time.time() < deadline - 60:
        refined = _local_refine(best_params, deadline, max_iter=500)
        new_side = _objective(refined, deadline)
        if new_side < best_side:
            best_params = refined
            best_side = new_side
    
    # Try multiple random restarts with small perturbations
    rng = np.random.RandomState(123)
    for restart in range(5):
        if time.time() > deadline - 120:
            break
        
        perturbed = best_params.copy()
        # Perturb the 4 outer hexagons and maybe ring
        for i in [3, 4, 5, 6, 7, 8, 9, 10]:  # ring and outer
            perturbed[3*i] += rng.normal(0, 0.2)
            perturbed[3*i+1] += rng.normal(0, 0.2)
            perturbed[3*i+2] += rng.normal(0, 10)
        
        side = _objective(perturbed, deadline)
        if side < best_side:
            best_params = perturbed
            best_side = side
        
        if time.time() < deadline - 60:
            refined = _local_refine(perturbed, deadline, max_iter=300)
            new_side = _objective(refined, deadline)
            if new_side < best_side:
                best_params = refined
                best_side = new_side
    
    # Final refinement
    if time.time() < deadline - 10:
        refined = _local_refine(best_params, deadline, max_iter=500)
        new_side = _objective(refined, deadline)
        if new_side < best_side:
            best_params = refined
            best_side = new_side
    
    # Build result
    inner_hex_data = np.empty((11, 3))
    for i in range(11):
        inner_hex_data[i] = best_params[3*i:3*i+3]
    
    verts = np.empty((11, 6, 2))
    for i in range(11):
        cx, cy, ad = best_params[3*i], best_params[3*i+1], best_params[3*i+2]
        verts[i] = _hex_vertices(cx, cy, 1.0, ad)
    
    all_verts = verts.reshape(-1, 2)
    centroid = all_verts.mean(axis=0)
    
    best_a = 0.0
    best_side_final = np.inf
    for a_deg in np.linspace(0, 60, 721):
        n = _face_normals(a_deg)
        dots = (all_verts - centroid) @ n.T
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side_final:
            best_side_final = side
            best_a = a_deg
    
    inner_hex_data[:, 0] -= centroid[0]
    inner_hex_data[:, 1] -= centroid[1]
    
    outer_hex_data = np.array([0.0, 0.0, best_a])
    outer_hex_side_length = float(best_side_final)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END

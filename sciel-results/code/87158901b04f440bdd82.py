# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

# Precompute unit hexagon vertices (angle 0)
_UNIT_COS = np.cos(np.deg2rad(60 * np.arange(6)))
_UNIT_SIN = np.sin(np.deg2rad(60 * np.arange(6)))
_UNIT_VERTS = np.column_stack([_UNIT_COS, _UNIT_SIN])  # (6,2)

def _compute_s_and_penalty(inner_positions, outer_center, outer_angle_deg):
    """
    Computes the minimum side length s for the outer hexagon to contain all inner hexagons.
    Also returns a penalty for overlaps.
    inner_positions: (12, 2)
    outer_center: (2,)
    outer_angle_deg: float
    Returns: (s, penalty)
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
    
    # All 72 vertices relative to outer center
    all_verts = inner_positions[:, None, :] + _UNIT_VERTS[None, :, :] 
    all_verts = all_verts.reshape(72, 2) - outer_center[None, :]
    
    # Project onto each normal direction
    projs = all_verts @ normals.T  # (72, 6)
    max_projs = np.max(projs, axis=0)  # (6,)
    max_inradius = np.max(max_projs)
    if max_inradius < 0:
        max_inradius = 0.0
    s = 2.0 * max_inradius / np.sqrt(3.0)
    
    # Check overlaps using center distance approximation
    # For regular hexagons, min center distance is sqrt(3).
    overlap_penalty = 0.0
    dists = np.linalg.norm(inner_positions[:, None, :] - inner_positions[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    overlaps = np.sum(np.maximum(0, np.sqrt(3) - min_dists))
    overlap_penalty = 10.0 * overlaps
    
    return s, overlap_penalty

def _objective(params, outer_angle_deg):
    inner_pos = params[:24].reshape(12, 2)
    oc = params[24:26]
    s, pen = _compute_s_and_penalty(inner_pos, oc, outer_angle_deg)
    return s + pen

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    
    outer_angle_deg = 0.0
    best_s = np.inf
    best_params = None
    
    # Initial guess: Lattice
    pos = []
    pos.append([0,0])
    for i in range(6):
        angle = np.deg2rad(60 * i)
        pos.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
    for i in range(5):
        angle = np.deg2rad(60 * i + 30)
        pos.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
    init_pos = np.array(pos[:12])
    init_params = np.concatenate([init_pos.flatten(), [0,0]])
    
    def de_obj(p):
        if time.time() > deadline - 10:
            return 1e10
        return _objective(p, outer_angle_deg)
    
    # Differential Evolution
    bounds = [(-5, 5)] * 24 + [(-5, 5), (-5, 5)]
    try:
        res = differential_evolution(de_obj, bounds, maxiter=50, tol=1e-5, 
                                      popsize=15, seed=42, polish=False, 
                                      init='sobol', update_interval=1)
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Local Refinement
    if best_params is not None:
        try:
            res2 = minimize(de_obj, best_params, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res2.fun < best_s:
                best_s = res2.fun
                best_params = res2.x
        except Exception:
            pass
        
        # Multi-start
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            perturbed = best_params + rng.normal(0, 0.1, size=best_params.shape)
            try:
                res3 = minimize(de_obj, perturbed, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                if res3.fun < best_s:
                    best_s = res3.fun
                    best_params = res3.x
            except Exception:
                pass

    if best_params is None:
        best_params = init_params
        best_s = _objective(best_params, outer_angle_deg)

    inner_positions = best_params[:24].reshape(12, 2)
    outer_center = best_params[24:26]
    
    s_final, _ = _compute_s_and_penalty(inner_positions, outer_center, outer_angle_deg)
    if s_final < np.sqrt(12):
        s_final = np.sqrt(12)
        
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = inner_positions[i, 0]
        inner_hex_data[i, 1] = inner_positions[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle_deg])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END

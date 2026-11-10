# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Precompute unit hexagon vertices (side=1, angle=0)
_ANGLES = np.deg2rad(60.0 * np.arange(6))
_UNIT_VERTS = np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)])  # (6, 2)

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.deg2rad(angle_deg)
    angles = a + _ANGLES
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _compute_min_side(inner_positions, outer_center, outer_angle_deg):
    """Min side s of outer hex (center, angle) containing all inner unit hexagons."""
    outer_angle = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    rel = inner_positions[:, None, :] + _UNIT_VERTS[None, :, :] - outer_center
    proj = np.einsum('ijk,jk->ij', rel, normals)
    max_inradius = np.max(proj)
    if max_inradius < 0:
        max_inradius = 0.0
    return 2.0 * max_inradius / np.sqrt(3.0)

def _check_separation(hex1_verts, hex2_verts, tol=1e-6):
    axes = []
    for verts in (hex1_verts, hex2_verts):
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    for axis in axes:
        proj1 = hex1_verts @ axis
        proj2 = hex2_verts @ axis
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return True
    return False

def _vertex_inside_outer(v, outer_verts, tol=1e-6):
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -tol:
            return False
    return True

def _validate(inner_hex_data, outer_hex_data, s):
    if s < np.sqrt(12) - 1e-8:
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            if abs(np.linalg.norm(verts[(k + 1) % 6] - verts[k]) - 1.0) > 1e-8:
                return False
        inner_verts_list.append(verts)
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    return True

def _build_result(params, outer_angle_deg):
    inner_positions = params[:24].reshape(12, 2)
    outer_center = params[24:26]
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = inner_positions[:, 0]
    inner_hex_data[:, 1] = inner_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle_deg])
    s = _compute_min_side(inner_positions, outer_center, outer_angle_deg)
    return inner_hex_data, outer_hex_data, s

def _objective(params, outer_angle_deg):
    inner_positions = params[:24].reshape(12, 2)
    outer_center = params[24:26]
    return _compute_min_side(inner_positions, outer_center, outer_angle_deg)

def _objective_with_angle(params):
    """Objective where last param is outer angle in degrees."""
    inner_positions = params[:24].reshape(12, 2)
    outer_center = params[24:26]
    outer_angle_deg = params[26]
    return _compute_min_side(inner_positions, outer_center, outer_angle_deg)

def _optimize_with_de(params0, deadline):
    """Use differential evolution with angle as a parameter."""
    def obj(p):
        if time.time() > deadline:
            return 1e6
        return _objective_with_angle(p)
    
    # Bounds: positions in [-5, 5], center in [-3, 3], angle in [0, 60]
    bounds = [(-5, 5)] * 24 + [(-3, 3)] * 2 + [(0, 60)]
    
    try:
        result = differential_evolution(
            obj, bounds, maxiter=100, popsize=10,
            tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
            seed=42, polish=True, init='latinhypercube',
            maxfev=5000, polish_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11}}
        )
        if result.success or result.fun < 1e5:
            return result.x, result.fun
    except Exception:
        pass
    return params0, _objective_with_angle(params0)

def _optimize_lm(params0, outer_angle_deg, deadline):
    """Nelder-Mead refinement with fixed angle."""
    def obj(p):
        if time.time() > deadline:
            return 1e6
        return _objective(p, outer_angle_deg)
    
    try:
        res = minimize(obj, params0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < 1e5:
            return res.x, res.fun
    except Exception:
        pass
    return params0, _objective(params0, outer_angle_deg)

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # safety margin

    sqrt3 = np.sqrt(3)
    
    # Generate diverse initial configurations
    configs = []
    
    # 1) Honeycomb 4x3
    dx = 2.5
    dy = sqrt3
    pos = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * dx
            y = (row - 1.0) * dy
            if row % 2 == 1:
                x += dx / 2
            pos.append([x, y])
    pos = np.array(pos)
    pos -= pos.mean(axis=0)
    configs.append(pos)
    
    # 2) Ring of 6 + inner 6
    pos2 = []
    r = 2.0
    for i in range(6):
        a = np.deg2rad(60 * i)
        pos2.append([r * np.cos(a), r * np.sin(a)])
    for i in range(6):
        a = np.deg2rad(30 + 60 * i)
        pos2.append([r * 0.5 * np.cos(a), r * 0.5 * np.sin(a)])
    pos2 = np.array(pos2)
    pos2 -= pos2.mean(axis=0)
    configs.append(pos2)
    
    # 3) 3x4 grid
    pos3 = []
    for row in range(4):
        for col in range(3):
            x = (col - 1.0) * dx
            y = (row - 1.5) * dy
            if row % 2 == 1:
                x += dx / 2
            pos3.append([x, y])
    pos3 = np.array(pos3)
    pos3 -= pos3.mean(axis=0)
    configs.append(pos3)
    
    # 4) Random perturbations of honeycomb
    np.random.seed(42)
    for _ in range(3):
        pos4 = configs[0] + np.random.randn(12, 2) * 0.3
        pos4 -= pos4.mean(axis=0)
        configs.append(pos4)
    
    # First pass: use DE to explore angle space broadly
    best_params = None
    best_s = np.inf
    best_angle = 0.0
    
    # Run DE from a few good starting points
    for cfg_idx in [0, 1, 2]:
        if time.time() > deadline - 60:
            break
        cfg = configs[cfg_idx]
        params0 = np.concatenate([cfg.flatten(), [0.0, 0.0], [30.0]])
        p, f = _optimize_with_de(params0, deadline - 60)
        if f < best_s and f < 1e5:
            best_s = f
            best_params = p.copy()
            best_angle = p[26]
    
    # Second pass: refine with Nelder-Mead at multiple angles
    angles_to_try = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0]
    
    for ang in angles_to_try:
        if time.time() > deadline - 30:
            break
        for cfg_idx in [0, 1, 2]:
            if time.time() > deadline - 30:
                break
            cfg = configs[cfg_idx]
            params0 = np.concatenate([cfg.flatten(), [0.0, 0.0]])
            p, f = _optimize_lm(params0, ang, deadline - 30)
            if f < best_s and f < 1e5:
                best_s = f
                best_params = p.copy()
                best_angle = ang
    
    # Final refinement from the best found
    if best_params is not None:
        p, f = _optimize_lm(best_params, best_angle, deadline)
        if f < best_s:
            best_s = f
            best_params = p.copy()
    
    # If DE found a good angle, try refining around it
    if best_params is not None and best_params[26] > 0:
        for da in [-10, -5, 5, 10]:
            ang_test = best_angle + da
            params0 = best_params[:26].copy()
            p, f = _optimize_lm(params0, ang_test, deadline)
            if f < best_s:
                best_s = f
                best_params = p.copy()
                best_angle = ang_test
    
    # Build result
    if best_params is not None:
        inner_hex_data, outer_hex_data, s_final = _build_result(best_params, best_angle)
    else:
        # Fallback
        pos_fb = configs[0].copy()
        inner_hex_data = np.zeros((12, 3))
        inner_hex_data[:, 0] = pos_fb[:, 0]
        inner_hex_data[:, 1] = pos_fb[:, 1]
        inner_hex_data[:, 2] = 0.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        s_final = _compute_min_side(pos_fb, np.array([0.0, 0.0]), 0.0) + 0.05
    
    if not _validate(inner_hex_data, outer_hex_data, s_final):
        # Fallback to a safe honeycomb
        pos_fb = configs[0].copy()
        inner_hex_data = np.zeros((12, 3))
        inner_hex_data[:, 0] = pos_fb[:, 0]
        inner_hex_data[:, 1] = pos_fb[:, 1]
        inner_hex_data[:, 2] = 0.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        s_final = _compute_min_side(pos_fb, np.array([0.0, 0.0]), 0.0) + 0.05
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _min_containing_hex_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon (centered at outer_center,
    with the given angle) that contains all inner unit hexagons.
    """
    ov_cx, ov_cy = outer_center[0], outer_center[1]
    ov_a = np.deg2rad(outer_angle_deg)
    
    # The 6 outward normals to the sides of the outer hexagon
    normal_angles = ov_a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # For each inner hexagon, find the maximum projection onto each normal
    max_projections = np.zeros(6)
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        hv_rel = hv - np.array([ov_cx, ov_cy])
        projections = hv_rel @ normals.T
        max_projections = np.maximum(max_projections, np.max(projections, axis=0))
    
    s_min = np.max(max_projections) / (np.sqrt(3.0) / 2.0)
    return s_min


def _check_separation(inner):
    """
    Check if all pairs of inner hexagons are separated using SAT.
    """
    n = len(inner)
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            
            edges = []
            for k in range(6):
                edges.append(v1[(k + 1) % 6] - v1[k])
                edges.append(v2[(k + 1) % 6] - v2[k])
            
            separated = False
            for edge in edges:
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                gap1 = min2 - max1
                gap2 = min1 - max2
                if gap1 > 1e-6 or gap2 > 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _objective(params, outer_center, outer_angle_deg):
    """
    Objective: minimize containing hexagon side length.
    params: 24 values = 12 * (x, y)
    """
    inner = np.zeros((12, 3))
    inner[:, 0] = params[0::2]
    inner[:, 1] = params[1::2]
    inner[:, 2] = 0.0
    
    if not _check_separation(inner):
        return 1000.0
    
    s = _min_containing_hex_side(inner, outer_center, outer_angle_deg)
    return s


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon.
    Returns (inner, outer, s).
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    # Known good initial configuration: hexagonal lattice
    d = np.sqrt(3.0)
    
    # 12 positions: center + 6 in ring 1 + 5 in ring 2
    initial_params = np.array([
        0.0, 0.0,          # center
        d, 0.0,            # right
        -d, 0.0,           # left
        d/2, d*1.5,        # top-right
        -d/2, d*1.5,       # top-left
        d/2, -d*1.5,       # bottom-right
        -d/2, -d*1.5,      # bottom-left
        2*d, 0.0,          # far right
        -2*d, 0.0,         # far left
        3*d/2, d*1.5,      # far top-right
        -3*d/2, d*1.5,     # far top-left
        0.0, -2*d*1.5,     # far bottom
    ])
    
    inner_init = np.zeros((12, 3))
    inner_init[:, 0] = initial_params[0::2]
    inner_init[:, 1] = initial_params[1::2]
    
    if not _check_separation(inner_init):
        # Fallback: linear arrangement
        spacing = 2.0 * np.sqrt(3.0) + 1e-3
        initial_params = np.array([])
        for i in range(12):
            initial_params = np.append(initial_params, [i * spacing, 0.0])
        
        inner_init = np.zeros((12, 3))
        inner_init[:, 0] = initial_params[0::2]
        inner_init[:, 1] = initial_params[1::2]
    
    best_params = initial_params.copy()
    best_s = _min_containing_hex_side(inner_init, outer_center, outer_angle)
    
    # Primary optimization: Nelder-Mead with high iterations
    try:
        result = minimize(
            _objective,
            initial_params,
            args=(outer_center, outer_angle),
            method='Nelder-Mead',
            options={
                'maxiter': 8000,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        if result.fun < best_s:
            best_params = result.x
            best_s = result.fun
    except Exception:
        pass
    
    # Secondary: Powell method
    if time.time() - start_time < deadline - 120.0:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(outer_center, outer_angle),
                method='Powell',
                options={'maxiter': 3000, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            if result.fun < best_s:
                best_params = result.x
                best_s = result.fun
        except Exception:
            pass
    
    # Tertiary: perturbed restarts
    np.random.seed(42)
    for trial in range(6):
        if time.time() - start_time > deadline - 60.0:
            break
        
        perturbed = initial_params + np.random.normal(0, 0.4, size=initial_params.shape)
        try:
            result = minimize(
                _objective,
                perturbed,
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={
                    'maxiter': 4000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < best_s:
                best_params = result.x
                best_s = result.fun
        except Exception:
            pass
    
    # Construct final result
    inner = np.zeros((12, 3))
    inner[:, 0] = best_params[0::2]
    inner[:, 1] = best_params[1::2]
    inner[:, 2] = 0.0
    
    s_final = _min_containing_hex_side(inner, outer_center, outer_angle) + 1e-4
    
    if not _check_separation(inner):
        # Fallback to initial
        inner = inner_init.copy()
        s_final = _min_containing_hex_side(inner, outer_center, outer_angle) + 1e-4
    
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, s_final


# EVOLVE-BLOCK-END

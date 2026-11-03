import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 140.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    def objective_with_penalty(params):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        angles_rad = np.radians(ang)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    # Estimate penalty based on center distance vs min touching distance
                    # Min touching distance for hexagons is sqrt(3) (side-by-side)
                    # If distance < sqrt(3), they might overlap.
                    # A simple penalty: (sqrt(3) - dist)^2
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if dist < sqrt3:
                        penalty += (sqrt3 - dist) ** 2
        
        if penalty > 1e-6:
            # Strong penalty for overlap
            return 1000.0 + penalty
        
        side, _, _, _ = compute_min_outer_side(pos, ang)
        return side

    # Initializations
    inits = []
    
    # 1. Honeycomb subset
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    pos1 = np.zeros((N, 2))
    ang1 = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        pos1[idx] = i * a1 + j * a2
    inits.append((pos1, ang1))

    # 2. Flower: 1 center, 6 ring, 5 outer
    pos2 = np.zeros((N, 2))
    ang2 = np.zeros(N)
    # Center
    pos2[0] = [0, 0]
    # Ring of 6 at distance sqrt(3)
    for i in range(6):
        angle = i * np.pi / 3
        pos2[i+1] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
    # 5 more: attach to the ring
    # Attach to 0, 1, 2, 3, 4 of the ring (indices 1-5)
    for i in range(5):
        ring_idx = i + 1
        angle = ring_idx * np.pi / 3
        # Move further out
        pos2[7+i] = pos2[ring_idx] + np.array([sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)])
    inits.append((pos2, ang2))

    # 3. Random Valid via Relaxation
    for _ in range(5):
        pos_r = np.random.randn(N, 2) * 3.0
        ang_r = np.random.rand(N) * 360.0
        # Relaxation: push apart if overlapping
        for _iter in range(100):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (pos_r[i, 0], pos_r[i, 1], np.radians(ang_r[i]))
                    h2 = (pos_r[j, 0], pos_r[j, 1], np.radians(ang_r[j]))
                    if not hexagons_disjoint_fast(h1, h2):
                        # Simple push: move apart by 0.1 along center vector
                        vec = pos_r[i] - pos_r[j]
                        norm = np.linalg.norm(vec)
                        if norm > 1e-6:
                            vec = vec / norm
                            pos_r[i] += vec * 0.1
                            pos_r[j] -= vec * 0.1
                            moved = True
            if not moved:
                break
        inits.append((pos_r, ang_r))

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Multi-Start Nelder-Mead
    time_per_start = (DEADLINE - (time.time() - t0)) / len(inits)
    
    for pos_init, ang_init in inits:
        if time.time() - t0 > DEADLINE - 5.0:
            break
            
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = pos_init[i, 0]
            init_params[3*i+1] = pos_init[i, 1]
            init_params[3*i+2] = ang_init[i]
        
        try:
            res = minimize(objective_with_penalty, init_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < best_side:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass

    # Final fallback if no valid found (should not happen)
    if best_positions is None:
        best_positions = inits[0][0]
        best_angles = inits[0][1]
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

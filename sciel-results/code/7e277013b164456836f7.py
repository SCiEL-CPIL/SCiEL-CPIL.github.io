import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 115.0  # Safety margin before 120s wall-clock kill

# Precompute unit hexagon vertices and edge normals
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

    # Precompute flat arrays for speed
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

    def compute_min_outer_side_fast(positions, angles_deg):
        """
        Computes the minimum side length of a regular hexagon enclosing the given hexagons.
        Uses a grid search over orientation followed by local refinement.
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Fast grid search for best orientation
        # The function is periodic with period 60 degrees.
        # We check 60 points (1 degree resolution)
        best_side = float('inf')
        best_phi = 0.0
        
        # Precompute normal vectors for grid points
        # phi is the angle of the normal.
        # Normals are at phi + k*pi/3
        # We can just iterate phi from 0 to 60 in steps of 1
        for phi_deg in range(0, 60, 1):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            # Project all vertices onto these 6 normals
            # proj shape: (num_verts, 6)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0) # Shape (6,)
            side_vals = (2.0 / sqrt3) * max_projs
            current_side = np.max(side_vals)
            
            if current_side < best_side:
                best_side = current_side
                best_phi = phi_deg
        
        # Refine around best_phi
        # Use minimize_scalar for high precision
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
        
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-10})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def compute_overlap_penalty(positions, angles_deg):
        """
        Computes a penalty for overlapping hexagons.
        Returns 0 if no overlaps, otherwise sum of squared penetration depths.
        """
        angles_rad = np.radians(angles_deg)
        penalty = 0.0
        
        # Vectorized vertex calculation for all hexagons
        all_verts = get_all_vertices(positions, angles_rad)
        # all_verts shape: (N, 6, 2)
        
        for i in range(N):
            v1 = all_verts[i]
            for j in range(i+1, N):
                v2 = all_verts[j]
                
                # Quick check: distance between centers
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.0: # Max distance for overlap is 2 (circumradius sum)
                    continue
                
                # SAT Check for penetration depth
                # We need the maximum separation distance for any normal.
                # If min separation > 0, no overlap.
                # Penetration depth = -min_separation.
                
                c1, s1 = math.cos(angles_rad[i]), math.sin(angles_rad[i])
                c2, s2 = math.cos(angles_rad[j]), math.sin(angles_rad[j])
                
                # Normals for hex i
                en1x = _enx * c1 - _eny * s1
                en1y = _enx * s1 + _eny * c1
                # Normals for hex j
                en2x = _enx * c2 - _eny * s2
                en2y = _enx * s2 + _eny * c2
                
                min_sep = float('inf')
                
                # Check normals of i
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1 = v1[:, 0] * nx + v1[:, 1] * ny
                    p2 = v2[:, 0] * nx + v2[:, 1] * ny
                    sep = max(p1.min() - p2.max(), p2.min() - p1.max())
                    if sep > min_sep:
                        min_sep = sep
                    if min_sep > 0:
                        break
                
                # Check normals of j if not separated by i's normals
                if min_sep <= 0:
                    for k in range(6):
                        nx, ny = en2x[k], en2y[k]
                        p1 = v1[:, 0] * nx + v1[:, 1] * ny
                        p2 = v2[:, 0] * nx + v2[:, 1] * ny
                        sep = max(p1.min() - p2.max(), p2.min() - p1.max())
                        if sep > min_sep:
                            min_sep = sep
                        if min_sep > 0:
                            break
                
                if min_sep < 0:
                    # Overlap exists
                    pen = -min_sep
                    penalty += pen * pen
        
        return penalty

    def objective(params):
        """
        Combined objective: Side length + Penalty for overlaps.
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        side, _, _, _ = compute_min_outer_side_fast(pos, ang)
        overlap_pen = compute_overlap_penalty(pos, ang)
        
        # Large weight for overlap to ensure validity
        return side + 1000.0 * overlap_pen

    # --- Initialization Strategies ---
    
    # Strategy 1: 1-6-5 Ring (Standard)
    def init_1_6_5():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # 0: Center
        # 1-6: Ring 1
        for i in range(6):
            a = i * math.pi / 3
            r = math.sqrt(3)
            pos[i+1] = [r * math.cos(a), r * math.sin(a)]
        # 7-11: Ring 2 (5 hexes)
        # Place them in the gaps of ring 1?
        # Gap angles are (i+0.5)*pi/3
        # Distance for ring 2 to touch ring 1 and each other?
        # Let's try r2 = 2*sqrt(3) approx
        for i in range(5):
            a = (i + 0.5) * math.pi / 3
            r = 2 * math.sqrt(3)
            pos[i+7] = [r * math.cos(a), r * math.sin(a)]
        return pos, ang

    # Strategy 2: 1-6-6 Ring (One pushed out)
    def init_1_6_6():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            a = i * math.pi / 3
            r = math.sqrt(3)
            pos[i+1] = [r * math.cos(a), r * math.sin(a)]
        for i in range(6):
            a = (i + 0.5) * math.pi / 3
            r = 2 * math.sqrt(3)
            pos[i+7] = [r * math.cos(a), r * math.sin(a)]
        return pos, ang

    # Strategy 3: Random
    def init_random():
        pos = np.random.randn(N, 2) * 2.0
        ang = np.random.rand(N) * 360
        return pos, ang

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    initializers = [init_1_6_5, init_1_6_6]
    # Add a few random starts
    for _ in range(3):
        initializers.append(init_random)

    for init_func in initializers:
        if time.time() - t0 > DEADLINE - 10:
            break
            
        pos, ang = init_func()
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = pos[i, 0]
            init_params[3*i+1] = pos[i, 1]
            init_params[3*i+2] = ang[i]
            
        try:
            # Nelder-Mead
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            # Check if the result is valid (no overlaps)
            final_pos = np.zeros((N, 2))
            final_ang = np.zeros(N)
            for i in range(N):
                final_pos[i, 0] = res.x[3*i]
                final_pos[i, 1] = res.x[3*i+1]
                final_ang[i] = res.x[3*i+2]
                
            overlap = compute_overlap_penalty(final_pos, final_ang)
            if overlap > 1e-6:
                continue # Invalid result from optimizer
            
            side, phi, cx, cy = compute_min_outer_side_fast(final_pos, final_ang)
            
            if side < best_side:
                best_side = side
                best_positions = final_pos
                best_angles = final_ang
                best_phi = phi
                best_cx = cx
                best_cy = cy
                
        except Exception as e:
            continue

    # Fallback if no valid solution found (should not happen with 1-6-5)
    if best_positions is None:
        pos, ang = init_1_6_5()
        best_positions = pos
        best_angles = ang
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side_fast(pos, ang)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

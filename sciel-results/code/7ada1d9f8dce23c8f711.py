import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 115.0  # Safety margin

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
    
    sqrt3 = math.sqrt(3)
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _UNIT_HEX_VERTS[0][None, :] * c[:, None] - _UNIT_HEX_VERTS[1][None, :] * s[:, None]
        # Wait, _UNIT_HEX_VERTS is (6,2).
        # Vectorized rotation:
        # v_rot = R * v
        # R = [[c, -s], [s, c]]
        # v_rot_x = c*vx - s*vy
        # v_rot_y = s*vx + c*vy
        
        vx = _UNIT_HEX_VERTS[:, 0]
        vy = _UNIT_HEX_VERTS[:, 1]
        
        # c, s are (N,)
        # vx, vy are (6,)
        # We need (N, 6)
        rx = c[:, None] * vx[None, :] - s[:, None] * vy[None, :]
        ry = s[:, None] * vx[None, :] + c[:, None] * vy[None, :]
        
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        
        return np.stack((verts_x, verts_y), axis=-1) # (N, 6, 2)

    def compute_widths(positions, angles_rad):
        # Returns (6,) array of widths
        verts = get_all_vertices(positions, angles_rad) # (N, 6, 2)
        verts_flat = verts.reshape(-1, 2)
        
        # Normals for outer hexagon (6 directions)
        # Outer hexagon normals are at 30, 90, 150, ... degrees?
        # A regular hexagon with side S has apothem S*sqrt(3)/2.
        # The width in direction theta is 2 * max projection.
        # The normals of a regular hexagon are spaced by 60 degrees.
        # Let's use angles 0, 60, 120, 180, 240, 300 for normals.
        # This corresponds to a hexagon with flat top/bottom?
        # Vertices at 0, 60... -> Normals at 30, 90...
        # Let's stick to the definition:
        # If outer hexagon has angle phi, its normals are phi + 30 + k*60.
        # We can optimize phi later or include it.
        # For the objective, we can just take the max over 360 degrees? No, that's expensive.
        # We know the optimal outer hexagon will have its normals aligned with some symmetry of the packing.
        # Let's assume the outer hexagon can rotate freely.
        # So we minimize max_{phi} ( max_k W_k(phi) ).
        # This is hard.
        # Instead, let's fix the outer hexagon orientation to 0 and allow the inner ones to rotate.
        # Due to symmetry, the optimal packing can be rotated to align with the outer hexagon.
        # So we can fix outer normals to 0, 60, 120, 180, 240, 300.
        
        normal_angles = np.radians(np.arange(0, 360, 60))
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        # Project all vertices onto these normals
        # verts_flat is (72, 2)
        # proj is (72, 6)
        proj = verts_flat[:, 0:1] * nx[None, :] + verts_flat[:, 1:2] * ny[None, :]
        
        widths = np.max(proj, axis=0) - np.min(proj, axis=0)
        return widths

    def get_overlap_penalty(positions, angles_rad):
        # Sum of squared penetration depths
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Check overlap between i and j
                c1, s1 = math.cos(angles_rad[i]), math.sin(angles_rad[i])
                c2, s2 = math.cos(angles_rad[j]), math.sin(angles_rad[j])
                
                v1x = positions[i, 0] + _UNIT_HEX_VERTS[:, 0] * c1 - _UNIT_HEX_VERTS[:, 1] * s1
                v1y = positions[i, 1] + _UNIT_HEX_VERTS[:, 0] * s1 + _UNIT_HEX_VERTS[:, 1] * c1
                v2x = positions[j, 0] + _UNIT_HEX_VERTS[:, 0] * c2 - _UNIT_HEX_VERTS[:, 1] * s2
                v2y = positions[j, 1] + _UNIT_HEX_VERTS[:, 0] * s2 + _UNIT_HEX_VERTS[:, 1] * c2
                
                en1x = _EDGE_NORMALS[:, 0] * c1 - _EDGE_NORMALS[:, 1] * s1
                en1y = _EDGE_NORMALS[:, 0] * s1 + _EDGE_NORMALS[:, 1] * c1
                en2x = _EDGE_NORMALS[:, 0] * c2 - _EDGE_NORMALS[:, 1] * s2
                en2y = _EDGE_NORMALS[:, 0] * s2 + _EDGE_NORMALS[:, 1] * c2
                
                all_nx = np.concatenate([en1x, en2x])
                all_ny = np.concatenate([en1y, en2y])
                
                overlap = True
                min_overlap_width = float('inf')
                
                for k in range(12):
                    nx, ny = all_nx[k], all_ny[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    min1, max1 = np.min(p1), np.max(p1)
                    min2, max2 = np.min(p2), np.max(p2)
                    
                    if max1 < min2 or max2 < min1:
                        overlap = False
                        break
                    else:
                        # Overlap width in this direction
                        # Intersection of [min1, max1] and [min2, max2]
                        # Width = min(max1, max2) - max(min1, min2)
                        ow = min(max1, max2) - max(min1, min2)
                        if ow < min_overlap_width:
                            min_overlap_width = ow
                
                if overlap:
                    penalty += min_overlap_width * min_overlap_width
        
        return penalty

    def objective(params):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        ang_rad = np.radians(ang)
        
        widths = compute_widths(pos, ang_rad)
        # Smooth max of widths
        beta = 20.0
        log_sum_exp = np.log(np.sum(np.exp(beta * widths))) / beta
        # Approximation of max(widths)
        
        overlap_pen = get_overlap_penalty(pos, ang_rad)
        
        return log_sum_exp + 10.0 * overlap_pen

    # Initial configurations
    configs = []
    
    # 1. Honeycomb
    a1 = np.array([math.sqrt(3), 0.0])
    a2 = np.array([math.sqrt(3)/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0
    configs.append((base_positions, base_angles))
    
    # 2. Grid 3-4-3-2
    grid_pos = np.zeros((N, 2))
    grid_ang = np.zeros(N)
    rows = [3, 4, 3, 2]
    y = 0
    idx = 0
    for r_count in rows:
        for c in range(r_count):
            # Centered rows
            x_offset = (c - (r_count - 1) / 2.0) * math.sqrt(3)
            grid_pos[idx, 0] = x_offset
            grid_pos[idx, 1] = y
            grid_ang[idx] = 0.0
            idx += 1
        y -= 1.5
    configs.append((grid_pos, grid_ang))
    
    # 3. Random with repulsion
    for _ in range(5):
        rp = np.random.uniform(-5, 5, (N, 2))
        ra = np.random.uniform(0, 60, N)
        # Simple repulsion pass
        for _ in range(50):
            for i in range(N):
                for j in range(i+1, N):
                    dx = rp[i, 0] - rp[j, 0]
                    dy = rp[i, 1] - rp[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 3.0 and dist > 0:
                        force = (3.0 - dist) / dist
                        rp[i, 0] += dx * force * 0.5
                        rp[i, 1] += dy * force * 0.5
                        rp[j, 0] -= dx * force * 0.5
                        rp[j, 1] -= dy * force * 0.5
        configs.append((rp, ra))

    best_side = float('inf')
    best_params = None
    
    for pos_init, ang_init in configs:
        if time.time() - t0 > DEADLINE - 10:
            break
            
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = pos_init[i, 0]
            init_params[3*i+1] = pos_init[i, 1]
            init_params[3*i+2] = ang_init[i]
            
        try:
            # Use Powell for robustness, or L-BFGS-B if gradients are good
            # L-BFGS-B is faster but might get stuck in local minima of the smooth approx
            # Powell is derivative-free and robust
            res = minimize(objective, init_params, method='Powell', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < 100: # Sanity check
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                # Verify validity
                valid = True
                ang_rad = np.radians(new_ang)
                for i in range(N):
                    for j in range(i+1, N):
                        c1, s1 = math.cos(ang_rad[i]), math.sin(ang_rad[i])
                        c2, s2 = math.cos(ang_rad[j]), math.sin(ang_rad[j])
                        v1x = new_pos[i, 0] + _UNIT_HEX_VERTS[:, 0] * c1 - _UNIT_HEX_VERTS[:, 1] * s1
                        v1y = new_pos[i, 1] + _UNIT_HEX_VERTS[:, 0] * s1 + _UNIT_HEX_VERTS[:, 1] * c1
                        v2x = new_pos[j, 0] + _UNIT_HEX_VERTS[:, 0] * c2 - _UNIT_HEX_VERTS[:, 1] * s2
                        v2y = new_pos[j, 1] + _UNIT_HEX_VERTS[:, 0] * s2 + _UNIT_HEX_VERTS[:, 1] * c2
                        en1x = _EDGE_NORMALS[:, 0] * c1 - _EDGE_NORMALS[:, 1] * s1
                        en1y = _EDGE_NORMALS[:, 0] * s1 + _EDGE_NORMALS[:, 1] * c1
                        en2x = _EDGE_NORMALS[:, 0] * c2 - _EDGE_NORMALS[:, 1] * s2
                        en2y = _EDGE_NORMALS[:, 0] * s2 + _EDGE_NORMALS[:, 1] * c2
                        all_nx = np.concatenate([en1x, en2x])
                        all_ny = np.concatenate([en1y, en2y])
                        
                        for k in range(12):
                            nx, ny = all_nx[k], all_ny[k]
                            p1 = v1x * nx + v1y * ny
                            p2 = v2x * nx + v2y * ny
                            if p1.max() < p2.min() or p2.max() < p1.min():
                                break
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Compute exact side
                    ang_rad = np.radians(new_ang)
                    widths = compute_widths(new_pos, ang_rad)
                    # The max width gives the side length for a hexagon with normals aligned to axes
                    # But we need to check if rotating the outer hexagon helps.
                    # The function compute_widths assumes outer normals at 0, 60...
                    # The true minimal enclosing hexagon side is min_phi max_k W_k(phi).
                    # We can scan phi.
                    best_s = float('inf')
                    for phi_deg in range(0, 60, 1):
                        # Rotate vertices? No, just change normals.
                        # Widths in direction phi + k*60
                        normal_angles = np.radians(phi_deg + np.arange(0, 360, 60))
                        nx = np.cos(normal_angles)
                        ny = np.sin(normal_angles)
                        verts = get_all_vertices(new_pos, ang_rad).reshape(-1, 2)
                        proj = verts[:, 0:1] * nx[None, :] + verts[:, 1:2] * ny[None, :]
                        w = np.max(proj, axis=0) - np.min(proj, axis=0)
                        s = (2.0 / sqrt3) * np.max(w)
                        if s < best_s:
                            best_s = s
                    
                    if best_s < best_side:
                        best_side = best_s
                        best_params = res.x.copy()
                        # Store best phi
                        # Re-find best phi
                        best_phi = 0
                        for phi_deg in range(0, 60, 1):
                            normal_angles = np.radians(phi_deg + np.arange(0, 360, 60))
                            nx = np.cos(normal_angles)
                            ny = np.sin(normal_angles)
                            verts = get_all_vertices(new_pos, ang_rad).reshape(-1, 2)
                            proj = verts[:, 0:1] * nx[None, :] + verts[:, 1:2] * ny[None, :]
                            w = np.max(proj, axis=0) - np.min(proj, axis=0)
                            s = (2.0 / sqrt3) * np.max(w)
                            if s < best_s:
                                best_s = s
                                best_phi = phi_deg
        except Exception:
            pass

    if best_params is None:
        # Fallback to initial honeycomb
        best_params = np.zeros(3*N)
        for i in range(N):
            best_params[3*i] = configs[0][0][i, 0]
            best_params[3*i+1] = configs[0][0][i, 1]
            best_params[3*i+2] = configs[0][1][i]
        best_side = 10.0 # Rough estimate
        best_phi = 0

    # Final Polish: Coordinate Descent on exact objective
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(N):
        pos[i, 0] = best_params[3*i]
        pos[i, 1] = best_params[3*i+1]
        ang[i] = best_params[3*i+2]
    
    def exact_side(pos, ang):
        ang_rad = np.radians(ang)
        verts = get_all_vertices(pos, ang_rad).reshape(-1, 2)
        best_s = float('inf')
        for phi_deg in np.arange(0, 60, 0.5):
            normal_angles = np.radians(phi_deg + np.arange(0, 360, 60))
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = verts[:, 0:1] * nx[None, :] + verts[:, 1:2] * ny[None, :]
            w = np.max(proj, axis=0) - np.min(proj, axis=0)
            s = (2.0 / sqrt3) * np.max(w)
            if s < best_s:
                best_s = s
        return best_s

    current_side = exact_side(pos, ang)
    step = 0.05
    for _ in range(50):
        if time.time() - t0 > DEADLINE:
            break
        improved = False
        for i in range(N):
            for dim in range(3):
                for d in [1, -1]:
                    new_pos = pos.copy()
                    new_ang = ang.copy()
                    if dim == 0: new_pos[i, 0] += d * step
                    elif dim == 1: new_pos[i, 1] += d * step
                    else: new_ang[i] += d * step
                    
                    # Quick validity check
                    valid = True
                    ang_rad = np.radians(new_ang)
                    for j in range(N):
                        if i == j: continue
                        c1, s1 = math.cos(ang_rad[i]), math.sin(ang_rad[i])
                        c2, s2 = math.cos(ang_rad[j]), math.sin(ang_rad[j])
                        v1x = new_pos[i, 0] + _UNIT_HEX_VERTS[:, 0] * c1 - _UNIT_HEX_VERTS[:, 1] * s1
                        v1y = new_pos[i, 1] + _UNIT_HEX_VERTS[:, 0] * s1 + _UNIT_HEX_VERTS[:, 1] * c1
                        v2x = new_pos[j, 0] + _UNIT_HEX_VERTS[:, 0] * c2 - _UNIT_HEX_VERTS[:, 1] * s2
                        v2y = new_pos[j, 1] + _UNIT_HEX_VERTS[:, 0] * s2 + _UNIT_HEX_VERTS[:, 1] * c2
                        en1x = _EDGE_NORMALS[:, 0] * c1 - _EDGE_NORMALS[:, 1] * s1
                        en1y = _EDGE_NORMALS[:, 0] * s1 + _EDGE_NORMALS[:, 1] * c1
                        en2x = _EDGE_NORMALS[:, 0] * c2 - _EDGE_NORMALS[:, 1] * s2
                        en2y = _EDGE_NORMALS[:, 0] * s2 + _EDGE_NORMALS[:, 1] * c2
                        all_nx = np.concatenate([en1x, en2x])
                        all_ny = np.concatenate([en1y, en2y])
                        for k in range(12):
                            nx, ny = all_nx[k], all_ny[k]
                            p1 = v1x * nx + v1y * ny
                            p2 = v2x * nx + v2y * ny
                            if p1.max() < p2.min() or p2.max() < p1.min():
                                break
                        else:
                            valid = False
                            break
                    if not valid: continue
                    
                    new_side = exact_side(new_pos, new_ang)
                    if new_side < current_side:
                        pos = new_pos
                        ang = new_ang
                        current_side = new_side
                        improved = True
                        break
                if improved: break
            if improved: break
        if not improved:
            step *= 0.8

    inner = []
    for i in range(N):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(ang[i])])
    
    # Compute center and angle for output
    ang_rad = np.radians(ang)
    verts = get_all_vertices(pos, ang_rad).reshape(-1, 2)
    center = np.mean(verts, axis=0)
    
    # Find optimal phi
    best_s = float('inf')
    best_phi = 0
    for phi_deg in np.arange(0, 60, 0.1):
        normal_angles = np.radians(phi_deg + np.arange(0, 360, 60))
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = verts[:, 0:1] * nx[None, :] + verts[:, 1:2] * ny[None, :]
        w = np.max(proj, axis=0) - np.min(proj, axis=0)
        s = (2.0 / sqrt3) * np.max(w)
        if s < best_s:
            best_s = s
            best_phi = phi_deg

    return inner, (float(center[0]), float(center[1])), float(best_s), float(best_phi)

import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar

N = 12
DEADLINE = 135.0  # Safety margin before 150s wall-clock kill

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

    def compute_side_length(positions, angles_deg):
        """Calculate the minimum side length of the enclosing regular hexagon."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        best_side = float('inf')
        best_phi = 0.0
        
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            s = np.max(side_vals)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune
        try:
            lo = max(0.0, best_phi - 2.0)
            hi = min(60.0, best_phi + 2.0)
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
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-7})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def calculate_penalty(positions, angles_rad, lambda_pen=100.0):
        """
        Calculate a soft penalty for overlaps.
        Returns (penalty_value, is_valid).
        Penalty is sum of squared penetration depths for overlapping pairs.
        """
        penalty = 0.0
        is_valid = True
        
        # Bounding box pre-filter for speed
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.5: # Max distance for potential overlap is ~2.0 (circumradius sum)
                    continue
                
                # Detailed SAT check
                a1 = angles_rad[i]
                a2 = angles_rad[j]
                
                c1, s1 = math.cos(a1), math.sin(a1)
                c2, s2 = math.cos(a2), math.sin(a2)
                
                # Vertices
                v1x = positions[i, 0] + _ux * c1 - _uy * s1
                v1y = positions[i, 1] + _ux * s1 + _uy * c1
                v2x = positions[j, 0] + _ux * c2 - _uy * s2
                v2y = positions[j, 1] + _ux * s2 + _uy * c2
                
                # Normals
                en1x = _enx * c1 - _eny * s1
                en1y = _enx * s1 + _eny * c1
                en2x = _enx * c2 - _eny * s2
                en2y = _enx * s2 + _eny * c2
                
                min_overlap = float('inf')
                overlap_found = False
                
                # Check axes from hexagon 1
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1_min = np.min(v1x * nx + v1y * ny)
                    p1_max = np.max(v1x * nx + v1y * ny)
                    p2_min = np.min(v2x * nx + v2y * ny)
                    p2_max = np.max(v2x * nx + v2y * ny)
                    
                    if p1_max < p2_min or p2_max < p1_min:
                        overlap_found = False
                        break
                    else:
                        overlap_found = True
                        overlap = min(p1_max, p2_max) - max(p1_min, p2_min)
                        if overlap < min_overlap:
                            min_overlap = overlap
                
                if overlap_found:
                    # Check axes from hexagon 2
                    for k in range(6):
                        nx, ny = en2x[k], en2y[k]
                        p1_min = np.min(v1x * nx + v1y * ny)
                        p1_max = np.max(v1x * nx + v1y * ny)
                        p2_min = np.min(v2x * nx + v2y * ny)
                        p2_max = np.max(v2x * nx + v2y * ny)
                        
                        if p1_max < p2_min or p2_max < p1_min:
                            overlap_found = False
                            break
                        else:
                            overlap = min(p1_max, p2_max) - max(p1_min, p2_min)
                            if overlap < min_overlap:
                                min_overlap = overlap
                    
                    if overlap_found:
                        is_valid = False
                        penalty += min_overlap * min_overlap
        
        return penalty, is_valid

    def objective_function(params):
        """
        Objective for optimization.
        params: [x0, y0, a0, x1, y1, a1, ...]
        """
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles_deg[i] = params[3*i+2]
            
        angles_rad = np.radians(angles_deg)
        
        # Check validity and get penalty
        penalty, is_valid = calculate_penalty(positions, angles_rad)
        
        if not is_valid:
            # Return a large value if invalid, but scaled by penalty to guide optimization
            # We want to minimize side length, but heavily penalize overlap
            side, _, _, _ = compute_side_length(positions, angles_deg)
            return side + 1000.0 * penalty
            
        side, _, _, _ = compute_side_length(positions, angles_deg)
        return side

    # --- Strategy: Differential Evolution + Local Polish ---
    
    # Bounds for DE
    # Estimate a reasonable box size. Initial config side was ~10.
    # Let's use a box of [-10, 10] for x,y and [0, 360] for angle.
    bounds = [(-10, 10), (-10, 10), (0, 360)] * N
    
    # Time budget for DE
    de_time_budget = DEADLINE * 0.5
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Initialize with a random spread to help DE
    initial_positions = np.random.uniform(-5, 5, (N, 2))
    initial_angles = np.random.uniform(0, 360, N)
    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[3*i] = initial_positions[i, 0]
        init_params[3*i+1] = initial_positions[i, 1]
        init_params[3*i+2] = initial_angles[i]
        
    # Run Differential Evolution
    try:
        def de_callback(xk, convergence):
            nonlocal best_side, best_positions, best_angles, best_phi, best_cx, best_cy
            # Evaluate current best candidate
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = xk[3*i]
                pos[i, 1] = xk[3*i+1]
                ang[i] = xk[3*i+2]
                
            angles_rad = np.radians(ang)
            penalty, is_valid = calculate_penalty(pos, angles_rad)
            
            if is_valid:
                side, phi, cx, cy = compute_side_length(pos, ang)
                if side < best_side:
                    best_side = side
                    best_positions = pos.copy()
                    best_angles = ang.copy()
                    best_phi = phi
                    best_cx = cx
                    best_cy = cy
                    
        res_de = differential_evolution(objective_function, bounds, 
                                        maxiter=50, 
                                        popsize=20, 
                                        tol=1e-6, 
                                        seed=42,
                                        callback=de_callback,
                                        polish=False, # We will polish separately
                                        disp=False)
        
        # Update best from final result if better
        if res_de.success:
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = res_de.x[3*i]
                pos[i, 1] = res_de.x[3*i+1]
                ang[i] = res_de.x[3*i+2]
                
            angles_rad = np.radians(ang)
            penalty, is_valid = calculate_penalty(pos, angles_rad)
            
            if is_valid:
                side, phi, cx, cy = compute_side_length(pos, ang)
                if side < best_side:
                    best_side = side
                    best_positions = pos.copy()
                    best_angles = ang.copy()
                    best_phi = phi
                    best_cx = cx
                    best_cy = cy
                    
    except Exception as e:
        # Fallback if DE fails or takes too long
        print(f"DE failed: {e}")
        
    # If DE didn't find a valid solution, use a heuristic fallback
    if best_positions is None:
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        config1_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_positions = np.zeros((N, 2))
        best_angles = np.zeros(N)
        for idx, (i, j) in enumerate(config1_coords):
            best_positions[idx] = i * a1 + j * a2
            best_angles[idx] = 0.0
        best_side, best_phi, best_cx, best_cy = compute_side_length(best_positions, best_angles)

    # --- Polish Phase ---
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 5.0 and best_positions is not None:
        # Use L-BFGS-B for local optimization
        # We need to ensure the start point is valid
        start_params = np.zeros(3*N)
        for i in range(N):
            start_params[3*i] = best_positions[i, 0]
            start_params[3*i+1] = best_positions[i, 1]
            start_params[3*i+2] = best_angles[i]
            
        try:
            # L-BFGS-B is good for large problems
            res_lm = minimize(objective_function, start_params, method='L-BFGS-B', 
                              bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res_lm.fun < best_side:
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = res_lm.x[3*i]
                    pos[i, 1] = res_lm.x[3*i+1]
                    ang[i] = res_lm.x[3*i+2]
                
                angles_rad = np.radians(ang)
                penalty, is_valid = calculate_penalty(pos, angles_rad)
                
                if is_valid:
                    side, phi, cx, cy = compute_side_length(pos, ang)
                    if side < best_side:
                        best_side = side
                        best_positions = pos.copy()
                        best_angles = ang.copy()
                        best_phi = phi
                        best_cx = cx
                        best_cy = cy
                        
        except Exception as e:
            print(f"L-BFGS-B failed: {e}")

    # Final verification
    if not best_positions is None:
        angles_rad = np.radians(best_angles)
        penalty, is_valid = calculate_penalty(best_positions, angles_rad)
        if not is_valid:
            # Fallback to a valid configuration
            a1_fb = np.array([sqrt3, 0.0])
            a2_fb = np.array([sqrt3/2.0, 1.5])
            fb_coords = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, -1), (-1, 1), (2, -1), (-2, 1),
                (1, 1), (-1, -1), (2, 0),
            ]
            best_positions = np.zeros((N, 2))
            best_angles = np.zeros(N)
            for idx, (i, j) in enumerate(fb_coords):
                best_positions[idx] = i * a1_fb + j * a2_fb
                best_angles[idx] = 0.0
            best_side, best_phi, best_cx, best_cy = compute_side_length(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

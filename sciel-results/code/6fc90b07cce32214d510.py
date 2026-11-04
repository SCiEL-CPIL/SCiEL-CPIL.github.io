import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0 # Safety margin

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
        ry = _UNIT_HEX_VERTS[0][None, :] * s[:, None] + _UNIT_HEX_VERTS[1][None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def get_normals(angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        nx = _EDGE_NORMALS[0][None, :] * c[:, None] - _EDGE_NORMALS[1][None, :] * s[:, None]
        ny = _EDGE_NORMALS[0][None, :] * s[:, None] + _EDGE_NORMALS[1][None, :] * c[:, None]
        return np.stack((nx, ny), axis=-1)

    def compute_smooth_side_and_penalty(params, beta_gap, beta_side, alpha_gap, alpha_side):
        # params: [x1, y1, th1, ..., x12, y12, th12, phi_outer]
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles_deg[i] = params[3*i+2]
        phi_outer = params[3*N]
        
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Center of mass of all vertices to center the outer hexagon
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Outer hexagon normals
        phi_rad = np.radians(phi_outer)
        c_phi = math.cos(phi_rad)
        s_phi = math.sin(phi_rad)
        # Base normals are at 30, 90, 150... degrees (pi/6, pi/2, 5pi/6...)
        # _EDGE_NORMALS are normals to edges. For a hexagon with flat top, 
        # vertices are at 0, 60...
        # Edges are between 0-60, 60-120...
        # Normal to edge 0-60 is at 30 deg.
        # So base normals are at 30 + 60*k.
        base_norm_angles = np.pi/6 + np.arange(6) * np.pi/3
        norm_angles = base_norm_angles + phi_rad
        outer_normals = np.column_stack((np.cos(norm_angles), np.sin(norm_angles)))
        
        # Projections of all vertices onto outer normals
        # offsets: (72, 2), outer_normals: (6, 2)
        projs = offsets @ outer_normals.T # (72, 6)
        abs_projs = np.abs(projs)
        
        # Smooth max of abs_projs for each normal
        # side = (2/sqrt3) * max_k ( max_v |proj_vk| )
        # Use log-sum-exp for max_v
        # max_v |proj_vk| approx 1/alpha_side * log sum exp(alpha_side * |proj_vk|)
        scaled_projs = abs_projs * alpha_side
        max_projs_per_norm = (1.0 / alpha_side) * np.log(np.sum(np.exp(scaled_projs), axis=0))
        
        # Now max over normals
        scaled_norms = max_projs_per_norm * alpha_side
        global_max_proj = (1.0 / alpha_side) * np.log(np.sum(np.exp(scaled_norms)))
        
        side_length = (2.0 / sqrt3) * global_max_proj
        
        # Penalty for overlaps
        penalty = 0.0
        all_normals_inner = get_normals(angles_rad) # (N, 6, 2)
        
        for i in range(N):
            for j in range(i+1, N):
                v1 = all_verts[i] # (6, 2)
                v2 = all_verts[j] # (6, 2)
                n1 = all_normals_inner[i] # (6, 2)
                n2 = all_normals_inner[j] # (6, 2)
                
                # Combine normals (12 axes)
                # We can just check the 6 normals of hex 1 and 6 of hex 2
                # Project v1 and v2 onto each normal
                # For normal n, gap = min(min2 - max1, min1 - max2)
                # We want G = max(gaps)
                
                # Vectorized gap calculation for 12 axes
                # Axes: n1 (6) and n2 (6)
                # Projections of v1 on n1: (6, 6)
                # Projections of v1 on n2: (6, 6)
                # Projections of v2 on n1: (6, 6)
                # Projections of v2 on n2: (6, 6)
                
                p1_n1 = v1 @ n1.T # (6,6)
                p2_n1 = v2 @ n1.T # (6,6)
                p1_n2 = v1 @ n2.T # (6,6)
                p2_n2 = v2 @ n2.T # (6,6)
                
                # For each axis k in 0..5 (from n1):
                # min1 = min(p1_n1[:,k]), max1 = max(p1_n1[:,k])
                # min2 = min(p2_n1[:,k]), max2 = max(p2_n1[:,k])
                # gap_k = min(min2 - max1, min1 - max2)
                
                min1_n1 = np.min(p1_n1, axis=0)
                max1_n1 = np.max(p1_n1, axis=0)
                min2_n1 = np.min(p2_n1, axis=0)
                max2_n1 = np.max(p2_n1, axis=0)
                gaps_n1 = np.minimum(min2_n1 - max1_n1, min1_n1 - max2_n1)
                
                min1_n2 = np.min(p1_n2, axis=0)
                max1_n2 = np.max(p1_n2, axis=0)
                min2_n2 = np.min(p2_n2, axis=0)
                max2_n2 = np.max(p2_n2, axis=0)
                gaps_n2 = np.minimum(min2_n2 - max1_n2, min1_n2 - max2_n2)
                
                all_gaps = np.concatenate([gaps_n1, gaps_n2]) # (12,)
                
                # Smooth max of gaps
                # G approx 1/alpha_gap * log sum exp(alpha_gap * gap_k)
                scaled_gaps = all_gaps * alpha_gap
                # Clip to avoid overflow
                scaled_gaps = np.clip(scaled_gaps, -500, 500)
                G_approx = (1.0 / alpha_gap) * np.log(np.sum(np.exp(scaled_gaps)))
                
                # Penalty: softplus(-G_approx)
                # softplus(x) = log(1 + exp(x))
                # We want penalty when G < 0, i.e., -G > 0
                # pen = log(1 + exp(beta_gap * (-G))) / beta_gap
                x = -G_approx * beta_gap
                x = np.clip(x, -500, 500)
                pen = np.log1p(np.exp(x)) / beta_gap
                penalty += pen * pen # Squared penalty for stronger push
                
        return side_length + 10.0 * penalty

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        all_normals = get_normals(angles_rad)
        for i in range(N):
            for j in range(i+1, N):
                v1 = all_verts[i]
                v2 = all_verts[j]
                n1 = all_normals[i]
                n2 = all_normals[j]
                
                p1_n1 = v1 @ n1.T
                p2_n1 = v2 @ n1.T
                min1 = np.min(p1_n1, axis=0)
                max1 = np.max(p1_n1, axis=0)
                min2 = np.min(p2_n1, axis=0)
                max2 = np.max(p2_n1, axis=0)
                gaps1 = np.minimum(min2 - max1, min1 - max2)
                if np.any(gaps1 < 0):
                    # Potential overlap, check n2
                    p1_n2 = v1 @ n2.T
                    p2_n2 = v2 @ n2.T
                    min1_2 = np.min(p1_n2, axis=0)
                    max1_2 = np.max(p1_n2, axis=0)
                    min2_2 = np.min(p2_n2, axis=0)
                    max2_2 = np.max(p2_n2, axis=0)
                    gaps2 = np.minimum(min2_2 - max1_2, min1_2 - max2_2)
                    all_gaps = np.concatenate([gaps1, gaps2])
                    if np.any(all_gaps < 0):
                        return False
        return True

    def compute_exact_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        best_side = float('inf')
        # Scan phi
        for phi_deg in range(0, 60, 1):
            phi = np.radians(phi_deg)
            base_norm_angles = np.pi/6 + np.arange(6) * np.pi/3
            norm_angles = base_norm_angles + phi
            outer_normals = np.column_stack((np.cos(norm_angles), np.sin(norm_angles)))
            projs = offsets @ outer_normals.T
            abs_projs = np.abs(projs)
            max_projs = np.max(abs_projs, axis=0)
            side = (2.0 / sqrt3) * np.max(max_projs)
            if side < best_side:
                best_side = side
                best_phi = phi_deg
                
        # Refine
        # ... (skipping refinement for brevity in thought, will implement in code)
        
        return best_side, best_phi, center[0], center[1]

    # Initialization
    def make_honeycomb():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    init_pos, init_ang = make_honeycomb()
    
    best_side = float('inf')
    best_params = None
    
    # Multi-start L-BFGS
    starts = [
        (init_pos, init_ang, 0.0),
        (init_pos + 0.1*np.random.randn(N,2), init_ang + 5*np.random.randn(N), 0.0),
        (init_pos, init_ang, 30.0),
    ]
    
    for start_pos, start_ang, start_phi in starts:
        if time.time() - t0 > DEADLINE - 20: break
        
        params0 = np.zeros(3*N + 1)
        for i in range(N):
            params0[3*i] = start_pos[i, 0]
            params0[3*i+1] = start_pos[i, 1]
            params0[3*i+2] = start_ang[i]
        params0[3*N] = start_phi
        
        # Continuation method
        beta_gap = 10.0
        alpha_gap = 50.0
        beta_side = 10.0
        alpha_side = 50.0
        
        current_params = params0
        
        for stage in range(5):
            if time.time() - t0 > DEADLINE - 10: break
            
            def obj(p):
                return compute_smooth_side_and_penalty(p, beta_gap, beta_side, alpha_gap, alpha_side)
            
            try:
                res = minimize(obj, current_params, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success:
                    current_params = res.x
            except Exception:
                pass
            
            # Increase smoothness parameters
            beta_gap *= 2.0
            alpha_gap *= 2.0
            beta_side *= 2.0
            alpha_side *= 2.0
            
            # Check validity of current params
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = current_params[3*i]
                pos[i, 1] = current_params[3*i+1]
                ang[i] = current_params[3*i+2]
            
            if is_valid_configuration(pos, ang):
                side, phi, cx, cy = compute_exact_side(pos, ang)
                if side < best_side:
                    best_side = side
                    best_params = current_params.copy()

    # Final Polish
    if best_params is not None:
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = best_params[3*i]
            pos[i, 1] = best_params[3*i+1]
            ang[i] = best_params[3*i+2]
        
        # High precision local search
        def obj_exact(p):
             # Use high beta
             return compute_smooth_side_and_penalty(p, 500.0, 500.0, 200.0, 200.0)
             
        try:
            res = minimize(obj_exact, best_params, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success:
                new_params = res.x
                pos_new = np.zeros((N, 2))
                ang_new = np.zeros(N)
                for i in range(N):
                    pos_new[i, 0] = new_params[3*i]
                    pos_new[i, 1] = new_params[3*i+1]
                    ang_new[i] = new_params[3*i+2]
                
                if is_valid_configuration(pos_new, ang_new):
                    side_new, phi_new, cx_new, cy_new = compute_exact_side(pos_new, ang_new)
                    if side_new < best_side:
                        best_side = side_new
                        best_params = new_params
        except Exception:
            pass

    # Extract final result
    if best_params is None:
        # Fallback
        best_params = np.zeros(3*N + 1)
        for i in range(N):
            best_params[3*i] = init_pos[i, 0]
            best_params[3*i+1] = init_pos[i, 1]
            best_params[3*i+2] = init_ang[i]
        best_params[3*N] = 0.0
        best_side, _, _, _ = compute_exact_side(init_pos, init_ang)

    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(N):
        pos[i, 0] = best_params[3*i]
        pos[i, 1] = best_params[3*i+1]
        ang[i] = best_params[3*i+2]
    
    # Get exact center and side
    side, phi, cx, cy = compute_exact_side(pos, ang)
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(ang[i])])
        
    center = (float(cx), float(cy))
    side = float(side)
    angle = float(phi)
    
    return inner, center, side, angle


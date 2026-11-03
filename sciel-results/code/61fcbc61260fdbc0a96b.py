import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # Safety margin

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

    def compute_min_outer_side(positions, angles_deg, phi_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        return np.max(side_vals), center[0], center[1]

    def calc_overlap_penalty(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        
        # Precompute rotated vertices and normals for all hexagons
        # verts[i, j, 0/1]
        verts = np.zeros((N, 6, 2))
        norms = np.zeros((N, 6, 2))
        
        for i in range(N):
            ci, si = c[i], s[i]
            # Rotate vertices
            rx = _ux * ci - _uy * si
            ry = _ux * si + _uy * ci
            verts[i, :, 0] = positions[i, 0] + rx
            verts[i, :, 1] = positions[i, 1] + ry
            
            # Rotate normals
            rnx = _enx * ci - _eny * si
            rny = _enx * si + _eny * ci
            norms[i, :, 0] = rnx
            norms[i, :, 1] = rny
            
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Check separation using normals of i and j
                # For each normal of i, project both
                for k in range(6):
                    nx, ny = norms[i, k, 0], norms[i, k, 1]
                    p_i = np.dot(verts[i], [nx, ny])
                    p_j = np.dot(verts[j], [nx, ny])
                    # Overlap if max(i) > min(j) and max(j) > min(i)
                    # Depth = max(0, min(max_i, max_j) - max(min_i, min_j))
                    # Actually, for convex polys, if they overlap, there is no separating axis.
                    # We sum the overlap depth along all axes as a penalty.
                    min_i, max_i = p_i.min(), p_i.max()
                    min_j, max_j = p_j.min(), p_j.max()
                    overlap = min(max_i, max_j) - max(min_i, min_j)
                    if overlap > 0:
                        penalty += overlap * overlap # Quadratic penalty for smoothness
                
                for k in range(6):
                    nx, ny = norms[j, k, 0], norms[j, k, 1]
                    p_i = np.dot(verts[i], [nx, ny])
                    p_j = np.dot(verts[j], [nx, ny])
                    min_i, max_i = p_i.min(), p_i.max()
                    min_j, max_j = p_j.min(), p_j.max()
                    overlap = min(max_i, max_j) - max(min_i, min_j)
                    if overlap > 0:
                        penalty += overlap * overlap
                        
        return penalty

    def objective(params):
        # params: 12 x (x, y, angle) + 1 (phi) = 37
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles_deg[i] = params[3*i+2]
        phi_deg = params[3*N]
        
        side, cx, cy = compute_min_outer_side(positions, angles_deg, phi_deg)
        pen = calc_overlap_penalty(positions, angles_deg)
        
        # Penalty weight
        lambda_pen = 100.0
        return side + lambda_pen * pen

    # Initialization: Honeycomb
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    
    best_side = float('inf')
    best_params = None
    
    # Run multiple optimizations with different initializations
    num_runs = 5
    run_budget = DEADLINE / num_runs
    
    for run_idx in range(num_runs):
        if time.time() - t0 > DEADLINE - 10:
            break
            
        # Create initial params
        init_params = np.zeros(3*N + 1)
        for idx, (i, j) in enumerate(lattice_coords):
            x, y = i * a1 + j * a2
            init_params[3*idx] = x
            init_params[3*idx+1] = y
            init_params[3*idx+2] = 0.0
            
        # Add some randomness to escape local minima
        if run_idx > 0:
            noise = random.gauss(0, 0.1)
            init_params[:3*N:3] += noise
            init_params[1:3*N:3] += noise
            init_params[2:3*N:3] += random.gauss(0, 5.0)
            
        # Initial phi
        init_params[3*N] = run_idx * 10.0
        
        # Optimize
        try:
            # Use Nelder-Mead for robustness with non-smooth penalties
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.fun < best_side:
                # Verify validity and get exact side
                positions = np.zeros((N, 2))
                angles_deg = np.zeros(N)
                for i in range(N):
                    positions[i, 0] = res.x[3*i]
                    positions[i, 1] = res.x[3*i+1]
                    angles_deg[i] = res.x[3*i+2]
                phi_deg = res.x[3*N]
                
                # Check if penalty is negligible
                pen = calc_overlap_penalty(positions, angles_deg)
                if pen < 1e-4:
                    side, cx, cy = compute_min_outer_side(positions, angles_deg, phi_deg)
                    if side < best_side:
                        best_side = side
                        best_params = res.x.copy()
                        
        except Exception as e:
            pass

    # If no valid solution found, fallback to initial honeycomb
    if best_params is None:
        best_params = np.zeros(3*N + 1)
        for idx, (i, j) in enumerate(lattice_coords):
            x, y = i * a1 + j * a2
            best_params[3*idx] = x
            best_params[3*idx+1] = y
            best_params[3*idx+2] = 0.0
        best_params[3*N] = 0.0
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = best_params[3*i]
            positions[i, 1] = best_params[3*i+1]
            angles_deg[i] = best_params[3*i+2]
        phi_deg = best_params[3*N]
        best_side, best_cx, best_cy = compute_min_outer_side(positions, angles_deg, phi_deg)
    else:
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = best_params[3*i]
            positions[i, 1] = best_params[3*i+1]
            angles_deg[i] = best_params[3*i+2]
        phi_deg = best_params[3*N]
        best_side, best_cx, best_cy = compute_min_outer_side(positions, angles_deg, phi_deg)

    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles_deg[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(phi_deg)
    
    return inner, center, side, angle

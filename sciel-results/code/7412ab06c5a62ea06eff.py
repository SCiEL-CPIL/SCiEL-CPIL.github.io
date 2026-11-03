import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 115.0  # Safety margin before 120s wall-clock kill

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
        ox = offsets[:, 0]
        oy = offsets[:, 1]
        
        best_side = float('inf')
        best_phi = 0.0
        
        normal_offsets = np.pi/6 + np.arange(6) * np.pi/3
        
        phi_vals = np.arange(0, 60, 0.5)
        phi_radians = np.radians(phi_vals)
        
        all_normals = phi_radians[:, None] + normal_offsets[None, :]
        nx_all = np.cos(all_normals)
        ny_all = np.sin(all_normals)
        
        proj = ox[None, :, None] * nx_all[:, None, :] + oy[None, :, None] * ny_all[:, None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=1)
        side_vals = (2.0 / sqrt3) * np.max(max_projs, axis=1)
        
        best_idx = np.argmin(side_vals)
        best_side = side_vals[best_idx]
        best_phi = phi_vals[best_idx]
        
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            
            def eval_phi_fast(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + normal_offsets
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = ox[:, None] * nx[None, :] + oy[:, None] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)
            
            res = minimize_scalar(eval_phi_fast, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-10, 'maxiter': 50})
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
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Honeycomb initial configuration
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
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

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # New Approach: Spectral Graph Partitioning with Gradient Refinement
    # 1. Build Graph
    # 2. Compute Fiedler Vector
    # 3. Use as Initial Layout for Nelder-Mead
    
    def spectral_layout_from_graph(adj_matrix):
        D = np.diag(adj_matrix.sum(axis=1))
        L = D - adj_matrix
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            idx_sorted = np.argsort(eigenvalues)
            eigenvectors = eigenvectors[:, idx_sorted]
            # Use 2nd and 3rd eigenvectors (indices 1 and 2)
            fiedler_1 = eigenvectors[:, 1]
            fiedler_2 = eigenvectors[:, 2]
            
            # Normalize and scale
            f1_range = np.max(fiedler_1) - np.min(fiedler_1)
            f2_range = np.max(fiedler_2) - np.min(fiedler_2)
            scale = 3.0 / max(f1_range, f2_range, 0.1) # Rough scale to honeycomb size
            
            pos = np.column_stack((fiedler_1 * scale, fiedler_2 * scale))
            pos -= np.mean(pos, axis=0)
            return pos
        except Exception:
            return base_positions.copy()

    # Build honeycomb adjacency
    W = np.zeros((N, N))
    lattice_to_idx = {coord: idx for idx, coord in enumerate(lattice_coords)}
    neighbor_vectors = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
    for i in range(N):
        li, lj = lattice_coords[i]
        for di, dj in neighbor_vectors:
            key = (li+di, lj+dj)
            if key in lattice_to_idx:
                j = lattice_to_idx[key]
                W[i, j] = 1.0
                W[j, i] = 1.0

    spec_pos = spectral_layout_from_graph(W)
    spec_ang = np.zeros(N)
    
    # Check validity of spectral layout
    if is_valid_configuration(spec_pos, spec_ang):
        s, p, cx, cy = compute_min_outer_side(spec_pos, spec_ang)
        if s < best_side:
            best_side = s
            best_positions = spec_pos
            best_angles = spec_ang
            best_phi = p
            best_cx = cx
            best_cy = cy
    else:
        # Fallback to honeycomb if spectral layout is invalid
        s, p, cx, cy = compute_min_outer_side(base_positions, base_angles)
        best_side = s
        best_positions = base_positions
        best_angles = base_angles
        best_phi = p
        best_cx = cx
        best_cy = cy

    # Refinement: Nelder-Mead with penalty for overlap
    # We optimize positions only, angles are fixed at 0 for simplicity in this phase
    # (Rotations can be added later if needed, but honeycomb is 0-rotated)
    
    def objective(params):
        pos = np.zeros((N, 2))
        for i in range(N):
            pos[i, 0] = params[2*i]
            pos[i, 1] = params[2*i+1]
        
        # Penalty for overlap
        penalty = 0.0
        angles_rad = np.zeros(N)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], 0.0)
                h2 = (pos[j, 0], pos[j, 1], 0.0)
                if not hexagons_disjoint_fast(h1, h2):
                    # Calculate penetration depth roughly
                    dx = pos[i,0] - pos[j,0]
                    dy = pos[i,1] - pos[j,1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6:
                        penalty += 100.0
                    else:
                        # Approximate min distance between centers for disjoint hexagons is sqrt(3)
                        # If dist < sqrt(3), they might overlap
                        # Simple penalty: square of violation
                        if dist < sqrt3:
                            penalty += (sqrt3 - dist)**2 * 10.0
        
        if penalty > 1.0:
            return penalty + 10.0 # Large penalty
            
        side, phi, cx, cy = compute_min_outer_side(pos, np.zeros(N))
        return side + penalty

    init_params = np.zeros(2*N)
    for i in range(N):
        init_params[2*i] = best_positions[i, 0]
        init_params[2*i+1] = best_positions[i, 1]

    # Run Nelder-Mead
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if res.fun < best_side:
            new_pos = np.zeros((N, 2))
            for i in range(N):
                new_pos[i, 0] = res.x[2*i]
                new_pos[i, 1] = res.x[2*i+1]
            
            if is_valid_configuration(new_pos, np.zeros(N)):
                new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, np.zeros(N))
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_pos
                    best_angles = np.zeros(N)
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
    except Exception:
        pass

    # Try a few random perturbations of the best solution and re-optimize
    for _ in range(3):
        if time.time() - t0 > DEADLINE - 5: break
        
        perturbed_pos = best_positions + np.random.randn(N, 2) * 0.1
        perturbed_ang = np.zeros(N)
        
        if is_valid_configuration(perturbed_pos, perturbed_ang):
            p_params = np.zeros(2*N)
            for i in range(N):
                p_params[2*i] = perturbed_pos[i, 0]
                p_params[2*i+1] = perturbed_pos[i, 1]
            
            try:
                res = minimize(objective, p_params, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_side:
                    new_pos = np.zeros((N, 2))
                    for i in range(N):
                        new_pos[i, 0] = res.x[2*i]
                        new_pos[i, 1] = res.x[2*i+1]
                    if is_valid_configuration(new_pos, np.zeros(N)):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, np.zeros(N))
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = np.zeros(N)
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, base_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

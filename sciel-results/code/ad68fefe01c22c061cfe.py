import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 118.0

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

    def compute_min_outer_side(positions, angles_deg, center_override=None, angle_override=None):
        angles_rad = np.radians(angles_deg)
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        all_verts = np.stack((verts_x, verts_y), axis=-1)
        verts_flat = all_verts.reshape(-1, 2)
        
        if center_override is not None:
            center = np.array(center_override)
        else:
            center = np.mean(verts_flat, axis=0)
            
        offsets = verts_flat - center
        
        if angle_override is not None:
            phi = np.radians(angle_override)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        else:
            # Find optimal phi for this configuration
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
            for phi_deg in range(0, 60, 2):
                s_val = eval_phi(phi_deg)
                if s_val < best_side:
                    best_side = s_val
            return best_side

    def generate_symmetric_positions(r1, r2, delta_deg, alpha_deg, phi_deg):
        """
        Generates 12 hexagons in two rings of 6.
        r1, r2: radii of the rings
        delta_deg: angular offset of outer ring relative to inner
        alpha_deg: rotation of all inner hexagons
        phi_deg: rotation of the outer container
        """
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        
        # Inner ring: 6 hexagons
        for k in range(6):
            ang_rad = np.radians(k * 60.0)
            positions[k, 0] = r1 * math.cos(ang_rad)
            positions[k, 1] = r1 * math.sin(ang_rad)
            angles[k] = alpha_deg
            
        # Outer ring: 6 hexagons
        for k in range(6):
            ang_rad = np.radians(k * 60.0 + delta_deg)
            positions[k+6, 0] = r2 * math.cos(ang_rad)
            positions[k+6, 1] = r2 * math.sin(ang_rad)
            angles[k+6] = alpha_deg
            
        return positions, angles

    def objective_symmetric(params):
        r1, r2, delta_deg, alpha_deg, phi_deg = params
        if r1 < 0 or r2 < 0:
            return 1e6
            
        positions, angles = generate_symmetric_positions(r1, r2, delta_deg, alpha_deg, phi_deg)
        
        # Check collisions
        angles_rad = np.radians(angles)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return 1e6 + (1.0 / (r1 + r2 + 0.1)) # Penalty for collision
            
        # Compute side length with fixed phi
        side = compute_min_outer_side(positions, angles, center_override=(0,0), angle_override=phi_deg)
        return side

    def refine_full_dof(positions, angles, time_limit):
        """
        Takes a symmetric configuration and refines it using Nelder-Mead on all 36 DOF.
        """
        def objective_full(params):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
                
            angles_rad = np.radians(ang)
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                    h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                    if not hexagons_disjoint_fast(h1, h2):
                        return 1e6
                        
            # For full DOF refinement, we also need to optimize center and phi.
            # To keep it simple and fast, we fix center to mean of vertices and optimize phi.
            side = compute_min_outer_side(pos, ang)
            return side

        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = positions[i, 0]
            init_params[3*i+1] = positions[i, 1]
            init_params[3*i+2] = angles[i]
            
        t_start = time.time()
        try:
            res = minimize(objective_full, init_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6})
            
            if time.time() - t_start > time_limit:
                return None, None, None
                
            if res.fun < 1e5:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                new_side = compute_min_outer_side(new_pos, new_ang)
                return new_side, new_pos, new_ang
        except Exception:
            pass
        return None, None, None

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Phase 1: Symmetric Optimization
    # Try several initial guesses for the symmetric parameters
    # r1, r2, delta, alpha, phi
    # Approximate distances: Hexagon width is 2, height is sqrt(3).
    # Inner ring radius should be > 1 (to avoid center collision if there was one, but here they touch each other).
    # Min distance between centers of adjacent hexagons in a ring of radius r is 2*r*sin(30) = r.
    # So r1 must be large enough so that hexagons don't overlap.
    # If aligned, min center dist is approx 1.5? Let's guess r1=1.5, r2=3.0.
    
    initial_guesses = [
        [1.5, 3.0, 0.0, 0.0, 0.0],
        [1.5, 3.0, 30.0, 0.0, 0.0],
        [1.2, 2.5, 0.0, 0.0, 0.0],
        [1.8, 3.5, 0.0, 30.0, 0.0],
        [1.5, 3.2, 15.0, 0.0, 30.0],
    ]
    
    for guess in initial_guesses:
        if time.time() - t0 > DEADLINE - 10.0:
            break
        try:
            res = minimize(objective_symmetric, guess, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_side and res.fun < 1e5:
                r1, r2, delta_deg, alpha_deg, phi_deg = res.x
                positions, angles = generate_symmetric_positions(r1, r2, delta_deg, alpha_deg, phi_deg)
                # Verify and get precise side
                side = compute_min_outer_side(positions, angles, center_override=(0,0), angle_override=phi_deg)
                if side < best_side:
                    best_side = side
                    best_positions = positions
                    best_angles = angles
                    best_phi = phi_deg
        except Exception:
            pass

    if best_positions is None:
        # Fallback to honeycomb if symmetric optimization failed
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_positions = np.zeros((N, 2))
        best_angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            best_positions[idx] = i * a1 + j * a2
            best_angles[idx] = 0.0
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Phase 2: Local Refinement on Full DOF
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        refine_time = min(remaining_time - 2.0, 30.0)
        refined_side, refined_pos, refined_ang = refine_full_dof(best_positions, best_angles, refine_time)
        
        if refined_side is not None and refined_side < best_side:
            best_side = refined_side
            best_positions = refined_pos
            best_angles = refined_ang
            # Recompute center and phi for the final result
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification and center calculation
    if not best_positions is None:
        # Ensure validity
        angles_rad = np.radians(best_angles)
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                h1 = (best_positions[i, 0], best_positions[i, 1], angles_rad[i])
                h2 = (best_positions[j, 0], best_positions[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback
            a1 = np.array([sqrt3, 0.0])
            a2 = np.array([sqrt3/2.0, 1.5])
            lattice_coords = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, -1), (-1, 1), (2, -1), (-2, 1),
                (1, 1), (-1, -1), (2, 0),
            ]
            best_positions = np.zeros((N, 2))
            best_angles = np.zeros(N)
            for idx, (i, j) in enumerate(lattice_coords):
                best_positions[idx] = i * a1 + j * a2
                best_angles[idx] = 0.0
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        else:
            # Recalculate optimal center and phi for the final positions
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

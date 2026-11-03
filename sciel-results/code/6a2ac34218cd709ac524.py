import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices (flat-top orientation at angle 0)
# Vertices at angles 0, 60, ..., 300 degrees
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

# Edge normals for the unit hexagon
# Edge i connects vertex i to vertex i+1
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    # Normal pointing outward
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

# Precompute flat arrays for speed
_ux = _UNIT_HEX_VERTS[:, 0]
_uy = _UNIT_HEX_VERTS[:, 1]
_enx = _EDGE_NORMALS[:, 0]
_eny = _EDGE_NORMALS[:, 1]

def get_rotated_hex_verts(x, y, angle_rad):
    """Returns (6,2) array of vertices for a unit hexagon at (x,y) with rotation angle_rad."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    # Rotation matrix R = [[c, -s], [s, c]]
    # v_rot = R * v_unit
    rx = _ux * c - _uy * s
    ry = _ux * s + _uy * c
    return np.column_stack((rx + x, ry + y))

def get_rotated_normals(angle_rad):
    """Returns (6,2) array of outward normals for a unit hexagon with rotation angle_rad."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rx = _enx * c - _eny * s
    ry = _enx * s + _eny * c
    return np.column_stack((rx, ry))

def compute_penetration_depth(v1, n1, v2, n2):
    """
    Computes the minimum penetration depth between two hexagons.
    v1, n1: vertices and normals of hex 1 (6,2)
    v2, n2: vertices and normals of hex 2 (6,2)
    Returns 0 if disjoint, >0 if overlapping.
    """
    min_depth = float('inf')
    
    # Check normals of hex 1
    for i in range(6):
        nx, ny = n1[i]
        # Project both hexagons onto this normal
        p1 = v1[:, 0] * nx + v1[:, 1] * ny
        p2 = v2[:, 0] * nx + v2[:, 1] * ny
        
        max1 = np.max(p1)
        min1 = np.min(p1)
        max2 = np.max(p2)
        min2 = np.min(p2)
        
        # Separation condition: max1 < min2 or max2 < min1
        # Overlap amount: max(0, max1 - min2, max2 - min1)
        overlap = max(0, max1 - min2, max2 - min1)
        if overlap < min_depth:
            min_depth = overlap
        if min_depth == 0:
            return 0 # Disjoint found
            
    # Check normals of hex 2
    for i in range(6):
        nx, ny = n2[i]
        p1 = v1[:, 0] * nx + v1[:, 1] * ny
        p2 = v2[:, 0] * nx + v2[:, 1] * ny
        
        max1 = np.max(p1)
        min1 = np.min(p1)
        max2 = np.max(p2)
        min2 = np.min(p2)
        
        overlap = max(0, max1 - min2, max2 - min1)
        if overlap < min_depth:
            min_depth = overlap
        if min_depth == 0:
            return 0

    return min_depth

def compute_outer_side_and_center(positions, angles_rad, outer_theta_rad):
    """
    Computes the minimal side length of a regular hexagon with rotation outer_theta_rad
    that encloses all inner hexagons. Also returns the optimal center (centroid of all vertices).
    """
    # Collect all vertices
    all_verts = []
    for i in range(N):
        v = get_rotated_hex_verts(positions[i, 0], positions[i, 1], angles_rad[i])
        all_verts.append(v)
    
    verts_flat = np.vstack(all_verts)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center
    
    # Normals of the outer hexagon
    # Outer hexagon has 6 faces. Normals are at angles outer_theta + k*60
    normal_angles = outer_theta_rad + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    
    # Project offsets onto normals
    # proj[k, m] = offset[m] dot normal[k]
    proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
    
    # The distance from center to the furthest vertex in direction k is max(proj[:, k])
    # But the hexagon boundary is at distance R_apothem = S * sqrt(3) / 2
    # So we need S * sqrt(3) / 2 >= max(abs(proj[:, k]))? 
    # Wait, the center is the centroid of the inner pack. The outer hexagon is centered at this point.
    # The condition is that all points must be inside.
    # For a regular hexagon centered at origin, the condition for a point p to be inside is:
    # |p dot n_k| <= S * sqrt(3) / 2 for all 6 normals n_k.
    # So S >= 2 * max_k ( max_m |proj[k, m]| ) / sqrt(3)
    
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=1) # Max projection for each of the 6 directions
    max_proj = np.max(max_projs)
    
    side = (2.0 / math.sqrt(3)) * max_proj
    return side, center

def objective(params):
    """
    Objective function for L-BFGS-B.
    params: [theta_outer, x1, y1, a1, ..., x12, y12, a12]
    """
    t_start = time.time()
    
    theta_outer = params[0]
    positions = np.zeros((N, 2))
    angles_rad = np.zeros(N)
    
    for i in range(N):
        positions[i, 0] = params[1 + 3*i]
        positions[i, 1] = params[2 + 3*i]
        angles_rad[i] = params[3 + 3*i]
        
    # Compute outer side
    side, center = compute_outer_side_and_center(positions, angles_rad, theta_outer)
    
    # Compute collision penalty
    penalty = 0.0
    # Precompute vertices and normals for all hexagons to avoid recalculation in inner loop if possible?
    # For N=12, 66 pairs. Recalculating is cheap enough.
    verts_list = []
    normals_list = []
    for i in range(N):
        v = get_rotated_hex_verts(positions[i, 0], positions[i, 1], angles_rad[i])
        n = get_rotated_normals(angles_rad[i])
        verts_list.append(v)
        normals_list.append(n)
        
    for i in range(N):
        for j in range(i+1, N):
            # Quick distance check
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist_sq = dx*dx + dy*dy
            # Max distance between centers for non-intersection is roughly 2*sqrt(3) ~ 3.46
            # If dist > 4.0, definitely disjoint
            if dist_sq > 16.0:
                continue
                
            depth = compute_penetration_depth(verts_list[i], normals_list[i], verts_list[j], normals_list[j])
            if depth > 0:
                penalty += depth * depth
                
    # Weight for penalty
    # We want penalty to dominate if there is overlap, but allow small overlaps during optimization
    # Side is around 1.2. Penalties should be scaled to be comparable.
    return side + 10.0 * penalty

def construct():
    t0 = time.time()
    
    # Honeycomb initial configuration
    sqrt3 = math.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    
    base_positions = np.zeros((N, 2))
    base_angles_deg = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles_deg[idx] = 0.0
        
    base_theta = 0.0
    
    # Initial params
    init_params = np.zeros(1 + 3*N)
    init_params[0] = base_theta
    for i in range(N):
        init_params[1 + 3*i] = base_positions[i, 0]
        init_params[2 + 3*i] = base_positions[i, 1]
        init_params[3 + 3*i] = np.radians(base_angles_deg[i])
        
    best_side = float('inf')
    best_params = init_params.copy()
    
    # Run multiple random initializations
    num_starts = 10
    time_per_start = (DEADLINE - (time.time() - t0)) / num_starts
    
    for k in range(num_starts):
        if time.time() - t0 > DEADLINE - 5:
            break
            
        # Jitter the initial configuration
        params = init_params.copy()
        if k > 0:
            # Random perturbation
            for i in range(1, len(params)):
                params[i] += random.gauss(0, 0.5)
            # Random outer rotation
            params[0] = random.uniform(0, np.pi/3)
            
        # Run L-BFGS-B
        try:
            res = minimize(objective, params, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 1000})
            
            if res.success or res.fun < best_side:
                # Verify validity
                p = res.x
                theta_outer = p[0]
                positions = np.zeros((N, 2))
                angles_rad = np.zeros(N)
                for i in range(N):
                    positions[i, 0] = p[1 + 3*i]
                    positions[i, 1] = p[2 + 3*i]
                    angles_rad[i] = p[3 + 3*i]
                    
                # Check for any penetration
                valid = True
                verts_list = []
                normals_list = []
                for i in range(N):
                    v = get_rotated_hex_verts(positions[i, 0], positions[i, 1], angles_rad[i])
                    n = get_rotated_normals(angles_rad[i])
                    verts_list.append(v)
                    normals_list.append(n)
                    
                for i in range(N):
                    for j in range(i+1, N):
                        depth = compute_penetration_depth(verts_list[i], normals_list[i], verts_list[j], normals_list[j])
                        if depth > 1e-6: # Small tolerance
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    side, center = compute_outer_side_and_center(positions, angles_rad, theta_outer)
                    if side < best_side:
                        best_side = side
                        best_params = p.copy()
                        
        except Exception as e:
            pass
            
    # Final Polish: Use the best params found and run a tighter optimization
    if time.time() - t0 < DEADLINE - 5:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000})
            p = res.x
            theta_outer = p[0]
            positions = np.zeros((N, 2))
            angles_rad = np.zeros(N)
            for i in range(N):
                positions[i, 0] = p[1 + 3*i]
                positions[i, 1] = p[2 + 3*i]
                angles_rad[i] = p[3 + 3*i]
                
            # Verify
            valid = True
            verts_list = []
            normals_list = []
            for i in range(N):
                v = get_rotated_hex_verts(positions[i, 0], positions[i, 1], angles_rad[i])
                n = get_rotated_normals(angles_rad[i])
                verts_list.append(v)
                normals_list.append(n)
                
            for i in range(N):
                for j in range(i+1, N):
                    depth = compute_penetration_depth(verts_list[i], normals_list[i], verts_list[j], normals_list[j])
                    if depth > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid:
                side, center = compute_outer_side_and_center(positions, angles_rad, theta_outer)
                if side < best_side:
                    best_side = side
                    best_params = p.copy()
        except Exception:
            pass

    # Extract final results
    p = best_params
    theta_outer = p[0]
    positions = np.zeros((N, 2))
    angles_rad = np.zeros(N)
    for i in range(N):
        positions[i, 0] = p[1 + 3*i]
        positions[i, 1] = p[2 + 3*i]
        angles_rad[i] = p[3 + 3*i]
        
    # Fallback if optimization failed or produced invalid result
    if not np.isfinite(best_side) or best_side > 2.0:
        # Use base configuration
        positions = base_positions
        angles_rad = np.radians(base_angles_deg)
        theta_outer = 0.0
        side, center = compute_outer_side_and_center(positions, angles_rad, theta_outer)
        best_side = side
    else:
        side, center = compute_outer_side_and_center(positions, angles_rad, theta_outer)
        
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(np.degrees(angles_rad[i]))])
        
    center_out = (float(center[0]), float(center[1]))
    side_out = float(best_side)
    angle_out = float(np.degrees(theta_outer))
    
    return inner, center_out, side_out, angle_out

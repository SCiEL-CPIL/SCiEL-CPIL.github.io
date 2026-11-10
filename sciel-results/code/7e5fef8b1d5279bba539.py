import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)

    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue
            return False
    return True

def _soft_max(x, beta=100.0):
    m = np.max(x)
    return m + np.log(np.mean(np.exp(x - m))) / beta

def _min_enclosing_side_smooth(positions, angles, beta=100.0):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = _soft_max(projs, beta)
    return max_proj / apothem_factor

def _min_enclosing_side(positions, angles):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

def _separation_penalty(positions, angles, beta=100.0):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    
    max_overlap = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            # overlap on each normal: max(0, min(max_i, max_j) - max(min_i, min_j))
            overlaps = np.maximum(0.0, np.minimum(max_i, max_j) - np.maximum(min_i, min_j))
            max_overlap = max(max_overlap, np.max(overlaps))
    return np.exp(max_overlap * beta)

def _build_symmetric_positions(ring_r, valley_r, angle_offset=0.0, ring_angle=0.0, valley_angle=0.0):
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    positions[0] = [0.0, 0.0]
    angles[0] = angle_offset
    for k in range(6):
        a = np.radians(ring_angle + 60 * k + angle_offset)
        positions[k + 1] = [ring_r * np.cos(a), ring_r * np.sin(a)]
        angles[k + 1] = angle_offset
    for k in range(4):
        a = np.radians(valley_angle + 120 * k + angle_offset)
        positions[7 + k] = [valley_r * np.cos(a), valley_r * np.sin(a)]
        angles[7 + k] = angle_offset
    return positions, angles

def _solve_topology():
    """Solve the 2-DOF symmetric problem: ring_r, valley_r.
    Topology: center + 6 ring at 30+60k + 4 valley at 0+120k (or 60+120k).
    We try both valley offsets and both ring offsets.
    """
    best_positions = None
    best_angles = None
    best_side = float('inf')
    
    # Valley options: which 4 of the 6 valleys?
    # Valleys are at 0, 60, 120, 180, 240, 300.
    # By symmetry, picking 4 is equivalent to picking 2 to exclude.
    # Excluding opposite valleys (e.g., 0 and 180) gives 60,120,240,300 -> 3-fold symmetry
    # Excluding adjacent valleys (e.g., 0 and 60) gives 120,180,240,300 -> lower symmetry
    # Let's try: valleys at [0, 120, 240, 300] (exclude 60, 180) -> not symmetric
    # Actually, 4 valleys: [0, 120, 240, 300] is not 4-fold. 
    # Let's just try all combinations of 4 valleys from 6.
    from itertools import combinations
    valley_angles_all = [0, 60, 120, 180, 240, 300]
    
    for valley_combo in combinations(range(6), 4):
        va_list = [valley_angles_all[i] for i in valley_combo]
        # Ring angles: 30+60k. But we can also rotate the whole thing.
        # For now, fix ring at 30+60k and vary valley selection.
        ring_angle = 30.0
        
        def obj(params):
            ring_r, valley_r = params
            if ring_r < 1.8 or valley_r < 1.8:
                return 1e10
            positions, angles = _build_symmetric_positions(ring_r, valley_r, 0.0, ring_angle, va_list[0])
            # Wait, _build_symmetric_positions assumes 4 valleys at 120k spacing.
            # I need a more general builder.
            return None
        
        # Let me write a general builder for this topology.
        def build_general(ring_r, valley_r, ring_angle, va_list, angle_offset=0.0):
            positions = np.zeros((11, 2))
            angles = np.zeros(11)
            positions[0] = [0.0, 0.0]
            angles[0] = angle_offset
            for k in range(6):
                a = np.radians(ring_angle + 60 * k + angle_offset)
                positions[k + 1] = [ring_r * np.cos(a), ring_r * np.sin(a)]
                angles[k + 1] = angle_offset
            for k, va in enumerate(va_list):
                a = np.radians(va + angle_offset)
                positions[7 + k] = [valley_r * np.cos(a), va * 0 + valley_r * np.sin(a)]
                angles[7 + k] = angle_offset
            return positions, angles
        
        def obj2(params):
            ring_r, valley_r = params
            if ring_r < 1.5 or valley_r < 1.5:
                return 1e10
            positions, angles = build_general(ring_r, valley_r, ring_angle, va_list)
            side = _min_enclosing_side_smooth(positions, angles, beta=50.0)
            pen = _separation_penalty(positions, angles, beta=50.0)
            return side + pen
        
        # Initial guess: ring_r=2, valley_r=3.5
        guess = [2.0, 3.5]
        try:
            res = minimize(obj2, guess, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.success or res.fun < best_side:
                ring_r, valley_r = res.x
                positions, angles = build_general(ring_r, valley_r, ring_angle, va_list)
                if _check_separation_fast(positions, angles):
                    side = _min_enclosing_side(positions, angles)
                    if side < best_side:
                        best_side = side
                        best_positions = positions.copy()
                        best_angles = angles.copy()
        except Exception:
            pass
    
    return best_positions, best_angles, best_side

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0

    best_positions = None
    best_angles = None
    best_side = float('inf')

    # Phase 1: Solve topology
    res = _solve_topology()
    if res[0] is not None:
        best_positions, best_angles, best_side = res

    # Fallback
    if best_positions is None:
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for k in range(4):
            a = np.radians(0 + 120 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)

    # Phase 2: Global rotation optimization
    if time.time() < deadline - 60:
        def obj_rot(theta):
            c = np.cos(np.radians(theta))
            s = np.sin(np.radians(theta))
            rot_mat = np.array([[c, -s], [s, c]])
            pos_rot = best_positions @ rot_mat.T
            angles_rot = (best_angles + theta) % 360
            return _min_enclosing_side_smooth(pos_rot, angles_rot, beta=100.0)

        best_rot = 0.0
        best_rot_side = best_side
        for t in np.linspace(0, 60, 121):
            s_t = obj_rot(t)
            if s_t < best_rot_side:
                best_rot_side = s_t
                best_rot = t
        try:
            res_rot = minimize(obj_rot, [best_rot], method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-12})
            if res_rot.fun < best_rot_side:
                best_rot = res_rot.x[0]
                best_rot_side = res_rot.fun
        except Exception:
            pass
        if best_rot_side < best_side:
            best_side = best_rot_side
            c = np.cos(np.radians(best_rot))
            s = np.sin(np.radians(best_rot))
            rot_mat = np.array([[c, -s], [s, c]])
            best_positions = best_positions @ rot_mat.T
            best_angles = (best_angles + best_rot) % 360

    # Final verification
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        # Fallback to known valid
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for k in range(4):
            a = np.radians(0 + 120 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)

    if not _check_containment_fast(best_positions, best_angles, final_side):
        final_side *= 1.001

    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)

    return inner, outer_hex_data, side_length

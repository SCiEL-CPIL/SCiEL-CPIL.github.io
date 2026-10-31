"""Packing 11 unit hexagons in a regular hexagon - optimized approach.
Uses honeycomb initial configurations with targeted optimization of
global rotation and local adjustments."""
import numpy as np
import math
import time

N = 11
S3 = math.sqrt(3)

def _hex_vertices(cx, cy, s, ang_deg):
    """Vertices of a regular hexagon (pointy-top by default)."""
    a = math.radians(ang_deg)
    # Pointy-top: first vertex at 90 degrees
    angs = a + np.pi/2 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _hex_vertices_flat(cx, cy, s, ang_deg):
    """Vertices of a flat-top hexagon."""
    a = math.radians(ang_deg)
    angs = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _support_projection(verts, normal):
    """Max projection of vertices onto a unit normal."""
    return np.max(verts @ normal)

def _required_side_length(positions, angles, outer_center, outer_rotation_deg):
    """
    Compute exact minimum side length of outer hexagon.
    The outer hexagon must contain all inner hexagons.
    For a regular hexagon with side S and rotation phi, the constraint is:
    For each of the 6 outward normals n_k, max_i max_v (v . n_k) <= S * sqrt(3)/2
    where v is relative to outer center.
    """
    all_verts = np.zeros((N * 6, 2))
    for i in range(N):
        all_verts[i*6:(i+1)*6] = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
    
    # Translate relative to outer center
    all_verts = all_verts - np.array(outer_center)
    
    # Outer hexagon normals: for a hexagon with rotation phi,
    # the 6 outward normals are at angles phi + 0, phi+60, ..., phi+300
    phi = math.radians(outer_rotation_deg)
    normals = []
    for k in range(6):
        ang = phi + k * math.pi / 3
        normals.append(np.array([math.cos(ang), math.sin(ang)]))
    
    max_proj = 0.0
    for n in normals:
        proj = np.max(all_verts @ n)
        if proj > max_proj:
            max_proj = proj
    
    # For a regular hexagon with side S, the apothem (distance from center to edge) is S*sqrt(3)/2
    # So S = max_proj / (sqrt(3)/2)
    return max_proj / (S3 / 2.0)

def _check_disjoint(positions, angles, tol=1e-9):
    """Check that all hexagons are pairwise disjoint using SAT."""
    for i in range(N):
        for j in range(i + 1, N):
            v1 = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            v2 = _hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            
            # SAT: project onto all edge normals of both hexagons
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            
            for e in list(edges1) + list(edges2):
                nv = np.array([-e[1], e[0]])
                norm = np.linalg.norm(nv)
                if norm < 1e-12:
                    continue
                nv /= norm
                proj1 = v1 @ nv
                proj2 = v2 @ nv
                gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
                if gap < -tol:
                    return False
    return True

def _axial_to_xy(q, r):
    """Convert axial coordinates to Cartesian for pointy-top hex grid."""
    x = 1.5 * q
    y = S3 * (r + q / 2.0)
    return (x, y)

def _build_honeycomb_configs():
    """Generate several honeycomb-based initial configurations."""
    configs = []
    
    # Config 1: 3-4-3 (center + 6 ring + 4 outer)
    coords_343 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    positions = [_axial_to_xy(q, r) for q, r in coords_343]
    angles = [0.0] * N
    configs.append((positions, angles))
    
    # Config 2: Different outer ring selection
    coords_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (0, -2)
    ]
    positions = [_axial_to_xy(q, r) for q, r in coords_2]
    angles = [0.0] * N
    configs.append((positions, angles))
    
    # Config 3: Another variant
    coords_3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-2, 0), (-1, -1)
    ]
    positions = [_axial_to_xy(q, r) for q, r in coords_3]
    angles = [0.0] * N
    configs.append((positions, angles))
    
    # Config 4: 2-3-3-3 variant
    coords_4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1),
        (2, 0), (1, 1), (0, 2),
        (-1, 0), (0, -1), (1, -1),
        (-2, 1)
    ]
    positions = [_axial_to_xy(q, r) for q, r in coords_4]
    angles = [0.0] * N
    configs.append((positions, angles))
    
    return configs

def _optimize_outer_rotation(positions, angles, outer_center):
    """Find optimal outer rotation for fixed inner configuration."""
    best_rot = 0.0
    best_side = float('inf')
    
    # Try rotations from 0 to 30 degrees (hexagonal symmetry)
    for rot in np.linspace(0, 30, 31):
        side = _required_side_length(positions, angles, outer_center, rot)
        if side < best_side:
            best_side = side
            best_rot = rot
    
    # Refine with golden section search
    lo, hi = max(0, best_rot - 1), min(30, best_rot + 1)
    gr = (math.sqrt(5) - 1) / 2
    for _ in range(50):
        m1 = hi - gr * (hi - lo)
        m2 = lo + gr * (hi - lo)
        f1 = _required_side_length(positions, angles, outer_center, m1)
        f2 = _required_side_length(positions, angles, outer_center, m2)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
        if hi - lo < 1e-10:
            break
    
    best_rot = (lo + hi) / 2
    best_side = _required_side_length(positions, angles, outer_center, best_rot)
    return best_rot, best_side

def _optimize_positions_nm(positions, angles, outer_center, outer_rot, max_iter=200):
    """Nelder-Mead optimization of inner positions (keeping angles fixed at 0)."""
    from scipy.optimize import minimize
    
    x0 = np.array([p[0] for p in positions] + [p[1] for p in positions])
    
    def objective(x):
        pos = [(x[i], x[i + N]) for i in range(N)]
        # Check disjointness with penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                v1 = _hex_vertices(pos[i][0], pos[i][1], 1.0, angles[i])
                v2 = _hex_vertices(pos[j][0], pos[j][1], 1.0, angles[j])
                edges1 = np.roll(v1, -1, axis=0) - v1
                edges2 = np.roll(v2, -1, axis=0) - v2
                for e in list(edges1) + list(edges2):
                    nv = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(nv)
                    if norm < 1e-12:
                        continue
                    nv /= norm
                    proj1 = v1 @ nv
                    proj2 = v2 @ nv
                    gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
                    if gap < 0:
                        penalty += gap * gap * 1000.0
        side = _required_side_length(pos, angles, outer_center, outer_rot)
        return side + penalty
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-9,
                               'adaptive': True})
        if res.success or res.fun < float('inf'):
            new_pos = [(res.x[i], res.x[i + N]) for i in range(N)]
            if _check_disjoint(new_pos, angles, tol=1e-8):
                return new_pos
    except Exception:
        pass
    return positions

def _try_rotated_hexagons(positions, base_angles):
    """Try giving some hexagons 30-degree rotation (flat-top) and see if it helps."""
    # Try rotating outer ring hexagons by 30 degrees
    best = (positions, base_angles)
    best_side = float('inf')
    
    for rot_mask in [0, 1, 2, 3, 4]:
        # rot_mask: which hexagons to rotate (bitmask for indices 7-10)
        angles = base_angles.copy()
        for k in range(4):
            if rot_mask & (1 << k):
                angles[7 + k] = 30.0
        
        # Quick check
        if not _check_disjoint(positions, angles, tol=1e-8):
            continue
        
        outer_center = (0.0, 0.0)
        rot, side = _optimize_outer_rotation(positions, angles, outer_center)
        if side < best_side:
            best_side = side
            best = (positions, angles)
    
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit
    
    best_result = None
    best_side = float('inf')
    
    configs = _build_honeycomb_configs()
    
    for idx, (positions, angles) in enumerate(configs):
        if time.time() > deadline - 30.0:
            break
        
        # Center the configuration
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        positions_centered = [(p[0] - cx, p[1] - cy) for p in positions]
        
        # Try with outer center at origin
        outer_center = (0.0, 0.0)
        
        # Optimize outer rotation
        outer_rot, side = _optimize_outer_rotation(positions_centered, angles, outer_center)
        
        # Check disjointness
        if _check_disjoint(positions_centered, angles, tol=1e-8):
            if side < best_side:
                best_side = side
                best_result = (positions_centered, angles, outer_rot)
        
        # Try position optimization if time allows
        if time.time() < deadline - 10.0:
            opt_positions = _optimize_positions_nm(positions_centered, angles, outer_center, outer_rot, max_iter=100)
            if opt_positions != positions_centered:
                # Re-optimize outer rotation
                outer_rot2, side2 = _optimize_outer_rotation(opt_positions, angles, outer_center)
                if _check_disjoint(opt_positions, angles, tol=1e-8) and side2 < best_side:
                    best_side = side2
                    best_result = (opt_positions, angles, outer_rot2)
        
        # Try 30-degree rotations for outer hexagons
        if time.time() < deadline - 5.0:
            pos_rot, ang_rot = _try_rotated_hexagons(positions_centered, angles)
            if ang_rot != angles:
                outer_center2 = (0.0, 0.0)
                rot2, side3 = _optimize_outer_rotation(pos_rot, ang_rot, outer_center2)
                if _check_disjoint(pos_rot, ang_rot, tol=1e-8) and side3 < best_side:
                    best_side = side3
                    best_result = (pos_rot, ang_rot, rot2)
    
    # If no valid result, use fallback
    if best_result is None:
        positions, angles = configs[0]
        outer_rot = 0.0
        best_side = _required_side_length(positions, angles, (0.0, 0.0), outer_rot)
        best_result = (positions, angles, outer_rot)
    
    positions, angles, outer_rot = best_result
    
    # Final verification
    if not _check_disjoint(positions, angles, tol=1e-7):
        # Fallback to linear arrangement
        positions = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        angles = [0.0] * N
        outer_rot = 0.0
        best_side = _required_side_length(positions, angles, (0.0, 0.0), outer_rot)
    
    # Add tiny epsilon for safety
    final_side = best_side * (1.0 + 1e-9)
    
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), final_side, outer_rot

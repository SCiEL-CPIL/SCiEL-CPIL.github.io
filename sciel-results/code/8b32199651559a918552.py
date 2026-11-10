# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_edge_normals(verts):
    """Compute outward unit normals for each edge of a hexagon given its 6 vertices."""
    norms = []
    for k in range(6):
        p1 = verts[k]
        p2 = verts[(k + 1) % 6]
        e = p2 - p1
        n = np.array([e[1], -e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-15:
            n = n / norm
        else:
            n = np.array([1.0, 0.0])
        norms.append(n)
    return np.array(norms)


def _max_overlap(v1, v2):
    """
    Compute the maximum overlap between two hexagons.
    Returns 0.0 if separated (gap > 0 on at least one normal),
    positive value if overlapping (no separating normal found).
    """
    n1 = _hex_edge_normals(v1)
    n2 = _hex_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_gap = float('inf')
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        gap = max(max1 - min2, max2 - min1)
        if gap < min_gap:
            min_gap = gap
    
    # If min_gap > 1e-6, they're separated
    # If min_gap <= 1e-6, they overlap or touch
    return max(0.0, 1e-6 - min_gap)


def _min_enclosing_hex_side(inner_hex_data, outer_angle_deg=0.0):
    """
    Compute minimum side of regular hexagon (centered at origin, given angle) containing all inner hexes.
    Returns side length (float).
    """
    outer_angle = np.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = outer_angle + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)  # shape (6, 2)
    
    max_proj = 0.0
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        shifted = verts  # centered at origin
        projs = shifted @ normals.T
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    
    return max_proj * 2.0 / np.sqrt(3.0)


def _compute_objective(params, outer_angle_deg=0.0, penalty_weight=100.0):
    """
    Penalty-based objective: minimize enclosing hex side + penalty for overlaps.
    params: 33 values = 11 * (x, y, angle_deg)
    """
    inner = params.reshape(11, 3)
    
    # Compute enclosing side
    side = _min_enclosing_hex_side(inner, outer_angle_deg)
    
    # Compute overlap penalty
    total_overlap = 0.0
    n = len(inner)
    verts_list = []
    for i in range(n):
        x, y, a_deg = inner[i]
        verts_list.append(_hex_vertices(x, y, 1.0, a_deg))
    
    for i in range(n):
        for j in range(i + 1, n):
            ov = _max_overlap(verts_list[i], verts_list[j])
            total_overlap += ov
    
    return side + penalty_weight * total_overlap


def _check_validity(inner_hex_data, outer_hex_data, side):
    """Check all validity conditions. Returns True if valid."""
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data))):
        return False
    if not np.isfinite(side):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    ox, oy, oa = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    # Check every inner vertex is inside outer hexagon
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        for p in verts:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j + 1) % 6]
                e = p2 - p1
                v = p - p1
                cross = e[0] * v[1] - e[1] * v[0]
                if cross < -1e-6:
                    return False
    
    # Check every pair of inner hexagons is separated
    n_inner = len(inner_hex_data)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            x1, y1, a1 = inner_hex_data[i]
            x2, y2, a2 = inner_hex_data[j]
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            
            n1 = _hex_edge_normals(v1)
            n2 = _hex_edge_normals(v2)
            all_normals = np.vstack([n1, n2])
            
            separated = False
            for n in all_normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin
    
    # Honeycomb arrangement: 1 center + 6 ring + 4 outer
    # Standard hex grid with spacing 2.5 (touching hexes)
    # Centers in honeycomb pattern:
    # Row 0 (top):    (-3.75, 2.165), (3.75, 2.165)
    # Row 1:          (-1.25, 2.165), (1.25, 2.165)
    # Row 2 (mid):    (-2.5, 0), (0, 0), (2.5, 0)
    # Row 3:          (-1.25, -2.165), (1.25, -2.165)
    # Row 4 (bot):    (-3.75, -2.165), (3.75, -2.165)
    
    h = np.sqrt(3) / 2 * 2.0  # = sqrt(3) ≈ 1.732
    # Actually for unit hexes touching: center-to-center distance = 2 * apothem = sqrt(3)
    # Horizontal spacing for adjacent in same row: 2.5? No.
    # For pointy-top hexes (angle 0 means vertex at 0 deg):
    # Width = 2*side = 2, height = sqrt(3)*side = sqrt(3)
    # Adjacent in row: dx = 2*apothem = sqrt(3) ≈ 1.732? No...
    # Let me use the known working seed positions.
    
    inner_seed = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.165, 0],
            [1.25, 2.165, 0],
            [-1.25, -2.165, 0],
            [1.25, -2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, -2.165, 0],
            [3.75, -2.165, 0],
        ]
    )
    
    outer_seed = np.array([0.0, 0.0, 0.0])
    side_seed = _min_enclosing_hex_side(inner_seed, 0.0)
    
    best_inner = inner_seed.copy()
    best_outer = outer_seed.copy()
    best_side = side_seed
    
    # Strategy: optimize global rotation of the entire cluster + outer hex angle
    # The cluster is rigid (honeycomb), so we only need to rotate it
    # and possibly adjust the outer hex angle
    
    # Precompute cluster vertex offsets (relative to center, angle 0)
    # Each hex has 6 vertices; total 66 vertices
    
    # Try different outer hex angles and cluster rotations
    rng = np.random.default_rng(123)
    
    # Phase 1: Optimize cluster rotation and outer hex angle
    # Parameters: [cluster_rot_deg, outer_angle_deg]
    def obj_rot(params):
        cluster_rot = params[0]
        outer_ang = params[1]
        
        # Rotate all inner hex positions and angles
        c, s = np.cos(np.radians(cluster_rot)), np.sin(np.radians(cluster_rot))
        R = np.array([[c, -s], [s, c]])
        
        inner_rot = np.zeros_like(inner_seed)
        for i in range(11):
            pos = inner_seed[i, :2]
            new_pos = R @ pos
            inner_rot[i, 0] = new_pos[0]
            inner_rot[i, 1] = new_pos[1]
            inner_rot[i, 2] = inner_seed[i, 2] + cluster_rot
        
        side = _min_enclosing_hex_side(inner_rot, outer_ang)
        
        # Check overlaps (should be fine for rigid rotation)
        # But let's verify
        verts_list = [_hex_vertices(inner_rot[i, 0], inner_rot[i, 1], 1.0, inner_rot[i, 2]) for i in range(11)]
        for i in range(11):
            for j in range(i + 1, 11):
                if _max_overlap(verts_list[i], verts_list[j]) > 1e-6:
                    return side + 100.0 * _max_overlap(verts_list[i], verts_list[j])
        
        return side
    
    # Try a grid of rotations
    best_val = best_side
    best_cluster_rot = 0.0
    best_outer_ang = 0.0
    
    for outer_ang in [0, 15, 30, 45, 60, 75, 90]:
        for cluster_rot in np.arange(0, 60, 1.0):
            if time.time() > deadline - 60:
                break
            val = obj_rot([cluster_rot, outer_ang])
            if val < best_val:
                best_val = val
                best_cluster_rot = cluster_rot
                best_outer_ang = outer_ang
    
    # Refine with optimization
    if time.time() < deadline - 60:
        try:
            res = minimize(obj_rot, [best_cluster_rot, best_outer_ang], method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_cluster_rot = res.x[0]
                best_outer_ang = res.x[1]
                best_val = res.fun
        except Exception:
            pass
    
    # Apply best rotation
    c, s = np.cos(np.radians(best_cluster_rot)), np.sin(np.radians(best_cluster_rot))
    R = np.array([[c, -s], [s, c]])
    best_inner = np.zeros_like(inner_seed)
    for i in range(11):
        pos = inner_seed[i, :2]
        new_pos = R @ pos
        best_inner[i, 0] = new_pos[0]
        best_inner[i, 1] = new_pos[1]
        best_inner[i, 2] = inner_seed[i, 2] + best_cluster_rot
    
    best_outer = np.array([0.0, 0.0, best_outer_ang])
    best_side = best_val
    
    # Phase 2: Local optimization of individual positions (penalty-based)
    if time.time() < deadline - 30:
        def full_obj(params):
            return _compute_objective(params, best_outer_ang, penalty_weight=50.0)
        
        x0 = best_inner.flatten()
        try:
            res = minimize(full_obj, x0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < best_side:
                candidate = res.x.reshape(11, 3)
                candidate[:, 2] = candidate[:, 2] % 360.0
                cand_side = _min_enclosing_hex_side(candidate, best_outer_ang)
                if cand_side < best_side and _check_validity(candidate, best_outer, cand_side):
                    best_inner = candidate
                    best_side = cand_side
        except Exception:
            pass
    
    # Phase 3: Try random perturbations of the full configuration
    if time.time() < deadline - 30:
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
            
            scale = 0.3
            x0 = best_inner.flatten() + rng.normal(0, scale, size=33)
            try:
                res = minimize(full_obj, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
                if res.fun < best_side:
                    candidate = res.x.reshape(11, 3)
                    candidate[:, 2] = candidate[:, 2] % 360.0
                    cand_side = _min_enclosing_hex_side(candidate, best_outer_ang)
                    if cand_side < best_side and _check_validity(candidate, best_outer, cand_side):
                        best_inner = candidate
                        best_side = cand_side
            except Exception:
                continue
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END

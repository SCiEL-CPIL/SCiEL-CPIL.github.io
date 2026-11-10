# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

# Precompute normals and width for unit hexagon with angle 0
# Vertices of unit hexagon (angle 0, center 0):
# (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
# Edges:
# 1. (0.5, -sqrt(3)/2) to (1, 0) -> Normal (sqrt(3)/2, 0.5) ? No.
# Let's calculate normals properly.
# Edge 0: v0=(1,0), v1=(0.5, sqrt(3)/2). d = (-0.5, sqrt(3)/2). n = (-sqrt(3)/2, -0.5) or (sqrt(3)/2, 0.5).
# The outward normal for the edge between v0 and v1.
# Center is (0,0). Midpoint is (0.75, sqrt(3)/4). Normal direction is (0.75, sqrt(3)/4) normalized -> (sqrt(3)/2, 0.5).
# Let's just use the 6 directions: 0, 60, 120, 180, 240, 300 degrees?
# The normals to the edges of a flat-top hexagon are at 30, 90, 150, 210, 270, 330 degrees?
# Flat top: Top edge is horizontal. Normal is vertical (90 deg).
# So normals are at 90, 30, -30, -90, -150, 150.
# i.e., 30 + 60*i degrees.

_NORMALS = np.array([[np.cos(np.radians(30 + 60*i)), np.sin(np.radians(30 + 60*i))] for i in range(6)])
_HEX_WIDTH = np.sqrt(3)

def _is_valid_fast(cfg):
    """
    Fast validity check assuming all hexagons have the same orientation (angle=0).
    Uses Separating Axis Theorem for identical regular hexagons.
    Two unit hexagons with same orientation are separated iff
    max_k | (p_i - p_j) . n_k | > sqrt(3).
    """
    # cfg is (11, 3), but we only use x, y
    pos = cfg[:, :2]
    n = len(pos)
    for i in range(n):
        for j in range(i+1, n):
            d = pos[i] - pos[j]
            # Project d onto all 6 normals
            # If any projection magnitude is > sqrt(3), they are separated
            projs = np.abs(d @ _NORMALS.T)
            if np.max(projs) > _HEX_WIDTH + 1e-6:
                continue
            else:
                return False
    return True

def _is_valid(cfg):
    """Full validity check for non-overlapping hexagons. Falls back to general check if angles differ."""
    # Check if all angles are effectively 0 (or same)
    angles = cfg[:, 2]
    if np.allclose(angles, angles[0], atol=1e-6):
        return _is_valid_fast(cfg)
    
    # General check (slow)
    for i in range(11):
        for j in range(i+1, 11):
            v1 = _hex_vertices(cfg[i,0], cfg[i,1], 1.0, cfg[i,2])
            v2 = _hex_vertices(cfg[j,0], cfg[j,1], 1.0, cfg[j,2])
            if not _hex_separated(v1, v2):
                return False
    return True

def _hex_separated(v1, v2, tol=1e-6):
    """Check if two hexagons (vertex arrays) are separated using all 12 edge normals."""
    for i in range(6):
        e = v1[(i+1)%6] - v1[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    for i in range(6):
        e = v2[(i+1)%6] - v2[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    return False

def _compute_outer_side_fast(inner_data, outer_angle=0.0):
    """Compute the minimum outer hexagon side length needed to contain all inner hexagons."""
    a = np.radians(outer_angle)
    all_verts = np.array([_hex_vertices(r[0], r[1], 1.0, r[2]) for r in inner_data])
    
    max_proj = 0.0
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = all_verts @ n
        max_proj = max(max_proj, np.max(proj))
    return 2.0 * max_proj / np.sqrt(3)

def _check_containment(inner_data, outer_center, outer_angle, side):
    """Check if all inner hexagons are contained in the outer hexagon."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    for hex_row in inner_data:
        v = _hex_vertices(hex_row[0], hex_row[1], 1.0, hex_row[2])
        for p in v:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i+1) % 6]
                cross_val = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False
    return True

def _build_hex_cluster_11():
    """
    Build a rigid hexagonal cluster of 11 unit hexagons.
    Structure: center + 6 in first ring + 4 in second ring at specific positions.
    All hexagons have the same orientation (angle=0).
    Returns positions as (11, 2) array.
    """
    sqrt3 = np.sqrt(3)
    dx_e = 3.0
    dx_ne = 1.5
    dy_ne = 2.0 * sqrt3
    
    positions = np.zeros((11, 2))
    positions[0] = [0.0, 0.0]
    positions[1] = [dx_e, 0.0]
    positions[2] = [dx_ne, dy_ne]
    positions[3] = [-dx_ne, dy_ne]
    positions[4] = [-dx_e, 0.0]
    positions[5] = [-dx_ne, -dy_ne]
    positions[6] = [dx_ne, -dy_ne]
    positions[7] = [dx_e, dy_ne]
    positions[8] = [-dx_e, dy_ne]
    positions[9] = [dx_e, -dy_ne]
    positions[10] = [-dx_e, -dy_ne]
    
    return positions

def _build_compact_cluster_11():
    """
    Build a more compact cluster by using a different topology.
    Try a 3-4-4 arrangement or other non-standard topologies.
    """
    sqrt3 = np.sqrt(3)
    dx_e = 3.0
    dx_ne = 1.5
    dy_ne = 2.0 * sqrt3
    
    positions = np.zeros((11, 2))
    positions[0] = [0.0, 0.0]
    positions[1] = [dx_e, 0.0]
    positions[2] = [2*dx_e, 0.0]
    positions[3] = [-dx_ne, dy_ne]
    positions[4] = [dx_ne, dy_ne]
    positions[5] = [dx_e + dx_ne, dy_ne]
    positions[6] = [2*dx_e + dx_ne, dy_ne]
    positions[7] = [0.0, 2*dy_ne]
    positions[8] = [dx_e, 2*dy_ne]
    positions[9] = [2*dx_e, 2*dy_ne]
    positions[10] = [3*dx_e, 2*dy_ne]
    
    return positions

def _build_staggered_11():
    """
    Build a staggered arrangement that might be more compact.
    """
    sqrt3 = np.sqrt(3)
    dx_e = 3.0
    dx_ne = 1.5
    dy_ne = 2.0 * sqrt3
    
    positions = np.zeros((11, 2))
    positions[0] = [0.0, 0.0]
    positions[1] = [dx_e, 0.0]
    positions[2] = [-dx_e, 0.0]
    positions[3] = [dx_ne, dy_ne]
    positions[4] = [-dx_ne, dy_ne]
    positions[5] = [dx_ne, -dy_ne]
    positions[6] = [-dx_ne, -dy_ne]
    positions[7] = [0.0, 2*dy_ne]
    positions[8] = [0.0, -2*dy_ne]
    positions[9] = [2*dx_e, 0.0]
    positions[10] = [-2*dx_e, 0.0]
    
    return positions

def _optimize_rigid_cluster(positions_base, rng, time_budget):
    """
    Optimize a rigid cluster by finding the best global rotation and translation
    to minimize the outer hexagon side length.
    The cluster shape is fixed; we only rotate and translate the whole thing.
    """
    best_side = float('inf')
    best_rotation = 0.0
    best_translation = np.array([0.0, 0.0])
    
    center = np.mean(positions_base, axis=0)
    centered = positions_base - center
    
    start_t = time.time()
    
    # Coarse search over rotation
    for rot_deg in np.arange(0, 60, 1.0):
        if time.time() > start_t + time_budget:
            break
        rot_rad = np.radians(rot_deg)
        cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        rotated = centered @ rot_matrix.T
        side = _compute_outer_side_fast(np.column_stack([rotated, np.zeros(11)]), 0.0)
        if side < best_side:
            best_side = side
            best_rotation = rot_deg
            best_translation = np.array([0.0, 0.0])
    
    # Fine-tune rotation around best
    for rot_deg in np.arange(best_rotation - 1.0, best_rotation + 1.0, 0.1):
        if time.time() > start_t + time_budget:
            break
        rot_rad = np.radians(rot_deg)
        cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        rotated = centered @ rot_matrix.T
        side = _compute_outer_side_fast(np.column_stack([rotated, np.zeros(11)]), 0.0)
        if side < best_side:
            best_side = side
            best_rotation = rot_deg
            best_translation = np.array([0.0, 0.0])
    
    # Now optimize translation for the best rotation
    rot_rad = np.radians(best_rotation)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    rotated = centered @ rot_matrix.T
    
    # Optimize translation using a simple grid search
    tx_range = np.arange(-1.0, 1.0, 0.1)
    ty_range = np.arange(-1.0, 1.0, 0.1)
    for tx in tx_range:
        if time.time() > start_t + time_budget:
            break
        for ty in ty_range:
            t = np.array([tx, ty])
            translated = rotated + t
            side = _compute_outer_side_fast(np.column_stack([translated, np.zeros(11)]), 0.0)
            if side < best_side:
                best_side = side
                best_translation = t.copy()
    
    # Fine-tune translation
    for tx in np.arange(best_translation[0] - 0.1, best_translation[0] + 0.1, 0.02):
        if time.time() > start_t + time_budget:
            break
        for ty in np.arange(best_translation[1] - 0.1, best_translation[1] + 0.1, 0.02):
            t = np.array([tx, ty])
            translated = rotated + t
            side = _compute_outer_side_fast(np.column_stack([translated, np.zeros(11)]), 0.0)
            if side < best_side:
                best_side = side
                best_translation = t.copy()
    
    # Apply the optimal transformation
    final_positions = (centered @ rot_matrix.T) + best_translation
    
    # Now also optimize the outer hexagon angle
    best_outer_angle = 0.0
    for angle in np.arange(0, 30, 0.5):
        side = _compute_outer_side_fast(np.column_stack([final_positions, np.zeros(11)]), angle)
        if side < best_side:
            best_side = side
            best_outer_angle = angle
    
    for angle in np.arange(best_outer_angle - 0.5, best_outer_angle + 0.5, 0.05):
        side = _compute_outer_side_fast(np.column_stack([final_positions, np.zeros(11)]), angle)
        if side < best_side:
            best_side = side
            best_outer_angle = angle
    
    # Build the final config
    cfg = np.zeros((11, 3))
    cfg[:, :2] = final_positions
    cfg[:, 2] = 0.0  # All inner hexagons have the same orientation
    
    return cfg, best_side, best_outer_angle

def _local_search(cfg, angle, rng, time_budget, start_step=0.15):
    """Local search to minimize outer side length with simulated annealing."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side_fast(best_cfg, angle)
    current_cfg = cfg.copy()
    current_side = best_side
    step_size = start_step
    temperature = 0.05
    
    start_t = time.time()
    iterations = 0
    while time.time() - start_t < time_budget:
        iterations += 1
        new_cfg = current_cfg.copy()
        mask = rng.random(11) < 0.3
        n_mask = np.sum(mask)
        if n_mask == 0:
            mask[rng.integers(0, 11)] = True
            n_mask = 1
        new_cfg[mask, :2] += rng.normal(0, step_size, size=(n_mask, 2))
        
        if _is_valid(new_cfg):
            s = _compute_outer_side_fast(new_cfg, angle)
            delta = s - current_side
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-8)):
                current_cfg = new_cfg.copy()
                current_side = s
                if s < best_side - 1e-10:
                    best_side = s
                    best_cfg = new_cfg.copy()
                    step_size = max(0.005, step_size * 0.98)
                temperature *= 0.995
            else:
                step_size = min(0.5, step_size * 1.01)
        else:
            step_size = min(0.5, step_size * 1.03)
    
    return best_cfg, best_side

def _refine_single(cfg, angle, rng, time_budget):
    """Fine-grained refinement of individual hexagon positions."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side_fast(best_cfg, angle)
    step_size = 0.02
    
    start_t = time.time()
    while time.time() - start_t < time_budget:
        idx = rng.integers(0, 11)
        new_cfg = best_cfg.copy()
        new_cfg[idx, :2] += rng.normal(0, step_size, size=2)
        
        if _is_valid(new_cfg):
            s = _compute_outer_side_fast(new_cfg, angle)
            if s < best_side - 1e-10:
                best_side = s
                best_cfg = new_cfg.copy()
                step_size = max(0.0005, step_size * 0.95)
            else:
                step_size = min(0.05, step_size * 1.02)
        else:
            step_size = min(0.05, step_size * 1.04)
    
    return best_cfg, best_side

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 615
    
    best_side = float('inf')
    best_cfg = None
    best_angle = 0.0

    rng = np.random.default_rng(12345)

    # Phase 1: Try multiple initial cluster topologies
    cluster_builders = [_build_hex_cluster_11, _build_compact_cluster_11, _build_staggered_11]
    
    for builder in cluster_builders:
        if time.time() > deadline - 180:
            break
        cluster_positions = builder()
        cfg_init = np.column_stack([cluster_positions, np.zeros(11)])
        
        if _is_valid(cfg_init):
            t_budget = min(90, (deadline - time.time()) * 0.25)
            if t_budget > 20:
                cfg_rigid, side_rigid, angle_rigid = _optimize_rigid_cluster(
                    cluster_positions, rng, time_budget=t_budget
                )
                if side_rigid < best_side:
                    best_side = side_rigid
                    best_cfg = cfg_rigid
                    best_angle = angle_rigid

    # Phase 2: Free local search from the best rigid cluster result
    # Increased time budget because _is_valid is now much faster
    if best_cfg is not None and time.time() < deadline - 120:
        t_budget = min((deadline - time.time()) * 0.4, 300)
        if t_budget > 30:
            best_cfg, best_side = _local_search(best_cfg, best_angle, rng, t_budget)

    # Phase 3: Re-optimize outer angle
    if best_cfg is not None and time.time() < deadline - 60:
        best_a_new = best_angle
        best_s_new = best_side
        for angle in np.arange(best_angle - 1.5, best_angle + 1.5, 0.05):
            s = _compute_outer_side_fast(best_cfg, angle)
            if s < best_s_new:
                best_s_new = s
                best_a_new = angle
        if best_s_new < best_side:
            best_side = best_s_new
            best_angle = best_a_new

    # Phase 4: Fine refinement
    if best_cfg is not None and time.time() < deadline - 40:
        t_budget = min((deadline - time.time()) * 0.4, 150)
        if t_budget > 20:
            best_cfg, best_side = _refine_single(best_cfg, best_angle, rng, t_budget)

    # Phase 5: Additional local search rounds with different seeds
    if best_cfg is not None and time.time() < deadline - 30:
        for seed in [42, 123, 999]:
            if time.time() > deadline - 20:
                break
            rng2 = np.random.default_rng(seed)
            t_budget = min(60, (deadline - time.time()) * 0.3)
            if t_budget > 10:
                cfg_candidate = best_cfg.copy()
                # Small perturbation to escape local optimum
                cfg_candidate[rng2.integers(0, 11), :2] += rng2.normal(0, 0.05, size=2)
                if _is_valid(cfg_candidate):
                    cfg_new, side_new = _local_search(cfg_candidate, best_angle, rng2, t_budget)
                    if side_new < best_side - 1e-8:
                        best_side = side_new
                        best_cfg = cfg_new

    # Final angle optimization
    if best_cfg is not None and time.time() < deadline - 10:
        for angle in np.arange(best_angle - 0.5, best_angle + 0.5, 0.02):
            s = _compute_outer_side_fast(best_cfg, angle)
            if s < best_side:
                best_side = s
                best_angle = angle

    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    # Verify containment and adjust if needed
    if best_cfg is not None:
        if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
            for _ in range(1000):
                if _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
                    break
                best_side *= 1.0002

    if best_cfg is not None:
        if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
            required = _compute_outer_side_fast(best_cfg, best_angle) * 1.0005
            best_side = max(best_side, required)

    return best_cfg, outer_hex_data, float(best_side)


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


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


def _is_valid(cfg):
    """Full validity check for non-overlapping hexagons."""
    for i in range(11):
        for j in range(i+1, 11):
            v1 = _hex_vertices(cfg[i,0], cfg[i,1], 1.0, cfg[i,2])
            v2 = _hex_vertices(cfg[j,0], cfg[j,1], 1.0, cfg[j,2])
            if not _hex_separated(v1, v2):
                return False
    return True


def _compute_outer_side_fast(inner_data, outer_angle=0.0):
    """Compute the minimum outer hexagon side length needed to contain all inner hexagons."""
    a = np.radians(outer_angle)
    # Precompute all 11*6 = 66 vertices
    all_verts = np.array([_hex_vertices(r[0], r[1], 1.0, r[2]) for r in inner_data])
    
    # For a regular hexagon with side s and angle a, the inradius is s*sqrt(3)/2
    # The hexagon is defined by 6 half-planes. The normal directions are at angles
    # a + 30 + 60*i for i=0..5 (these are the directions perpendicular to each edge)
    # A point is inside if its projection onto each normal is <= s*sqrt(3)/2
    # So we need: max over all vertices of (dot(v, n_i)) for each i <= s*sqrt(3)/2
    # Thus s >= max_i (max_v dot(v, n_i)) * 2/sqrt(3)
    
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


def _local_search(cfg, angle, rng, time_budget, start_step=0.15):
    """Local search to minimize outer side length."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side_fast(best_cfg, angle)
    step_size = start_step
    
    start_t = time.time()
    it = 0
    while time.time() - start_t < time_budget:
        it += 1
        new_cfg = best_cfg.copy()
        mask = rng.random(11) < 0.3
        n_mask = np.sum(mask)
        if n_mask == 0:
            mask[rng.integers(0, 11)] = True
            n_mask = 1
        new_cfg[mask, :2] += rng.normal(0, step_size, size=(n_mask, 2))
        
        if _is_valid(new_cfg):
            s = _compute_outer_side_fast(new_cfg, angle)
            if s < best_side - 1e-10:
                best_side = s
                best_cfg = new_cfg.copy()
                step_size = max(0.005, step_size * 0.97)
            else:
                step_size = min(0.5, step_size * 1.02)
        else:
            step_size = min(0.5, step_size * 1.05)
    
    return best_cfg, best_side


def _refine_single(cfg, angle, rng, time_budget):
    """Fine-grained refinement of individual hexagon positions."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side_fast(best_cfg, angle)
    step_size = 0.03
    
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
                step_size = max(0.001, step_size * 0.93)
            else:
                step_size = min(0.1, step_size * 1.04)
        else:
            step_size = min(0.1, step_size * 1.06)
    
    return best_cfg, best_side


def _build_honeycomb_ring(n_inner, center_idx=0):
    """Build a honeycomb-like arrangement: 1 center + ring of 6 + additional."""
    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3
    dy = 1.5 * sqrt3
    
    config = np.zeros((n_inner, 3))
    # Center
    config[0] = [0.0, 0.0, 0.0]
    # First ring (6 hexagons)
    ring1_angles = [0, 60, 120, 180, 240, 300]
    for i, ang in enumerate(ring1_angles):
        rad = np.radians(ang)
        config[i+1, 0] = dx * np.cos(rad)
        config[i+1, 1] = dx * np.sin(rad)
        config[i+1, 2] = 0.0
    # Second ring (4 more to make 11)
    # Place at diagonal positions
    positions_2nd = [
        (1.5*dx, 1.5*dy),
        (-1.5*dx, 1.5*dy),
        (1.5*dx, -1.5*dy),
        (-1.5*dx, -1.5*dy),
    ]
    for i, (px, py) in enumerate(positions_2nd):
        idx = 7 + i
        config[idx, 0] = px
        config[idx, 1] = py
        config[idx, 2] = 0.0
    return config


def _build_tight_ring(n_inner=11):
    """Alternative: tighter ring arrangement with rotated outer hexagons."""
    sqrt3 = np.sqrt(3)
    # Center at origin
    config = np.zeros((n_inner, 3))
    config[0] = [0.0, 0.0, 0.0]
    # 6 around center at distance 2*sqrt(3)
    d = 2.0 * sqrt3
    for i in range(6):
        ang = np.radians(60 * i)
        config[i+1, 0] = d * np.cos(ang)
        config[i+1, 1] = d * np.sin(ang)
        config[i+1, 2] = 0.0
    # 4 more at further positions
    d2 = d * 2.0 / np.sqrt(3) * np.sqrt(3)  # = 2*d
    angles_2nd = [30, 150, 210, 330]
    for i, ang_deg in enumerate(angles_2nd):
        ang = np.radians(ang_deg)
        config[7+i, 0] = 3.0 * sqrt3 * np.cos(ang)
        config[7+i, 1] = 3.0 * sqrt3 * np.sin(ang)
        config[7+i, 2] = 0.0
    return config


def _build_staggered_rows(n_inner=11):
    """Staggered row arrangement."""
    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3
    dy = 1.5 * sqrt3
    
    config = np.zeros((n_inner, 3))
    # Row 0: 3 hexagons
    for i in range(3):
        config[i, 0] = (i - 1) * dx
        config[i, 1] = 0.0
        config[i, 2] = 0.0
    # Row 1: 4 hexagons (staggered)
    for i in range(4):
        config[3+i, 0] = (i - 1.5) * dx
        config[3+i, 1] = dy
        config[3+i, 2] = 0.0
    # Row 2: 4 hexagons (staggered)
    for i in range(4):
        config[7+i, 0] = (i - 1.5) * dx
        config[7+i, 1] = -dy
        config[7+i, 2] = 0.0
    return config


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 615  # Safety margin before 630s kill

    best_side = float('inf')
    best_cfg = None
    best_angle = 0.0

    rng = np.random.default_rng(12345)

    # Multiple initial configurations
    initial_configs = [
        _build_honeycomb_ring(11),
        _build_tight_ring(11),
        _build_staggered_rows(11),
    ]

    # Add a few random perturbations of the honeycomb
    for _ in range(3):
        cfg_perturbed = _build_honeycomb_ring(11).copy()
        cfg_perturbed[:, :2] += rng.normal(0, 0.5, size=(11, 2))
        if _is_valid(cfg_perturbed):
            initial_configs.append(cfg_perturbed)

    # Phase 1: Evaluate each initial config with angle search
    for cfg0 in initial_configs:
        if time.time() > deadline - 60:
            break
        
        # Check validity
        if not _is_valid(cfg0):
            continue
        
        # Search for best outer angle
        best_a_for_cfg = 0.0
        best_s_for_cfg = float('inf')
        for angle in np.arange(0, 30, 0.5):
            s = _compute_outer_side_fast(cfg0, angle)
            if s < best_s_for_cfg:
                best_s_for_cfg = s
                best_a_for_cfg = angle
        
        # Fine-tune angle
        for angle in np.arange(best_a_for_cfg - 0.5, best_a_for_cfg + 0.5, 0.05):
            s = _compute_outer_side_fast(cfg0, angle)
            if s < best_s_for_cfg:
                best_s_for_cfg = s
                best_a_for_cfg = angle
        
        if best_s_for_cfg < best_side:
            best_side = best_s_for_cfg
            best_cfg = cfg0.copy()
            best_angle = best_a_for_cfg

    # Phase 2: Local search from best config
    if best_cfg is not None and time.time() < deadline - 60:
        t_budget = min((deadline - time.time()) * 0.45, 180)
        if t_budget > 10:
            best_cfg, best_side = _local_search(best_cfg, best_angle, rng, t_budget)

    # Phase 3: Re-search angle after local search
    if best_cfg is not None and time.time() < deadline - 30:
        best_a_new = best_angle
        best_s_new = best_side
        for angle in np.arange(best_angle - 1.0, best_angle + 1.0, 0.05):
            s = _compute_outer_side_fast(best_cfg, angle)
            if s < best_s_new:
                best_s_new = s
                best_a_new = angle
        if best_s_new < best_side:
            best_side = best_s_new
            best_angle = best_a_new

    # Phase 4: Fine refinement
    if best_cfg is not None and time.time() < deadline - 20:
        t_budget = min((deadline - time.time()) * 0.55, 150)
        if t_budget > 10:
            best_cfg, best_side = _refine_single(best_cfg, best_angle, rng, t_budget)

    # Phase 5: More restarts if time permits
    if time.time() < deadline - 30:
        for restart in range(8):
            if time.time() > deadline - 15:
                break
            
            # Random perturbation of current best
            cfg_restart = best_cfg.copy()
            cfg_restart[:, :2] += rng.normal(0, 0.2, size=(11, 2))
            
            if not _is_valid(cfg_restart):
                continue
            
            # Quick angle search
            best_a_r = best_angle
            best_s_r = _compute_outer_side_fast(cfg_restart, best_angle)
            for angle in np.arange(best_angle - 2.0, best_angle + 2.0, 0.25):
                s = _compute_outer_side_fast(cfg_restart, angle)
                if s < best_s_r:
                    best_s_r = s
                    best_a_r = angle
            
            if best_s_r < best_side:
                t_budget = min((deadline - time.time()) * 0.3, 40)
                if t_budget > 5:
                    cfg_r, s_r = _local_search(cfg_restart, best_a_r, rng, t_budget)
                    if s_r < best_side:
                        best_side = s_r
                        best_cfg = cfg_r
                        best_angle = best_a_r

    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    # Verify containment and adjust if needed
    if best_cfg is not None:
        if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
            for _ in range(500):
                if _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
                    break
                best_side *= 1.0003

    # Final safety: if still not contained, bump side
    if best_cfg is not None:
        if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
            # Compute actual required side
            required = _compute_outer_side_fast(best_cfg, best_angle) * 1.001
            best_side = max(best_side, required)

    return best_cfg, outer_hex_data, float(best_side)


# EVOLVE-BLOCK-END

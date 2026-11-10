# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _all_vertices(cfg):
    """Compute all 11*6 = 66 vertices at once. Returns (11, 6, 2)."""
    cx = cfg[:, 0]
    cy = cfg[:, 1]
    a = np.radians(cfg[:, 2])
    angles = a[:, None] + np.radians(60 * np.arange(6))[None, :]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    vx = cx[:, None] + cos_a
    vy = cy[:, None] + sin_a
    return np.stack([vx, vy], axis=-1)  # (11, 6, 2)


def _edge_normals(verts):
    """Compute 6 edge normals per hexagon. verts shape (11, 6, 2) -> normals shape (11, 6, 2)."""
    edges = np.roll(verts, -1, axis=1) - verts  # (11, 6, 2)
    # Rotate 90 deg CCW: (x,y) -> (-y, x)
    normals = np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=-1)
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return normals / norms


def _separated_all(cfg):
    """Check all pairs of hexagons for separation. Returns True if all separated."""
    verts = _all_vertices(cfg)  # (11, 6, 2)
    n1 = _edge_normals(verts)   # (11, 6, 2)
    
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = verts[i]  # (6, 2)
            v2 = verts[j]
            
            # Check all 12 normals (6 from hex i, 6 from hex j)
            # For each normal, project both hexagons and check if intervals are apart
            # Use normals from hex i
            for k in range(6):
                n = n1[i, k]  # (2,)
                p1 = v1 @ n  # (6,)
                p2 = v2 @ n  # (6,)
                if np.max(p1) < np.min(p2) - 1e-6 or np.max(p2) < np.min(p1) - 1e-6:
                    break
            else:
                # Check normals from hex j
                found = False
                for k in range(6):
                    n = n1[j, k]
                    p1 = v1 @ n
                    p2 = v2 @ n
                    if np.max(p1) < np.min(p2) - 1e-6 or np.max(p2) < np.min(p1) - 1e-6:
                        found = True
                        break
                if not found:
                    return False
    return True


def _compute_outer_side(cfg, outer_angle=0.0):
    """Compute minimum outer hex side to contain all inner hexagons."""
    verts = _all_vertices(cfg)  # (11, 6, 2)
    a = np.radians(outer_angle)
    max_proj = 0.0
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = verts.reshape(-1, 2) @ n  # (66,)
        max_proj = max(max_proj, np.max(proj))
    return 2.0 * max_proj / np.sqrt(3)


def _check_containment(cfg, outer_center, outer_angle, side):
    """Check if all inner hexagons are contained in the outer hexagon."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    inner_verts = _all_vertices(cfg)  # (11, 6, 2)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        # For each inner vertex, check cross(e, p - p1) >= -1e-6
        for h in range(11):
            for v_idx in range(6):
                p = inner_verts[h, v_idx]
                cross_val = e[0] * (p[1] - p1[1]) - e[1] * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False
    return True


def _local_search(cfg, angle, rng, time_budget, start_step=0.3):
    """Simulated-annealing style local search to minimize outer side length."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side(best_cfg, angle)
    step_size = start_step
    current_side = best_side
    current_cfg = best_cfg.copy()
    
    start_t = time.time()
    it = 0
    while time.time() - start_t < time_budget:
        it += 1
        new_cfg = current_cfg.copy()
        # Perturb a random subset
        n_perturb = rng.integers(1, 5)
        indices = rng.choice(11, size=n_perturb, replace=False)
        new_cfg[indices, :2] += rng.normal(0, step_size, size=(n_perturb, 2))
        
        if _separated_all(new_cfg):
            s = _compute_outer_side(new_cfg, angle)
            # Accept if better, or with probability exp(-delta/T)
            if s < current_side:
                current_cfg = new_cfg
                current_side = s
                if s < best_side - 1e-10:
                    best_side = s
                    best_cfg = new_cfg.copy()
                step_size = max(0.01, step_size * 0.98)
            else:
                # Geometric cooling
                progress = (time.time() - start_t) / time_budget
                T = max(0.01, (1 - progress) * 0.5)
                if rng.random() < np.exp(-(s - current_side) / T):
                    current_cfg = new_cfg
                    current_side = s
                    step_size = max(0.01, step_size * 0.99)
                else:
                    step_size = min(1.0, step_size * 1.03)
        else:
            step_size = min(1.0, step_size * 1.05)
    
    return best_cfg, best_side


def _refine(cfg, angle, rng, time_budget):
    """Fine-grained coordinate descent refinement."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side(best_cfg, angle)
    step_size = 0.05
    
    start_t = time.time()
    while time.time() - start_t < time_budget:
        idx = rng.integers(0, 11)
        new_cfg = best_cfg.copy()
        new_cfg[idx, :2] += rng.normal(0, step_size, size=2)
        
        if _separated_all(new_cfg):
            s = _compute_outer_side(new_cfg, angle)
            if s < best_side - 1e-10:
                best_side = s
                best_cfg = new_cfg.copy()
                step_size = max(0.002, step_size * 0.92)
            else:
                step_size = min(0.15, step_size * 1.04)
        else:
            step_size = min(0.15, step_size * 1.06)
    
    return best_cfg, best_side


def _initial_config_hexgrid():
    """Honeycomb: 1 center + 6 ring + 4 more."""
    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3
    dy = 1.5 * sqrt3
    return np.array([
        [0.0, 0.0, 0.0],
        [dx, 0.0, 0.0],
        [0.5*dx, dy, 0.0],
        [-0.5*dx, dy, 0.0],
        [-dx, 0.0, 0.0],
        [-0.5*dx, -dy, 0.0],
        [0.5*dx, -dy, 0.0],
        [1.5*dx, 1.5*dy, 0.0],
        [-1.5*dx, 1.5*dy, 0.0],
        [1.5*dx, -1.5*dy, 0.0],
        [-1.5*dx, -1.5*dy, 0.0],
    ])


def _initial_config_ring(rng, n_ring1=6, n_ring2=4):
    """Ring-based initial config with random angles."""
    config = np.zeros((11, 3))
    # Center
    config[0] = [0.0, 0.0, 0.0]
    # First ring
    r1 = 2.0 * np.sqrt(3)
    for i in range(n_ring1):
        a = 2 * np.pi * i / n_ring1 + rng.normal(0, 0.1)
        config[1 + i] = [r1 * np.cos(a), r1 * np.sin(a), 0.0]
    # Second ring
    r2 = 3.5 * np.sqrt(3)
    for i in range(n_ring2):
        a = 2 * np.pi * i / n_ring2 + np.pi / n_ring2 + rng.normal(0, 0.1)
        config[1 + n_ring1 + i] = [r2 * np.cos(a), r2 * np.sin(a), 0.0]
    return config


def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 575  # Safety margin (evaluator kills at 630s)
    
    rng = np.random.default_rng(42)
    best_side = float('inf')
    best_cfg = None
    best_angle = 0.0
    
    # Generate diverse initial configurations
    inits = [_initial_config_hexgrid()]
    for _ in range(4):
        inits.append(_initial_config_ring(rng))
    # Perturbed hexgrid
    base = _initial_config_hexgrid()
    for _ in range(3):
        p = base.copy()
        p[:, :2] += rng.normal(0, 0.5, size=(11, 2))
        inits.append(p)
    
    for init_idx, config in enumerate(inits):
        if time.time() > deadline - 60:
            break
        
        # Skip if not valid
        if not _separated_all(config):
            continue
        
        # Find best outer angle
        best_a = 0.0
        best_s = float('inf')
        for angle in np.arange(0, 30, 0.5):
            s = _compute_outer_side(config, angle)
            if s < best_s:
                best_s = s
                best_a = angle
        
        # Fine-tune angle
        for angle in np.arange(best_a - 0.5, best_a + 0.5, 0.05):
            s = _compute_outer_side(config, angle)
            if s < best_s:
                best_s = s
                best_a = angle
        
        if best_s < best_side:
            best_side = best_s
            best_cfg = config.copy()
            best_angle = best_a
        
        # Local search if time permits
        remaining = deadline - time.time()
        if remaining > 30 and best_s < best_side + 0.1:
            t_budget = min(remaining * 0.4, 120)
            cfg2, s2 = _local_search(config, best_a, rng, t_budget)
            if s2 < best_side:
                best_side = s2
                best_cfg = cfg2
                best_angle = best_a
            
            # Refine
            remaining = deadline - time.time()
            if remaining > 20:
                t_budget = min(remaining * 0.5, 60)
                cfg3, s3 = _refine(best_cfg, best_angle, rng, t_budget)
                if s3 < best_side:
                    best_side = s3
                    best_cfg = cfg3
    
    # Final refinement on best config
    remaining = deadline - time.time()
    if remaining > 15 and best_cfg is not None:
        t_budget = min(remaining * 0.7, 60)
        best_cfg, best_side = _refine(best_cfg, best_angle, rng, t_budget)
    
    # One more angle sweep on final config
    for angle in np.arange(best_angle - 2.0, best_angle + 2.0, 0.1):
        s = _compute_outer_side(best_cfg, angle)
        if s < best_side:
            best_side = s
            best_angle = angle
    
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    # Verify containment and adjust if needed
    if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
        for _ in range(500):
            if _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
                break
            best_side *= 1.0005
    
    return best_cfg, outer_hex_data, best_side


# EVOLVE-BLOCK-END

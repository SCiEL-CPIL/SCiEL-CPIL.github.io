# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_separated(v1, v2, tol=1e-6):
    """Check if two hexagons (vertex arrays) are separated."""
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


def _compute_outer_side(inner_data, outer_angle=0.0):
    """Compute minimum outer hex side to contain all inner hexagons."""
    a = np.radians(outer_angle)
    max_proj = 0.0
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        n = np.array([np.cos(theta), np.sin(theta)])
        for hex_row in inner_data:
            v = _hex_vertices(hex_row[0], hex_row[1], 1.0, hex_row[2])
            proj = v @ n
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


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    sqrt3 = np.sqrt(3)
    dx = 2.0 * sqrt3  # Horizontal center distance for touching flat-top hexagons
    dy = 1.5 * sqrt3  # Vertical offset for staggered rows

    # Honeycomb: 1 center + 6 ring + 4 outer
    config = np.array([
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

    # Verify initial config is valid
    def is_valid(cfg):
        for i in range(11):
            for j in range(i+1, 11):
                v1 = _hex_vertices(cfg[i,0], cfg[i,1], 1.0, cfg[i,2])
                v2 = _hex_vertices(cfg[j,0], cfg[j,1], 1.0, cfg[j,2])
                if not _hex_separated(v1, v2):
                    return False
        return True

    best_side = _compute_outer_side(config, 0.0)
    best_cfg = config.copy()

    # Try different outer angles
    for angle in np.arange(0, 30, 0.5):
        s = _compute_outer_side(config, angle)
        if s < best_side:
            best_side = s

    # Local search with random perturbations
    rng = np.random.default_rng(42)
    step_size = 0.2
    for _ in range(5000):
        if time.time() > deadline:
            break
        new_cfg = best_cfg.copy()
        mask = rng.random(11) < 0.3
        new_cfg[mask, :2] += rng.normal(0, step_size, size=(np.sum(mask), 2))
        
        if is_valid(new_cfg):
            s = _compute_outer_side(new_cfg, 0.0)
            if s < best_side - 1e-8:
                best_side = s
                best_cfg = new_cfg.copy()
                step_size = max(0.05, step_size * 0.95)
            else:
                step_size = min(0.5, step_size * 1.05)

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    # Verify containment and adjust if needed
    if not _check_containment(best_cfg, np.array([0.0, 0.0]), 0.0, best_side):
        # Increase side length until containment is satisfied
        best_side *= 1.01
        for _ in range(100):
            if _check_containment(best_cfg, np.array([0.0, 0.0]), 0.0, best_side):
                break
            best_side *= 1.01
    
    return best_cfg, outer_hex_data, best_side


# EVOLVE-BLOCK-END

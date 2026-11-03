"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.linspace(0, 2 * np.pi, 7)[:-1]
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])


def _compute_min_side(inner, center, angle_deg):
    """Compute minimum outer hex side to contain all inner hexagons."""
    cx, cy = center
    edge_angles = np.radians(angle_deg) + np.linspace(0, 60, 6) * np.pi / 180
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])

    max_proj = 0.0
    for ix, iy, iangle in inner:
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        dx = verts[:, 0] - cx
        dy = verts[:, 1] - cy
        projs = dx * normals[:, 0] + dy * normals[:, 1]
        max_proj = max(max_proj, np.max(projs))

    inradius_factor = np.sqrt(3) / 2
    return max_proj / inradius_factor


def _objective(params):
    """Objective: minimize outer hex side length."""
    inner_params = params[:36].reshape(12, 3)
    cx = params[36]
    cy = params[37]
    outer_angle = params[38]
    return _compute_min_side(inner_params.tolist(), (cx, cy), outer_angle)


def construct():
    # Initial configuration: hexagonal grid
    spacing_h = np.sqrt(3)
    spacing_v = 1.5

    positions = []
    for row in range(3):
        for col in range(4):
            x = col * spacing_h + (row % 2) * spacing_h / 2
            y = row * spacing_v
            positions.append([x, y, 0.0])

    # Center
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_init = (min(xs) + max(xs)) / 2
    cy_init = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx_init
        p[1] -= cy_init

    init_params = []
    for p in positions:
        init_params.extend([p[0], p[1], p[2]])
    init_params.extend([0.0, 0.0, 0.0])
    init_params = np.array(init_params)

    best_side = _compute_min_side(positions, (0.0, 0.0), 0.0)
    best_params = init_params.copy()

    for trial in range(3):
        if trial == 0:
            p0 = init_params.copy()
        else:
            p0 = init_params + np.random.randn(39) * 0.3

        try:
            result = minimize(_objective, p0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x
        except Exception:
            pass

    best_inner = best_params[:36].reshape(12, 3)
    best_cx = best_params[36]
    best_cy = best_params[37]
    best_angle = best_params[38]

    return best_inner.tolist(), (best_cx, best_cy), best_side, best_angle
# EVOLVE-BLOCK-END

"""Packing 12 unit hexagons in a regular hexagon - grid layout with optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _min_enclosing_hexagon(inner_hexagons):
    """
    Find minimum side length of a regular hexagon (with variable center and rotation)
    that contains all given inner hexagons.
    Returns: (center_x, center_y, side_length, rotation_deg)
    """
    all_verts = []
    for (cx, cy, ang) in inner_hexagons:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    def objective(params):
        hx, hy, theta_deg = params
        theta = np.radians(theta_deg)
        phi = np.radians(30 + 60 * np.arange(6)) + theta
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        dx = all_verts[:, 0] - hx
        dy = all_verts[:, 1] - hy
        proj = dx[:, np.newaxis] * cos_phi[np.newaxis, :] + dy[:, np.newaxis] * sin_phi[np.newaxis, :]
        max_proj = np.max(proj, axis=1)
        max_over_all = np.max(max_proj)
        if max_over_all <= 0:
            return 0.0
        return (2.0 / np.sqrt(3.0)) * max_over_all
    
    best_side = np.inf
    best_params = None
    hx0 = np.mean(all_verts[:, 0])
    hy0 = np.mean(all_verts[:, 1])
    
    for init_theta in [0, 15, 30, 45, 60, 10, 20, 40, 50]:
        result = minimize(objective, [hx0, hy0, init_theta], method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x
    
    if best_params is None:
        return (0.0, 0.0, 40.0, 0.0)
    
    hx, hy, theta_deg = best_params
    return (hx, hy, best_side, theta_deg)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    dx = np.sqrt(3)
    dy = np.sqrt(3)
    
    inner = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * dx
            y = (row - 1) * dy
            inner.append([x, y, 0.0])
    
    center, side, angle = _min_enclosing_hexagon(inner)
    
    return inner, (center[0], center[1]), side, angle
# EVOLVE-BLOCK-END

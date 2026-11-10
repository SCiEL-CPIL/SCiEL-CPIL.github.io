"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def min_enclosing_hexagon_side(inner_hexes, outer_cx, outer_cy, outer_angle):
    """Find minimum side length of regular hexagon with given center and angle enclosing all inner hexagons."""
    angle = np.radians(outer_angle)
    max_proj = 0.0
    for (ix, iy, iangle) in inner_hexes:
        verts = hexagon_vertices(ix, iy, 1.0, iangle)
        for v in verts:
            dx = v[0] - outer_cx
            dy = v[1] - outer_cy
            rx = dx * np.cos(angle) + dy * np.sin(angle)
            ry = -dx * np.sin(angle) + dy * np.cos(angle)
            for i in range(6):
                theta = (i + 0.5) * np.pi / 3
                proj = rx * np.cos(theta) + ry * np.sin(theta)
                if proj > max_proj:
                    max_proj = proj
    return 2.0 * max_proj / np.sqrt(3)

def evaluate_configuration(inner_hexes, outer_cx, outer_cy, outer_angle):
    """Return the minimum outer side length needed."""
    return min_enclosing_hexagon_side(inner_hexes, outer_cx, outer_cy, outer_angle)

def construct():
    sqrt3 = np.sqrt(3)
    
    # Axial coordinates for hexagonal grid
    # Center + ring 1 (6) + ring 2 partial (4) = 11
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    def axial_to_cartesian(q, r, size=1.0):
        x = size * sqrt3 * (q + r / 2.0)
        y = size * 1.5 * r
        return x, y
    
    initial_positions = []
    for q, r in axial_coords:
        x, y = axial_to_cartesian(q, r)
        initial_positions.append([x, y, 0.0])
    
    # Find best outer hexagon for initial configuration
    best_side = float('inf')
    best_outer_cx, best_outer_cy, best_outer_angle = 0.0, 0.0, 0.0
    best_inner = [list(p) for p in initial_positions]
    
    for outer_angle in range(0, 180, 15):
        for outer_cx in np.linspace(-2, 2, 9):
            for outer_cy in np.linspace(-2, 2, 9):
                side = evaluate_configuration(best_inner, float(outer_cx), float(outer_cy), float(outer_angle))
                if side < best_side:
                    best_side = side
                    best_outer_cx, best_outer_cy, best_outer_angle = float(outer_cx), float(outer_cy), float(outer_angle)
    
    # Local optimization
    def objective(params):
        inner_params = params[:33].reshape(11, 3)
        outer_cx = params[33]
        outer_cy = params[34]
        outer_angle = params[35]
        inner = [[inner_params[i, 0], inner_params[i, 1], inner_params[i, 2]] for i in range(11)]
        return evaluate_configuration(inner, outer_cx, outer_cy, outer_angle)
    
    x0_flat = []
    for p in initial_positions:
        x0_flat.extend([p[0], p[1], p[2]])
    x0_flat.extend([best_outer_cx, best_outer_cy, best_outer_angle])
    x0 = np.array(x0_flat)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-10})
        if result.fun < best_side:
            best_side = result.fun
            res_params = result.x
            best_inner = []
            for i in range(11):
                best_inner.append([res_params[i*3], res_params[i*3+1], res_params[i*3+2]])
            best_outer_cx = res_params[33]
            best_outer_cy = res_params[34]
            best_outer_angle = res_params[35]
    except:
        pass
    
    return best_inner, (best_outer_cx, best_outer_cy), best_side, best_outer_angle
# EVOLVE-BLOCK-END

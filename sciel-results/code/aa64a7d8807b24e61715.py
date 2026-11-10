# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _min_enclosing_hex_side(points):
    """
    Given a set of 2D points, find the minimum side length of a regular hexagon
    (any orientation, centered at origin) that contains all points.
    """
    if len(points) == 0:
        return 0.0
    best_s = np.inf
    thetas = np.linspace(0, np.pi/3, 181)  # 0 to 60 degrees
    for theta in thetas:
        normals = np.column_stack([
            np.cos(theta + np.radians(30 + 60*k)),
            np.sin(theta + np.radians(30 + 60*k))
        ] for k in range(6))
        dots = points @ normals.T
        max_dot = np.max(dots, axis=1)
        s_candidate = np.max(max_dot) / (np.sqrt(3)/2)
        if s_candidate < best_s:
            best_s = s_candidate
    return best_s

def _compute_side_from_positions(positions):
    """Compute the minimum enclosing hexagon side given positions of unit hexagons (angle=0)."""
    hex_verts_unit = np.column_stack([np.cos(np.radians(60*np.arange(6))), np.sin(np.radians(60*np.arange(6)))])
    all_vertices = positions[:, np.newaxis, :] + hex_verts_unit[np.newaxis, :, :]
    all_vertices = all_vertices.reshape(-1, 2)
    return _min_enclosing_hex_side(all_vertices)

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 570
    
    s3 = np.sqrt(3)
    
    # Initial configuration: 7-hexagon cluster + 4 more
    pos7 = np.array([
        [0, 0],
        [s3, 0],
        [s3/2, 1.5],
        [-s3/2, 1.5],
        [-s3, 0],
        [-s3/2, -1.5],
        [s3/2, -1.5],
    ])
    
    extra_angles = [30, 150, 210, 330]
    extra_pos = np.array([[3*np.cos(np.radians(a)), 3*np.sin(np.radians(a))] for a in extra_angles])
    
    positions = np.vstack([pos7, extra_pos])
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        pos = params.reshape(11, 2)
        return _compute_side_from_positions(pos)
    
    x0 = positions.flatten()
    
    best_result = None
    best_val = objective(x0)
    best_result = x0.copy()
    
    np.random.seed(42)
    
    for restart in range(15):
        if time.time() > deadline - 90:
            break
        
        if restart == 0:
            x_start = x0
        else:
            x_start = x0 + np.random.randn(22) * 0.3
        
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            
            if result.fun < best_val:
                best_val = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
    
    if best_result is not None and time.time() < deadline - 60:
        try:
            result2 = minimize(objective, best_result, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12})
            if result2.fun < best_val:
                best_val = result2.fun
                best_result = result2.x.copy()
        except Exception:
            pass
    
    best_positions = best_result.reshape(11, 2)
    side = float(best_val)
    
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = best_positions[:, 0]
    inner_hex_data[:, 1] = best_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, side

# EVOLVE-BLOCK-END

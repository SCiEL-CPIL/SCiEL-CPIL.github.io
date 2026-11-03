"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    theta = np.radians(angle_deg)
    vertices = []
    for k in range(6):
        phi = theta + np.pi * k / 3
        x = cx + s * np.cos(phi)
        y = cy + s * np.sin(phi)
        vertices.append((x, y))
    return np.array(vertices)

def min_enclosing_hex_side(points, outer_center, outer_angle):
    """
    Find minimal side length of a regular hexagon with given center and orientation
    that contains all given points.
    """
    cx, cy = outer_center
    theta = np.radians(outer_angle)
    normals = []
    for k in range(6):
        phi = theta + np.pi * k / 3
        normals.append((np.cos(phi), np.sin(phi)))
    normals = np.array(normals)
    
    diffs = points - np.array([cx, cy])
    dots = diffs @ normals.T
    max_dots = np.max(dots, axis=0)
    max_dot = np.max(max_dots)
    if max_dot <= 0:
        return 0.0
    s = max_dot / (np.sqrt(3) / 2)
    return s

def cost(params):
    inner_params = params[:36].reshape(12, 3)
    cx, cy, outer_angle = params[36], params[37], params[38]
    
    all_vertices = []
    for i in range(12):
        x, y, angle = inner_params[i]
        verts = hex_vertices(x, y, 1.0, angle)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    s = min_enclosing_hex_side(all_vertices, (cx, cy), outer_angle)
    return s

def construct():
    positions = []
    # Center
    positions.append((0.0, 0.0))
    # First ring: 6 hexagons
    for k in range(6):
        angle = np.pi / 3 * k
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        positions.append((x, y))
    # Second ring: 5 more hexagons
    for k in range(5):
        angle = np.pi / 3 * k + np.pi / 6
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        positions.append((x, y))
    
    cx, cy = 0.0, 0.0
    outer_angle = 0.0
    
    params = []
    for x, y in positions:
        params.extend([x, y, 0.0])
    params.extend([cx, cy, outer_angle])
    params = np.array(params)
    
    result = minimize(cost, params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
    
    optimal_params = result.x
    inner_params = optimal_params[:36].reshape(12, 3)
    cx, cy, outer_angle = optimal_params[36], optimal_params[37], optimal_params[38]
    side = min_enclosing_hex_side(
        np.vstack([hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2]) for i in range(12)]),
        (cx, cy), outer_angle
    )
    
    inner = []
    for i in range(12):
        x, y, angle = inner_params[i]
        inner.append([float(x), float(y), float(angle)])
    
    return inner, (float(cx), float(cy)), float(side), float(outer_angle)

import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def _get_hex_vertices(x, y, angle_deg):
    """Return 6 vertices of a unit hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3
        vx = x + np.cos(theta)
        vy = y + np.sin(theta)
        vertices.append((vx, vy))
    return np.array(vertices)

def _min_bounding_hex_side(points, alpha_deg):
    """
    Calculate the minimum side length S of a regular hexagon 
    with rotation alpha_deg and center at (0,0) that contains all points.
    """
    alpha_rad = np.deg2rad(alpha_deg)
    max_proj = 0.0
    for k in range(6):
        theta = alpha_rad + k * np.pi / 3
        nx = np.cos(theta)
        ny = np.sin(theta)
        projs = points[:, 0] * nx + points[:, 1] * ny
        max_proj = max(max_proj, np.max(projs))
    return 2.0 * max_proj / np.sqrt(3.0)

def _objective(params):
    if time.time() - START_TIME > TIME_LIMIT:
        return 1e10
    
    positions = params[:2*N].reshape(N, 2)
    alpha = params[2*N]
    
    all_vertices = []
    for i in range(N):
        x, y = positions[i]
        verts = _get_hex_vertices(x, y, 0.0)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    S = _min_bounding_hex_side(all_vertices, alpha)
    
    penalty = 0.0
    min_dist = np.sqrt(3.0) - 1e-5
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    return S + 100.0 * penalty

def _initial_guess():
    positions = np.zeros((N, 2))
    positions[0] = [0.0, 0.0]
    for k in range(6):
        theta = k * np.pi / 3
        positions[k+1] = [np.sqrt(3.0) * np.cos(theta), np.sqrt(3.0) * np.sin(theta)]
    for k in range(4):
        theta = k * np.pi / 2
        positions[k+7] = [2.0 * np.sqrt(3.0) * np.cos(theta), 2.0 * np.sqrt(3.0) * np.sin(theta)]
    positions += np.random.uniform(-0.1, 0.1, positions.shape)
    alpha = 0.0
    return np.concatenate([positions.flatten(), [alpha]])

# EVOLVE-BLOCK-START
def construct():
    best_score = float('inf')
    best_params = None
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        x0 = _initial_guess()
        if r > 0:
            x0[:2*N] += np.random.uniform(-0.5, 0.5, 2*N)
            
        try:
            result = minimize(_objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass

    if best_params is None:
        best_params = _initial_guess()
        best_score = _objective(best_params)
        
    positions = best_params[:2*N].reshape(N, 2)
    alpha = best_params[2*N]
    
    inner = []
    for i in range(N):
        x, y = positions[i]
        inner.append([float(x), float(y), 0.0])
        
    all_vertices = []
    for i in range(N):
        x, y = positions[i]
        verts = _get_hex_vertices(x, y, 0.0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    true_S = _min_bounding_hex_side(all_vertices, alpha)
    
    return inner, (0.0, 0.0), float(true_S), float(alpha)
# EVOLVE-BLOCK-END

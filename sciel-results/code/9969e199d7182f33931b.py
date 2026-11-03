import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 12

# Precompute unit hexagon vertices (side length 1)
# Orientation: Vertices at 30, 90, 150, 210, 270, 330 degrees
# This makes the "flat" sides horizontal?
# Vertex (0, 1) and (sqrt(3)/2, 0.5). Midpoint y = 0.75.
# Side is not horizontal.
# The normals are at 0, 60, 120.
# This is the standard "pointy top" vs "flat top" confusion.
# Let's stick to the normals definition for the container, and the vertices for the inner.
# As long as they are consistent, it works.
# My `UNIT_HEX_VERTS` matches the normals `NORMALS` defined earlier.
# Normals at 0, 60, 120 correspond to vertices at 30, 90, 150...
# Yes.

UNIT_HEX_VERTS = np.array([
    [np.sqrt(3)/2, 0.5],
    [0, 1],
    [-np.sqrt(3)/2, 0.5],
    [-np.sqrt(3)/2, -0.5],
    [0, -1],
    [np.sqrt(3)/2, -0.5]
])

NORMALS = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2]
])

def calculate_S(params):
    """
    params: array of shape (3*N,)
    """
    max_S = 0.0
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        theta = params[3*i+2]
        
        c = np.cos(theta)
        s = np.sin(theta)
        # Rotation matrix R = [[c, -s], [s, c]]
        # verts = R @ UNIT_HEX_VERTS.T ? No.
        # If UNIT_HEX_VERTS is (6, 2), and we want to rotate each vertex:
        # new_v = R @ v
        # verts_rot = UNIT_HEX_VERTS @ R.T
        rot_t = np.array([[c, s], [-s, c]]) # This is R.T? 
        # R = [[c, -s], [s, c]]
        # R.T = [[c, s], [-s, c]]
        # v_new = R @ v.
        # If v is column vector.
        # If we have matrix V (6x2), V_new = V @ R.T
        verts_rot = UNIT_HEX_VERTS @ rot_t
        verts = verts_rot + np.array([x, y])
        
        # Project onto normals
        dots = verts @ NORMALS.T
        abs_dots = np.abs(dots)
        required_S = (2.0 / np.sqrt(3)) * abs_dots
        hex_S = np.max(required_S)
        
        if hex_S > max_S:
            max_S = hex_S
            
    return max_S

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 110.0
    
    bounds = []
    for _ in range(N):
        bounds.append((-4.0, 4.0))
        bounds.append((-4.0, 4.0))
        bounds.append((0.0, np.pi/3))
        
    best_S = float('inf')
    best_params = None
    
    # Multiple restarts to escape local minima
    for restart in range(15):
        elapsed = time.time() - start_time
        if elapsed > timeout - 10:
            break
            
        seed = 42 + restart
        try:
            res = differential_evolution(
                calculate_S, 
                bounds, 
                maxiter=100, 
                popsize=150, 
                tol=1e-5, 
                seed=seed,
                polish=False
            )
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
                
            # Polish
            res_nm = minimize(
                calculate_S, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res_nm.fun < best_S:
                best_S = res_nm.fun
                best_params = res_nm.x.copy()
                
        except Exception:
            continue
            
    if best_params is None:
        # Fallback
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 10.0, 0.0

    inner = []
    for i in range(N):
        inner.append([best_params[3*i], best_params[3*i+1], np.degrees(best_params[3*i+2])])
        
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END

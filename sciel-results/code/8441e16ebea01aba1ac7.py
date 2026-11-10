# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    angles = np.radians(a_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _edge_normals(vertices):
    normals = np.zeros((6, 2))
    for i in range(6):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % 6]
        edge = v2 - v1
        n = np.array([edge[1], -edge[0]])
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-12:
            n /= n_norm
        normals[i] = n
    return normals

def _are_separated(h1, h2):
    v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if min1 - max2 > 1e-6 or min2 - max1 > 1e-6:
            return True
    return False

def _outer_side_for(inner_hexes, outer_center):
    shifted = inner_hexes.copy()
    shifted[:, 0] -= outer_center[0]
    shifted[:, 1] -= outer_center[1]
    sqrt3 = np.sqrt(3)
    normals = np.array([
        [sqrt3/2, 1/2], [0, 1], [-sqrt3/2, 1/2],
        [-sqrt3/2, -1/2], [0, -1], [sqrt3/2, -1/2]
    ])
    max_s = 0.0
    for hex_data in shifted:
        vertices = _hex_vertices(hex_data[0], hex_data[1], 1.0, hex_data[2])
        for v in vertices:
            projections = normals @ v
            required_s = (2.0 / sqrt3) * np.max(np.maximum(projections, 0))
            if required_s > max_s:
                max_s = required_s
    return max_s

def _objective(params, outer_center):
    inner_hexes = params.reshape(11, 3)
    for i in range(11):
        for j in range(i + 1, 11):
            if not _are_separated(inner_hexes[i], inner_hexes[j]):
                return 1e6
    return _outer_side_for(inner_hexes, outer_center)

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice: 7 hexes in center cluster, 4 more
    # For unit flat-top hexagons, center-to-center distance for touching = 2 * apothem = sqrt(3)
    # Directions: 0, 60, 120, 180, 240, 300
    d = sqrt3  # touching distance
    
    initial = np.array([
        [0, 0, 0],
        [d, 0, 0],
        [d/2, d*np.sqrt(3)/2, 0],
        [-d/2, d*np.sqrt(3)/2, 0],
        [-d, 0, 0],
        [-d/2, -d*np.sqrt(3)/2, 0],
        [d/2, -d*np.sqrt(3)/2, 0],
        [2*d, 0, 0],
        [d, d*np.sqrt(3), 0],
        [-d, d*np.sqrt(3), 0],
        [-2*d, 0, 0],
    ])
    
    # Add small gap to ensure separation
    initial[:, :2] *= 1.001
    
    outer_center = np.array([0.0, 0.0])
    
    best_params = initial.flatten()
    best_side = _outer_side_for(initial, outer_center)
    
    # Verify initial is valid
    valid = True
    for i in range(11):
        for j in range(i + 1, 11):
            if not _are_separated(initial[i], initial[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Increase spacing
        initial[:, :2] *= 1.01
        best_params = initial.flatten()
        best_side = _outer_side_for(initial, outer_center)
    
    # Optimize
    for attempt in range(3):
        if time.time() > deadline - 30:
            break
        
        if attempt == 0:
            x0 = best_params
        else:
            x0 = best_params + np.random.randn(33) * 0.05
        
        try:
            result = minimize(
                _objective, x0, args=(outer_center,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if result.fun < best_side:
                best_params = result.x
                best_side = result.fun
        except Exception:
            pass
    
    inner_hexes = best_params.reshape(11, 3)
    outer_hex_data = np.array([outer_center[0], outer_center[1], 0.0])
    outer_side = float(max(best_side, np.sqrt(11)))
    
    return inner_hexes, outer_hex_data, outer_side

# EVOLVE-BLOCK-END

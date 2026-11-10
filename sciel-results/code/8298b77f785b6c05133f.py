# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.deg2rad(60 * np.arange(6))
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack((x, y))

def _check_separation(hex1, hex2):
    # hex1, hex2 are arrays of shape (6, 2)
    # Check if any edge normal of hex1 or hex2 separates them
    # We check all 12 normals (6 from hex1, 6 from hex2)
    
    def get_normals(vertices):
        normals = np.zeros((6, 2))
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edge = p2 - p1
            # Normal is perpendicular to edge
            # Rotate edge by 90 deg
            n = np.array([-edge[1], edge[0]])
            normals[i] = n
        return normals

    n1 = get_normals(hex1)
    n2 = get_normals(hex2)
    all_normals = np.vstack([n1, n2])
    
    # Normalize normals
    norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
    all_normals = all_normals / norms
    
    for n in all_normals:
        # Project vertices of hex1 and hex2 onto normal n
        proj1 = hex1 @ n
        proj2 = hex2 @ n
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check if intervals [min1, max1] and [min2, max2] are apart
        # Gap > 1e-6
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
            
    return False

def _required_outer_size(vertices, outer_center, outer_angle_deg):
    # vertices: (N, 2)
    # outer_center: (2,)
    # outer_angle_deg: float
    # Returns minimum S such that all vertices are inside outer hex
    # Outer hex vertices are at distance S from center, at angles outer_angle + 60i
    # This means the hexagon is defined by its circumradius S.
    # The distance from center to each side (apothem) is S * sqrt(3) / 2.
    # The normals to the sides are at angles outer_angle + 30 + 60i
    
    a_rad = np.deg2rad(outer_angle_deg)
    normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # Shift vertices relative to outer center
    rel_vertices = vertices - outer_center
    
    # Project onto normals
    # proj[i, j] is projection of vertex i onto normal j
    proj = rel_vertices @ normals.T
    
    # For each vertex, find the max projection on any normal
    # The required S must satisfy: proj <= S * sqrt(3)/2 for all normals
    # So S >= 2/sqrt(3) * max(proj)
    # We take the max over all vertices and all normals
    
    max_proj = np.max(proj)
    if max_proj < 0:
        return 0.0 # Should not happen if centered properly
        
    S = (2.0 / np.sqrt(3.0)) * max_proj
    return S

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590.0 # 10s margin
    
    # Initial guesses for inner hexagon centers
    # Angles are fixed to 0 for simplicity in initial optimization
    # We will optimize positions and outer hex params
    
    guesses = []
    
    # Guess 1: 3-4-4 layout
    # Row 0 (y=0): 3 hexes
    # Row 1 (y=sqrt(3)): 4 hexes
    # Row 2 (y=2*sqrt(3)): 4 hexes
    h = np.sqrt(3)
    g1 = []
    # Row 0
    for i in range(3):
        g1.append([ (i-1)*2.01, 0.0 ])
    # Row 1
    for i in range(4):
        g1.append([ (i-1.5)*2.01, h*1.01 ])
    # Row 2
    for i in range(4):
        g1.append([ (i-1.5)*2.01, 2*h*1.01 ])
    guesses.append(np.array(g1))
    
    # Guess 2: 2-3-3-3 layout
    g2 = []
    for i in range(2):
        g2.append([ (i-0.5)*2.01, 0.0 ])
    for i in range(3):
        g2.append([ (i-1.0)*2.01, h*1.01 ])
    for i in range(3):
        g2.append([ (i-1.0)*2.01, 2*h*1.01 ])
    for i in range(3):
        g2.append([ (i-1.0)*2.01, 3*h*1.01 ])
    guesses.append(np.array(g2))
    
    # Guess 3: Flower (7) + 4
    # Center (0,0)
    # 6 neighbors
    # 4 more
    g3 = [[0,0]]
    for i in range(6):
        angle = i * np.pi / 3
        g3.append([ np.cos(angle)*1.733, np.sin(angle)*1.733 ])
    # Add 4 more in the gaps?
    # Just scatter them a bit
    g3.append([3.5, 0])
    g3.append([0, 3.5])
    g3.append([-3.5, 0])
    g3.append([0, -3.5])
    guesses.append(np.array(g3))
    
    best_result = None
    best_score = float('inf')
    
    for guess in guesses:
        if time.time() > deadline:
            break
            
        x0 = guess.flatten().tolist()
        # Add outer center (0,0) and angle (0)
        x0.extend([0.0, 0.0, 0.0])
        
        def objective(params):
            # 22 inner coords
            inner_centers = np.array(params[:22]).reshape(11, 2)
            outer_center = np.array(params[22:24])
            outer_angle = params[24]
            
            # Check time
            if time.time() > deadline:
                return 1e9
                
            # Construct inner hexagons
            inner_hexes = []
            for i in range(11):
                cx, cy = inner_centers[i]
                verts = _hex_vertices(cx, cy, 1.0, 0.0)
                inner_hexes.append(verts)
                
            # Check separation
            for i in range(11):
                for j in range(i+1, 11):
                    if not _check_separation(inner_hexes[i], inner_hexes[j]):
                        return 1e9 # Penalty for intersection
                        
            # Calculate required outer size
            all_vertices = np.vstack(inner_hexes)
            S = _required_outer_size(all_vertices, outer_center, outer_angle)
            
            return S
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                if res.fun < 1e8: # Valid solution
                    best_score = res.fun
                    best_result = res.x
                    
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to seed solution
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, 8.0

    # Extract best result
    inner_centers = best_result[:22].reshape(11, 2)
    outer_center = best_result[22:24]
    outer_angle = best_result[24]
    S = best_score
    
    # Construct return arrays
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [inner_centers[i][0], inner_centers[i][1], 0.0]
        
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
    
    return inner_hex_data, outer_hex_data, S

# EVOLVE-BLOCK-END

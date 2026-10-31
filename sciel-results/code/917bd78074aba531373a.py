import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    # Unit hexagon vertices (flat-top)
    # (1, 0), (0.5, sqrt(3)/2), ...
    verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    # Rotate
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    # Translate
    verts[:, 0] += cx
    verts[:, 1] += cy
    return verts

def point_in_hexagon(px, py, cx, cy, angle_deg):
    # Transform point to local coords
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    # Inverse rotation
    dx = px - cx
    dy = py - cy
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    # Check if (lx, ly) is in unit hexagon
    # Unit hexagon (flat-top) inequalities:
    # |x| <= 1
    # |y| <= sqrt(3)/2
    # |x| + |y|/sqrt(3) <= 1 ? No.
    # Vertices: (1,0), (0.5, 0.866), ...
    # Lines:
    # y = sqrt(3)(x-1) for x in [0.5, 1] -> sqrt(3)x - y - sqrt(3) = 0
    # y = -sqrt(3)(x-1) ...
    # Simpler: Check 6 half planes.
    # Normals: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Distance to side: sqrt(3)/2
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    dots = normals @ np.array([lx, ly])
    # The support function of unit hex in these directions is sqrt(3)/2
    return np.all(dots <= np.sqrt(3)/2 + 1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial positions
    # 1 center, 6 ring 1, 4 ring 2
    init_pos = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, np.sqrt(3), 0.0],
        [-1.0, np.sqrt(3), 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, -np.sqrt(3), 0.0],
        [1.0, -np.sqrt(3), 0.0],
        [4.0, 0.0, 0.0],
        [3.0, np.sqrt(3), 0.0],
        [2.0, 2*np.sqrt(3), 0.0],
        [0.0, 2*np.sqrt(3), 0.0]
    ]
    
    x0 = np.array(init_pos).flatten()
    
    def objective(params):
        # Check time
        if time.time() > deadline:
            return 1e9
        positions = params.reshape(N, 3)
        # Calculate min S
        max_proj = 0.0
        all_verts = []
        for i in range(N):
            cx, cy, ang = positions[i]
            verts = hex_vertices(cx, cy, ang)
            all_verts.append(verts)
        
        # Flatten all vertices
        all_verts = np.vstack(all_verts)
        
        # Normals for outer hexagon (flat-top)
        # The outer hexagon is fixed orientation 0.
        # Min S such that max(v . n) <= S * sqrt(3)/2 for all v, n
        normals = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        # Calculate max dot product for each normal
        # dots: (N*6, 6)
        dots = all_verts @ normals.T
        max_dots = np.max(dots, axis=0)
        # S = max(max_dots) / (sqrt(3)/2)
        S = np.max(max_dots) / (np.sqrt(3)/2)
        
        # Check overlaps
        # Penalty if any pair overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                cx1, cy1, a1 = positions[i]
                cx2, cy2, a2 = positions[j]
                # Quick center distance check
                if (cx1-cx2)**2 + (cy1-cy2)**2 < 4.0:
                    # Check vertices
                    verts1 = hex_vertices(cx1, cy1, a1)
                    verts2 = hex_vertices(cx2, cy2, a2)
                    # Check if any v1 in hex2
                    for v in verts1:
                        if point_in_hexagon(v[0], v[1], cx2, cy2, a2):
                            penalty += 100.0
                            break
                    else:
                        # Check if any v2 in hex1
                        for v in verts2:
                            if point_in_hexagon(v[0], v[1], cx1, cy1, a1):
                                penalty += 100.0
                                break
                    # If penalty is already high, break early?
                    if penalty > 1000:
                        break
            if penalty > 1000:
                break
        
        return S + penalty

    best_result = None
    best_score = 1e9
    
    # Try a few restarts if time permits
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            x0_curr = x0
        else:
            # Perturb
            x0_curr = x0 + np.random.normal(0, 0.5, size=x0.shape)
            
        result = minimize(objective, x0_curr, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
            
        if time.time() > deadline - 10:
            break
            
    if best_result is None:
        # Fallback to initial
        best_result = x0
        best_score = objective(x0)
        
    # Extract best
    positions = best_result.reshape(N, 3)
    # Recalculate exact S for reporting
    all_verts = np.vstack([hex_vertices(p[0], p[1], p[2]) for p in positions])
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    dots = all_verts @ normals.T
    max_dots = np.max(dots, axis=0)
    S = np.max(max_dots) / (np.sqrt(3)/2)
    
    inner = [[p[0], p[1], p[2]] for p in positions]
    return inner, (0.0, 0.0), S, 0.0


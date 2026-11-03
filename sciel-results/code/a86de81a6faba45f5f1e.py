import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def min_outer_side(inner_centers, inner_angles, ocx, ocy, oangle_deg):
    """
    Compute minimum outer hexagon side length to contain all inner hexagons.
    inner_centers: list of (x,y)
    inner_angles: list of angle_degrees
    ocx, ocy: outer hexagon center
    oangle_deg: outer hexagon rotation
    """
    angle_rad = np.radians(oangle_deg)
    # Edge normals for outer hexagon (pointing inward)
    # Hexagon edges are at angles 30, 90, 150, 210, 270, 330 from center
    edge_angles = np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([-np.sin(edge_angles), np.cos(edge_angles)])
    
    max_proj = 0.0
    for (icx, icy), iang in zip(inner_centers, inner_angles):
        iv = hex_vertices(icx, icy, 1.0, iang)
        for vx, vy in iv:
            dx = vx - ocx
            dy = vy - ocy
            projs = dx * normals[:, 0] + dy * normals[:, 1]
            mp = np.max(projs)
            if mp > max_proj:
                max_proj = mp
    
    if max_proj <= 0:
        return 1e-9
    return 2.0 / np.sqrt(3) * max_proj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Hex lattice: axial (q,r) to Cartesian
    # x = 3/2*q + sqrt(3)/2*r, y = sqrt(3)*r
    def axial_to_cart(q, r):
        return (1.5 * q + np.sqrt(3)/2 * r, np.sqrt(3) * r)
    
    # Generate several compact arrangements of 12 hexagons
    arrangements = []
    
    # Arrangement A: 7 (hex radius 1) + 5 contiguous from radius 2
    axA = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
           (2,0),(1,1),(0,2),(-1,2),(-2,2)]
    arrangements.append([axial_to_cart(q,r) for q,r in axA])
    
    # Arrangement B: 7 + 5 different
    axB = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
           (2,0),(1,1),(0,2),(-2,2),(-2,1)]
    arrangements.append([axial_to_cart(q,r) for q,r in axB])
    
    # Arrangement C: 4x3 patch
    axC = [(q,r) for q in range(-1,3) for r in range(-1,2)]
    arrangements.append([axial_to_cart(q,r) for q,r in axC])
    
    # Arrangement D: 2 rows of 6
    axD = [(q,r) for q in range(-3,3) for r in [0,1]]
    arrangements.append([axial_to_cart(q,r) for q,r in axD])
    
    # Arrangement E: rows 3,4,3,2
    axE = []
    for q in range(-1,2): axE.append((q,-1))
    for q in range(-2,2): axE.append((q,0))
    for q in range(-2,1): axE.append((q,1))
    for q in range(-1,1): axE.append((q,2))
    arrangements.append([axial_to_cart(q,r) for q,r in axE])
    
    # Arrangement F: rows 2,4,4,2
    axF = []
    for q in range(-1,1): axF.append((q,-1))
    for q in range(-2,2): axF.append((q,0))
    for q in range(-2,2): axF.append((q,1))
    for q in range(-1,1): axF.append((q,2))
    arrangements.append([axial_to_cart(q,r) for q,r in axF])
    
    best_score = float('inf')
    best_inner = None
    best_center = None
    best_angle = None
    
    np.random.seed(42)
    
    for arr in arrangements:
        if time.time() > deadline:
            break
        
        # Center the arrangement
        xs = np.array([p[0] for p in arr])
        ys = np.array([p[1] for p in arr])
        cx_arr = np.mean(xs)
        cy_arr = np.mean(ys)
        centered = [(x - cx_arr, y - cy_arr) for x, y in arr]
        angles = [0.0] * N
        
        def objective(params):
            ocx, ocy, oang = params
            return min_outer_side(centered, angles, ocx, ocy, oang)
        
        best_local = float('inf')
        best_params = None
        
        # Multi-start with Nelder-Mead
        n_starts = 30
        for i in range(n_starts):
            if time.time() > deadline:
                break
            
            if i == 0:
                init = np.array([0.0, 0.0, 0.0])
            elif i == 1:
                init = np.array([0.0, 0.0, 30.0])
            elif i == 2:
                init = np.array([0.0, 0.0, 15.0])
            else:
                init = np.array([
                    np.random.uniform(-1.5, 1.5),
                    np.random.uniform(-1.5, 1.5),
                    np.random.uniform(0, 60)
                ])
            
            try:
                res = minimize(objective, init, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
                if res.fun < best_local:
                    best_local = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        if best_params is not None and best_local < best_score:
            best_score = best_local
            best_inner = [(x, y, 0.0) for x, y in centered]
            best_center = (float(best_params[0]), float(best_params[1]))
            best_angle = float(best_params[2])
    
    if best_inner is None:
        # Fallback
        inner = [(4.0 * k, 0.0, 0.0) for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_score, best_angle

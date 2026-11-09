import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def point_in_hex(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) side s rotated angle_deg."""
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Regular hexagon with vertices at angles 0,60,...: 
    # Half-plane i: cos(theta_i)*lx + sin(theta_i)*ly <= s * sqrt(3)/2
    for i in range(6):
        theta = i * np.pi / 3
        if lx * np.cos(theta) + ly * np.sin(theta) > s * np.sqrt(3) / 2 + 1e-12:
            return False
    return True

def distance_point_to_segment(px, py, p1, p2):
    """Distance from point (px,py) to segment p1-p2."""
    vx = p2[0] - p1[0]
    vy = p2[1] - p1[1]
    wx = px - p1[0]
    wy = py - p1[1]
    
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return np.sqrt(wx*wx + wy*wy)
    
    c2 = vx*vx + vy*vy
    if c2 <= c1:
        return np.sqrt((px-p2[0])**2 + (py-p2[1])**2)
    
    b = c1 / c2
    px_proj = p1[0] + b * vx
    py_proj = p1[1] + b * vy
    return np.sqrt((px-px_proj)**2 + (py-py_proj)**2)

def penetration_depth(px, py, poly):
    """If point is inside poly, return min distance to boundary. Else 0."""
    # Check if inside
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    if not inside:
        return 0.0
    
    # If inside, find min distance to edges
    min_dist = float('inf')
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i+1)%n]
        d = distance_point_to_segment(px, py, p1, p2)
        if d < min_dist:
            min_dist = d
    return min_dist

def overlap_penalty(centers, angles):
    """Calculate smooth overlap penalty for all pairs."""
    penalty = 0.0
    verts_list = []
    for i in range(N):
        verts = hex_vertices(centers[i*2], centers[i*2+1], 1.0, angles[i])
        verts_list.append(verts)
        
    for i in range(N):
        for j in range(i+1, N):
            v1 = verts_list[i]
            v2 = verts_list[j]
            
            # Check vertices of v1 against v2
            for v in v1:
                d = penetration_depth(v[0], v[1], v2)
                if d > 0:
                    penalty += d * d
                    
            # Check vertices of v2 against v1
            for v in v2:
                d = penetration_depth(v[0], v[1], v1)
                if d > 0:
                    penalty += d * d
                    
    return penalty

def objective(params):
    """
    params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, Cx, Cy, Phi]
    Total 34 params.
    """
    centers = params[0:22].reshape(N, 2)
    angles = params[22:33]
    Cx = params[33]
    Cy = params[34]
    Phi = params[35] # Wait, 11*3 = 33. Cx, Cy, Phi is 3 more. Total 36?
    # Let's re-count.
    # 11 hexagons. Each has x, y, angle. 33 params.
    # Outer: Cx, Cy, Phi. 3 params.
    # Total 36.
    
    # Correct slicing:
    centers = params[0:33].reshape(N, 2) # No, x,y are pairs.
    # Better layout:
    # 0..21: x,y for 11 hexes (22 params)
    # 22..32: angles for 11 hexes (11 params)
    # 33: Cx
    # 34: Cy
    # 35: Phi
    
    centers = params[0:22].reshape(N, 2)
    angles = params[22:33]
    Cx = params[33]
    Cy = params[34]
    Phi = params[35]
    
    # Calculate Outer Side Length S
    a = np.radians(Phi)
    normals = []
    for i in range(6):
        theta = a + np.pi/6 + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(N):
        verts = hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        # Project vertices onto normals
        # verts: 6x2, normals: 6x2
        # We need max over all vertices and all normals
        # proj_k = dot(verts - C, n_k)
        dx = verts[:, 0] - Cx
        dy = verts[:, 1] - Cy
        for k in range(6):
            proj = dx * normals[k, 0] + dy * normals[k, 1]
            if proj.max() > max_proj:
                max_proj = proj.max()
            # Also check negative side? 
            # The hexagon is defined by max_k (n_k . (v-C)) <= S*sqrt(3)/2
            # Since normals cover all directions, max is sufficient.
            
    S = max_proj / (np.sqrt(3) / 2)
    
    # Calculate Overlap Penalty
    pen = overlap_penalty(centers, angles)
    
    return S + 1000.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    sqrt3 = np.sqrt(3)
    
    # Initial Configuration 1: Standard Lattice
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    
    centers_init = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers_init.append((x, y))
        
    angles_init = [0.0] * N
    
    # Flatten for optimizer
    x0 = []
    for i in range(N):
        x0.append(centers_init[i][0])
        x0.append(centers_init[i][1])
    x0.extend(angles_init)
    
    # Outer center estimate: mean of centers
    cx_mean = np.mean([c[0] for c in centers_init])
    cy_mean = np.mean([c[1] for c in centers_init])
    x0.extend([cx_mean, cy_mean, 0.0])
    
    x0 = np.array(x0)
    
    best_score = float('inf')
    best_params = x0.copy()
    
    # Multi-start
    starts = [x0]
    
    # Start 2: Perturbed
    np.random.seed(42)
    for _ in range(5):
        perturbed = x0 + np.random.normal(0, 0.1, x0.shape)
        starts.append(perturbed)
        
    # Start 3: Different 2nd ring selection?
    # Let's try rotating the whole cluster
    for angle_rot in [15, 30, 45]:
        rot = np.radians(angle_rot)
        new_centers = []
        for cx, cy in centers_init:
            dx = cx - cx_mean
            dy = cy - cy_mean
            nx = dx * np.cos(rot) - dy * np.sin(rot) + cx_mean
            ny = dx * np.sin(rot) + dy * np.cos(rot) + cy_mean
            new_centers.append((nx, ny))
        new_angles = [a + angle_rot for a in angles_init]
        
        p = []
        for i in range(N):
            p.append(new_centers[i][0])
            p.append(new_centers[i][1])
        p.extend(new_angles)
        p.extend([cx_mean, cy_mean, angle_rot])
        starts.append(np.array(p))
        
    for start in starts:
        if time.time() > deadline:
            break
            
        try:
            result = minimize(objective, start, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
            
    # Extract best solution
    centers = best_params[0:22].reshape(N, 2)
    angles = best_params[22:33]
    Cx = best_params[33]
    Cy = best_params[34]
    Phi = best_params[35]
    
    # Verify and adjust S if necessary
    # Recalculate S precisely
    a = np.radians(Phi)
    normals = []
    for i in range(6):
        theta = a + np.pi/6 + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(N):
        verts = hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        dx = verts[:, 0] - Cx
        dy = verts[:, 1] - Cy
        for k in range(6):
            proj = dx * normals[k, 0] + dy * normals[k, 1]
            if proj.max() > max_proj:
                max_proj = proj.max()
                
    S = max_proj / (np.sqrt(3) / 2)
    
    # Add safety margin
    S += 1e-7
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(angles[i])])
        
    return inner, (float(Cx), float(Cy)), float(S), float(Phi)


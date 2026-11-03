import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_vertices(center, side, angle_deg):
    """
    Returns vertices of a regular hexagon.
    center: (x, y)
    side: length
    angle_deg: rotation in degrees
    """
    cx, cy = center
    angle = np.deg2rad(angle_deg)
    # Vertices at 0, 60, 120... relative to angle
    angles = angle + np.deg2rad(np.arange(6) * 60)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def outer_hexagon_side_length(inner_centers, inner_angles, outer_center, outer_angle):
    """
    Calculate the minimum side length of the outer hexagon 
    centered at outer_center with rotation outer_angle 
    that contains all inner hexagons (side 1).
    """
    outer_angle_rad = np.deg2rad(outer_angle)
    # Normals for flat-top hexagon (vertices at 0, 60...) are at 30, 90...
    normal_angles = outer_angle_rad + np.deg2rad(30 + np.arange(6) * 60)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_proj = 0.0
    for i in range(N):
        verts = hexagon_vertices(inner_centers[i], 1.0, inner_angles[i])
        for v in verts:
            diff = v - outer_center
            projs = np.dot(normals, diff)
            current_max = np.max(np.abs(projs))
            if current_max > max_proj:
                max_proj = current_max
                
    apothem = max_proj
    side_length = 2.0 * apothem / np.sqrt(3.0)
    return side_length

def penalty_term(inner_centers):
    """
    Penalty for overlapping inner hexagons.
    Minimum center distance is sqrt(3).
    """
    pen = 0.0
    min_dist = np.sqrt(3.0)
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(inner_centers[i] - inner_centers[j])
            if d < min_dist:
                # Strong penalty to prevent overlap
                pen += (min_dist - d)**2 * 1000.0
    return pen

def objective(params):
    """
    params: 
    0..21: inner centers (x1, y1, x2, y2, ...)
    22..32: inner angles (a1, a2, ...)
    33..34: outer center (cx, cy)
    35: outer angle
    """
    inner_centers = np.zeros((N, 2))
    for i in range(N):
        inner_centers[i][0] = params[2*i]
        inner_centers[i][1] = params[2*i+1]
    
    inner_angles = params[22:33]
    
    outer_center = np.array([params[33], params[34]])
    outer_angle = params[35]
    
    if not np.all(np.isfinite(params)):
        return 1e10
        
    side_len = outer_hexagon_side_length(inner_centers, inner_angles, outer_center, outer_angle)
    pen = penalty_term(inner_centers)
    
    return side_len + pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 1e10
    best_params = None
    
    inits = []
    
    # Init 1: Compact 11-hexagon cluster
    # Center (0,0)
    # 6 neighbors at dist sqrt(3)
    # 4 more at dist sqrt(3) from some of those neighbors
    centers = [[0.0, 0.0]]
    sqrt3 = np.sqrt(3.0)
    # Ring 1
    for k in range(6):
        ang = np.deg2rad(k * 60)
        x = sqrt3 * np.cos(ang)
        y = sqrt3 * np.sin(ang)
        centers.append([x, y])
        
    # Ring 2 (partial) - 4 hexes
    # Placed at (sqrt3, 3), (-sqrt3, 3), (sqrt3, -3), (-sqrt3, -3)
    # Check dist from (sqrt3/2, 1.5) to (sqrt3, 3):
    # dx = sqrt3/2, dy = 1.5. dist^2 = 3/4 + 2.25 = 3. dist = sqrt3. Correct.
    centers.append([sqrt3, 3.0])
    centers.append([-sqrt3, 3.0])
    centers.append([sqrt3, -3.0])
    centers.append([-sqrt3, -3.0])
    
    angles_11 = np.zeros(11)
    outer_c = [0.0, 0.0]
    outer_a = 0.0
    
    p1 = np.array([c for coord in centers for c in coord] + list(angles_11) + outer_c + [outer_a])
    inits.append(p1)
    
    # Init 2: Perturbed Init 1
    np.random.seed(42)
    p2 = p1 + np.random.normal(0, 0.1, size=len(p1))
    inits.append(p2)

    # Run optimization for each init
    for i, p0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, p0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-8, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    # If we have time, try more random restarts with strong penalty
    while time.time() < deadline - 5:
        # Random start near the compact cluster
        p_rand = p1 + np.random.normal(0, 0.5, size=36)
        p_rand[22:33] = np.random.uniform(-10, 10, 11)
        try:
            res = minimize(objective, p_rand, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-7})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except:
            pass

    if best_params is None:
        best_params = p1
        best_score = objective(p1)

    # Extract results
    inner = []
    for i in range(N):
        x = best_params[2*i]
        y = best_params[2*i+1]
        a = best_params[22+i]
        inner.append([x, y, a])
        
    cx = best_params[33]
    cy = best_params[34]
    
    inner_centers = np.array([[inner[i][0], inner[i][1]] for i in range(N)])
    inner_angles = np.array([inner[i][2] for i in range(N)])
    outer_center = np.array([cx, cy])
    outer_angle = best_params[35]
    
    final_side = outer_hexagon_side_length(inner_centers, inner_angles, outer_center, outer_angle)
    
    return inner, (cx, cy), final_side, outer_angle

import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Precompute vertices of a unit regular hexagon (side length 1)
# Centered at origin, one vertex at (1, 0)
angles_hex = np.deg2rad(np.arange(0, 360, 60))
unit_hex_vertices = np.column_stack([np.cos(angles_hex), np.sin(angles_hex)])

def get_hex_vertices(center, angle_deg):
    """
    Get vertices of a unit hexagon centered at 'center' with rotation 'angle_deg'.
    """
    angle_rad = np.deg2rad(angle_deg)
    # Rotation matrix
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return center + unit_hex_vertices @ R.T

def point_in_hexagon(point, side_length, angle_deg=0.0):
    """
    Check if a point is inside a regular hexagon of given side length and rotation.
    Returns True if inside, False otherwise.
    Also returns the minimum distance to the boundary if outside (for penalty).
    """
    # Transform point to hexagon frame
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(-angle_rad), np.sin(-angle_rad)
    R_inv = np.array([[c, -s], [s, c]])
    p = R_inv @ point
    
    # Hexagon with side L, flat top/bottom? 
    # Standard: Vertices at 0, 60...
    # Inradius r = L * sqrt(3) / 2
    r = side_length * np.sqrt(3) / 2
    
    # Normals of the 6 sides are at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.deg2rad(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Distances to sides
    dists = normals @ p
    
    # Max distance should be <= r
    max_dist = np.max(dists)
    
    if max_dist <= r:
        return True, 0.0
    else:
        return False, max_dist - r

def calculate_penalty(positions, angles_deg, L):
    """
    Calculate penalty for a configuration.
    positions: (N, 2) array
    angles_deg: (N,) array
    L: outer side length
    """
    penalty = 0.0
    
    # 1. Containment Penalty
    for i in range(N):
        verts = get_hex_vertices(positions[i], angles_deg[i])
        for v in verts:
            inside, dist = point_in_hexagon(v, L, 0.0)
            if not inside:
                penalty += dist**2
        
    # 2. Overlap Penalty
    # Min distance between centers must be >= sqrt(3)
    min_dist = np.sqrt(3)
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                penalty += (min_dist - d)**2
                
    return penalty

def objective(params):
    """
    Objective function for optimizer.
    params: [L, x1, y1, a1, x2, y2, a2, ...]
    """
    L = params[0]
    if L <= 0:
        return 1e6
    
    positions = []
    angles_deg = []
    for i in range(1, N+1):
        idx = 1 + i*3
        x = params[idx]
        y = params[idx+1]
        a = params[idx+2]
        positions.append([x, y])
        angles_deg.append(a)
        
    positions = np.array(positions)
    angles_deg = np.array(angles_deg)
    
    pen = calculate_penalty(positions, angles_deg, L)
    
    # We want to minimize L + Penalty
    return L + 10.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial guess
    # 7 hexes in cluster
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    # 6 neighbors
    for k in range(6):
        ang = np.deg2rad(k * 60)
        x = np.sqrt(3) * np.cos(ang)
        y = np.sqrt(3) * np.sin(ang)
        centers.append([x, y])
        angles.append(0.0)
        
    # 4 extra hexes in corners
    # Corners of L=3 hex are at distance 3, angles 0, 60, ...
    # Place them at distance 2.8, angles 30, 90, 150, 210 (4 corners)
    for k in [30, 90, 150, 210]:
        ang = np.deg2rad(k)
        x = 2.8 * np.cos(ang)
        y = 2.8 * np.sin(ang)
        centers.append([x, y])
        angles.append(0.0)
        
    # Flatten params
    x0 = [3.2]
    for c, a in zip(centers, angles):
        x0.append(c[0])
        x0.append(c[1])
        x0.append(a)
        
    x0 = np.array(x0)
    
    best_params = x0
    best_score = objective(x0)
    
    # Optimization loop
    # Use Nelder-Mead for robustness
    while time.time() < deadline:
        res = minimize(objective, best_params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
        # Check time
        if time.time() > deadline - 5:
            break
            
    # Extract best L
    best_L = best_params[0]
    
    # Reconstruct positions and angles
    positions = []
    angles_deg = []
    for i in range(1, N+1):
        idx = 1 + i*3
        positions.append([best_params[idx], best_params[idx+1]])
        angles_deg.append(best_params[idx+2])
        
    # Final verification and polish
    # Ensure L is large enough to actually contain them (safety margin)
    # The optimizer minimizes L + penalty. If penalty is 0, L is minimal.
    # If penalty > 0, L might be underestimated.
    # Let's verify and adjust L if necessary.
    
    final_positions = np.array(positions)
    final_angles = np.array(angles_deg)
    
    # Check max violation
    max_violation = 0.0
    for i in range(N):
        verts = get_hex_vertices(final_positions[i], final_angles[i])
        for v in verts:
            inside, dist = point_in_hexagon(v, best_L, 0.0)
            if not inside:
                max_violation = max(max_violation, dist)
                
    if max_violation > 1e-6:
        best_L += max_violation + 0.01
        
    # Return
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
        
    return inner, (0.0, 0.0), float(best_L), 0.0


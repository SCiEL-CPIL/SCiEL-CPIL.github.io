import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_hex_vertices(cx, cy, angle_deg):
    # Regular hexagon of side 1
    # Vertices at distance 1 from center
    angles = np.deg2rad(angle_deg + np.arange(6) * 60)
    xs = cx + np.cos(angles)
    ys = cy + np.sin(angles)
    return np.column_stack((xs, ys))

def max_radial_extent(vertices, container_angle_deg):
    # vertices: (M, 2) array
    # container_angle_deg: rotation of the container hexagon
    # We check 6 normals of the container
    normals = np.deg2rad(container_angle_deg + np.arange(6) * 60)
    # Support function in each normal direction
    # dist = v . n
    # We need max of |dist| for all v?
    # No, the hexagon is centered at origin.
    # The condition is |v . n| <= S * sqrt(3)/2
    # So S >= 2/sqrt(3) * max(|v . n|)
    
    max_proj = 0.0
    for n in normals:
        nx, ny = np.cos(n), np.sin(n)
        projs = np.abs(vertices[:, 0] * nx + vertices[:, 1] * ny)
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj * 2.0 / np.sqrt(3)

def calculate_min_side(params):
    # params: [d2, phi_r1, phi_r2, omega]
    # d2: distance of ring 2
    # phi_r1: rotation of ring 1 hexes (relative to radial)
    # phi_r2: rotation of ring 2 hexes (relative to radial)
    # omega: global rotation of the cluster
    
    d1 = np.sqrt(3)
    d2 = params[0]
    phi_r1 = np.deg2rad(params[1])
    phi_r2 = np.deg2rad(params[2])
    omega = np.deg2rad(params[3])
    
    centers = []
    angles = []
    
    # Center hex
    centers.append((0.0, 0.0))
    angles.append(0.0) # Angle doesn't matter for center, but set to 0
    
    # Ring 1
    for i in range(6):
        theta = np.deg2rad(i * 60 + 0) + omega
        cx = d1 * np.cos(theta)
        cy = d1 * np.sin(theta)
        centers.append((cx, cy))
        # Angle of hexagon: radial direction + offset
        # Radial direction is theta
        angles.append(theta + phi_r1)
        
    # Ring 2
    for i in range(5):
        theta = np.deg2rad(i * 60 + 30) + omega
        cx = d2 * np.cos(theta)
        cy = d2 * np.sin(theta)
        centers.append((cx, cy))
        angles.append(theta + phi_r2)
        
    # Check overlaps
    min_dist_sq = 3.0 # (sqrt(3))^2
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq - 1e-9:
                # Penalty
                return dist_sq * 1000.0 # Return large number if overlap
                
    # Calculate vertices
    all_vertices = []
    for (cx, cy), ang in zip(centers, angles):
        verts = get_hex_vertices(cx, cy, np.rad2deg(ang))
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices)
    
    # Find min side length S
    # Try a few container orientations
    min_S = float('inf')
    # The cluster is not fully symmetric, so we should check multiple container rotations
    # But the cluster has some symmetry.
    # Let's check 0, 30, 60, 90, 120, 150
    for cont_rot in [0, 30, 60, 90, 120, 150]:
        S = max_radial_extent(all_vertices, cont_rot)
        if S < min_S:
            min_S = S
            
    return min_S

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess
    # d2 approx 3.06
    # phi_r1 approx 30 deg (flat side out)
    # phi_r2 approx 30 deg
    # omega 0
    
    x0 = [3.06, 30.0, 30.0, 0.0]
    
    best_params = x0
    best_score = calculate_min_side(x0)
    
    # Optimization
    try:
        res = minimize(calculate_min_side, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
            
        # Check time
        if time.time() > deadline:
            pass # Use best so far
            
    except Exception as e:
        # Fallback to initial guess if optimization fails
        pass
        
    # Construct final result
    d2, phi_r1, phi_r2, omega = best_params
    
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    d1 = np.sqrt(3)
    # Ring 1
    for i in range(6):
        theta_deg = i * 60 + np.rad2deg(omega)
        cx = d1 * np.cos(np.deg2rad(theta_deg))
        cy = d1 * np.sin(np.deg2rad(theta_deg))
        ang = theta_deg + phi_r1
        inner.append([cx, cy, ang])
        
    # Ring 2
    for i in range(5):
        theta_deg = i * 60 + 30 + np.rad2deg(omega)
        cx = d2 * np.cos(np.deg2rad(theta_deg))
        cy = d2 * np.sin(np.deg2rad(theta_deg))
        ang = theta_deg + phi_r2
        inner.append([cx, cy, ang])
        
    # Container
    center = (0.0, 0.0)
    side = best_score
    # Container angle: We used 0-150 in check. 
    # The score function returned the min S over those angles.
    # We should return the angle that achieved this min S.
    # Let's re-calculate to find the best angle.
    
    # Re-generate vertices to find best angle
    all_vertices = []
    for row in inner:
        verts = get_hex_vertices(row[0], row[1], row[2])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    min_S = float('inf')
    best_angle = 0.0
    for cont_rot in [0, 30, 60, 90, 120, 150]:
        S = max_radial_extent(all_vertices, cont_rot)
        if S < min_S:
            min_S = S
            best_angle = cont_rot
            
    return inner, center, min_S, best_angle


import numpy as np
from scipy.optimize import minimize_scalar

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def point_min_side(px, py, theta_outer_rad):
    r = np.sqrt(px*px + py*py)
    if r < 1e-12:
        return 0.0
    phi = np.arctan2(py, px)
    delta = (phi - theta_outer_rad) % (np.pi / 3)
    if delta > np.pi / 6:
        delta = np.pi / 3 - delta
    return r / np.cos(delta)

def compute_min_side(inner_positions, theta_outer_deg):
    theta_outer_rad = np.radians(theta_outer_deg)
    max_s = 0.0
    for (cx, cy, adeg) in inner_positions:
        vx, vy = hex_vertices(cx, cy, 1.0, adeg)
        for i in range(6):
            s_needed = point_min_side(vx[i], vy[i], theta_outer_rad)
            if s_needed > max_s:
                max_s = s_needed
    return max_s

def construct():
    a1x, a1y = np.sqrt(3), 0.0
    a2x, a2y = np.sqrt(3)/2, 1.5
    
    def axial_to_cartesian(q, r):
        return a1x * q + a2x * r, a1y * q + a2y * r
    
    # Multiple candidate clusters
    clusters = [
        # Cluster 1: hexagonal-ish
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (1,1), (0,2), (-1,2), (-2,1)],
        # Cluster 2: 
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (0,2), (-2,0), (-1,-1), (1,-2)],
        # Cluster 3:
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0)],
    ]
    
    best_side = float('inf')
    best_config = None
    best_theta = 0.0
    
    for axial_list in clusters:
        positions = []
        for (q, r) in axial_list:
            x, y = axial_to_cartesian(q, r)
            positions.append((x, y, 0.0))
        
        # Center the cluster
        cx_mean = np.mean([p[0] for p in positions])
        cy_mean = np.mean([p[1] for p in positions])
        positions = [(x - cx_mean, y - cy_mean, a) for (x, y, a) in positions]
        
        result = minimize_scalar(
            lambda t: compute_min_side(positions, t),
            bounds=(0, 60),
            method='bounded',
            options={'xatol': 1e-10}
        )
        if result.success and result.fun < best_side:
            best_side = result.fun
            best_config = positions
            best_theta = result.x
    
    if best_config is None:
        return [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)], (0.0, 0.0), 40.0, 0.0
    
    inner = [[x, y, a] for (x, y, a) in best_config]
    return inner, (0.0, 0.0), best_side, best_theta

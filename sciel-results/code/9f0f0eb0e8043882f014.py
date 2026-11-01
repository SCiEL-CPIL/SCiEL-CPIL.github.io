import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

def _get_hex_vertices(x, y, angle_deg):
    """Get vertices of a unit hexagon centered at (x,y) with rotation angle_deg."""
    angle_rad = np.deg2rad(angle_deg)
    # Base vertices for unit hexagon (side 1) centered at origin
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_angles = np.arange(6) * np.pi / 3
    # Rotate
    angles = base_angles + angle_rad
    xs = np.cos(angles)
    ys = np.sin(angles)
    # Translate
    xs += x
    ys += y
    return np.column_stack((xs, ys))

def _compute_S_and_penalty(params):
    """
    params: [r1, phi1, theta1, r2, phi2, theta2]
    Returns: (S, penalty)
    """
    r1, phi1, theta1 = params[0], params[1], params[2]
    r2, phi2, theta2 = params[3], params[4], params[5]
    
    # Ensure positive radii
    r1 = abs(r1)
    r2 = abs(r2)
    
    # Normalize angles to [0, 60)
    phi1 = phi1 % 60.0
    phi2 = phi2 % 60.0
    theta1 = theta1 % 60.0
    theta2 = theta2 % 60.0
    
    centers = []
    angles = []
    
    # Generate 12 hexagons from 2 orbits
    for k in range(6):
        # Orbit 1
        ang_pos1 = phi1 + k * 60.0
        x1 = r1 * np.cos(np.deg2rad(ang_pos1))
        y1 = r1 * np.sin(np.deg2rad(ang_pos1))
        # The hexagon orientation should rotate with the position to maintain symmetry
        # relative to the radial vector.
        # If the base hexagon at phi1 has angle theta1 (relative to x-axis? or relative to radial vector?)
        # Let's define theta as the rotation of the hexagon itself.
        # To maintain 6-fold symmetry of the *entire* configuration, if we rotate the position by 60,
        # we must rotate the hexagon by 60.
        ang_hex1 = theta1 + k * 60.0
        
        centers.append((x1, y1))
        angles.append(ang_hex1)
        
        # Orbit 2
        ang_pos2 = phi2 + k * 60.0
        x2 = r2 * np.cos(np.deg2rad(ang_pos2))
        y2 = r2 * np.sin(np.deg2rad(ang_pos2))
        ang_hex2 = theta2 + k * 60.0
        
        centers.append((x2, y2))
        angles.append(ang_hex2)
        
    # Calculate S (min side length of outer hexagon containing all inner hexagons)
    # Outer hexagon is centered at (0,0) with angle 0.
    # Normals of outer hexagon sides:
    # A regular hexagon with side S has inradius d = S * sqrt(3) / 2.
    # The sides are perpendicular to normals at angles 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = np.deg2rad([30 + k * 60 for k in range(6)])
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_support = 0.0
    for i in range(N):
        cx, cy = centers[i]
        ang = angles[i]
        verts = _get_hex_vertices(cx, cy, ang)
        # Project vertices onto normals
        # dots = verts @ normals.T
        # max_proj = np.max(dots, axis=0)
        # support = np.max(max_proj)
        # Instead of full matrix mult, just loop for speed in python? 
        # N=12, 6 verts, 6 normals. 12*6*6 = 432 ops. Fast enough.
        for v in verts:
            for n in normals:
                dot = v[0]*n[0] + v[1]*n[1]
                if dot > max_support:
                    max_support = dot
                    
    S = max_support * 2.0 / np.sqrt(3.0)
    
    # Calculate Penalty for overlap
    # Min distance between centers for non-overlap is sqrt(3)
    min_dist = np.sqrt(3.0)
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                diff = min_dist - dist
                penalty += diff * diff
                
    return S + 100.0 * penalty, S, penalty

def construct():
    # Bounds for [r1, phi1, theta1, r2, phi2, theta2]
    # r: 0 to 5
    # phi: 0 to 60
    # theta: 0 to 60
    bounds = [
        (0.0, 5.0), (0.0, 60.0), (0.0, 60.0),
        (0.0, 5.0), (0.0, 60.0), (0.0, 60.0)
    ]
    
    # Differential Evolution
    # seed for reproducibility
    result_de = differential_evolution(
        _compute_S_and_penalty, 
        bounds, 
        seed=42, 
        maxiter=200, 
        popsize=15,
        tol=1e-4,
        init='sobol'
    )
    
    best_params = result_de.x
    best_S, best_S_val, best_pen = _compute_S_and_penalty(best_params)
    
    # Refine with L-BFGS-B
    # We need a function that returns just the scalar objective
    def obj_func(p):
        val, S, pen = _compute_S_and_penalty(p)
        return val
        
    result_min = minimize(
        obj_func, 
        best_params, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    final_params = result_min.x
    final_S, final_S_val, final_pen = _compute_S_and_penalty(final_params)
    
    # If penalty is still high, try a few random restarts? 
    # Given time limit, we stick with the best found.
    
    r1, phi1, theta1 = final_params[0], final_params[1], final_params[2]
    r2, phi2, theta2 = final_params[3], final_params[4], final_params[5]
    
    r1 = abs(r1)
    r2 = abs(r2)
    phi1 = phi1 % 60.0
    phi2 = phi2 % 60.0
    theta1 = theta1 % 60.0
    theta2 = theta2 % 60.0
    
    inner = []
    for k in range(6):
        # Orbit 1
        ang_pos1 = phi1 + k * 60.0
        x1 = r1 * np.cos(np.deg2rad(ang_pos1))
        y1 = r1 * np.sin(np.deg2rad(ang_pos1))
        ang_hex1 = theta1 + k * 60.0
        inner.append([x1, y1, ang_hex1])
        
        # Orbit 2
        ang_pos2 = phi2 + k * 60.0
        x2 = r2 * np.cos(np.deg2rad(ang_pos2))
        y2 = r2 * np.sin(np.deg2rad(ang_pos2))
        ang_hex2 = theta2 + k * 60.0
        inner.append([x2, y2, ang_hex2])
        
    # Sort inner list for consistency? Not required, but nice.
    # The task doesn't specify order.
    
    center = (0.0, 0.0)
    side = final_S_val
    angle = 0.0
    
    return inner, center, side, angle


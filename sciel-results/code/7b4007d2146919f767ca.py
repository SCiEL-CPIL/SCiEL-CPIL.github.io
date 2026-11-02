import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar, minimize
import time

N = 11

# Precompute unit hexagon vertices
# Side length 1
# Vertices at angles 0, 60, ... 300 degrees
# Distance from center to vertex is 1
_hex_angles = np.deg2rad(np.arange(0, 360, 60))
_hex_vertices = np.array([np.cos(_hex_angles), np.sin(_hex_angles)]).T

def _get_hex_vertices(center, angle_deg):
    """
    Returns vertices of a unit hexagon centered at 'center' with rotation 'angle_deg'.
    """
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return center + _hex_vertices @ rot.T

def _sat_overlap(v1, v2):
    """
    Check if two convex polygons v1, v2 overlap using Separating Axis Theorem.
    Returns True if they overlap (or touch).
    """
    # Get edges
    n1 = len(v1)
    n2 = len(v2)
    
    # Check axes from v1
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        # Normal to edge
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        if np.linalg.norm(normal) < 1e-9: continue
        normal = normal / np.linalg.norm(normal)
        
        # Project v1
        proj1 = np.dot(v1, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        
        # Project v2
        proj2 = np.dot(v2, normal)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            return False # Separated
            
    # Check axes from v2
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i + 1) % n2]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        if np.linalg.norm(normal) < 1e-9: continue
        normal = normal / np.linalg.norm(normal)
        
        proj1 = np.dot(v1, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        
        proj2 = np.dot(v2, normal)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False # Separated
            
    return True # Overlap

def _check_all_overlaps(positions, angles):
    """
    Check if any pair of hexagons overlap.
    positions: (N, 2)
    angles: (N,)
    """
    verts = []
    for i in range(N):
        v = _get_hex_vertices(positions[i], angles[i])
        verts.append(v)
        
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                return True
    return False

def _calc_min_side(positions, angles, theta_outer_deg):
    """
    Calculate the minimum side length S of a regular hexagon with rotation theta_outer_deg
    that contains all hexagons defined by positions and angles.
    """
    # Collect all vertices
    all_verts = []
    for i in range(N):
        v = _get_hex_vertices(positions[i], angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    # Normals for outer hexagon
    theta = np.deg2rad(theta_outer_deg)
    # The sides of a regular hexagon are perpendicular to angles theta, theta+60, ...
    # The "width" direction is the normal to the side.
    # Normals are at theta, theta+60, theta+120, theta+180, theta+240, theta+300
    # But opposite normals are redundant for width calculation (max - min).
    # We just need 3 unique directions.
    normals = []
    for k in range(3):
        ang = theta + np.deg2rad(60 * k)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    
    max_width = 0.0
    for n in normals:
        proj = np.dot(all_verts, n)
        width = np.max(proj) - np.min(proj)
        if width > max_width:
            max_width = width
            
    # Side length S relates to width W by W = S * sqrt(3)
    return max_width / np.sqrt(3)

def _objective(params):
    """
    Objective function for the optimizer.
    params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11]
    """
    positions = np.array(params[0:N*3:3]).reshape(-1, 2)
    angles = np.array(params[1:N*3:3])
    
    # Check overlap
    if _check_all_overlaps(positions, angles):
        return 1e6 # Penalty
        
    # Find best outer rotation
    # We can use a coarse grid + local search
    # Or just a fine grid since N is small
    # theta_outer in [0, 60)
    # Let's do a quick optimization for theta_outer
    def f_theta(t):
        return _calc_min_side(positions, angles, t)
        
    res = minimize_scalar(f_theta, bounds=(0, 60), method='bounded', options={'xatol': 1e-4})
    return res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: Honeycomb
    # 11 hexagons. 
    # Center 1, surrounded by 6, then 4 more?
    # Let's just use a random or grid init for DE.
    
    # Bounds
    # Positions in [-5, 5]
    # Angles in [0, 60]
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
        bounds.append((0.0, 60.0)) # angle
        
    # Use Differential Evolution
    # To save time, we limit maxiter
    # N=11, 33 variables.
    
    # If we have time, run DE.
    try:
        # Use a smaller popsize to be faster, but risk local minima
        # We can run multiple short DEs or one longer one.
        # Let's try one DE with maxiter=100
        de_res = differential_evolution(
            _objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=100, 
            popsize=10, 
            tol=1e-4, 
            polish=False # We will polish manually
        )
        
        best_params = de_res.x
        best_score = de_res.fun
        
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to a simple heuristic
        best_params = np.zeros(N*3)
        # Place them in a row
        for i in range(N):
            best_params[i*3] = i * 2.0
            best_params[i*3+1] = 0.0
            best_params[i*3+2] = 0.0
        best_score = _objective(best_params)

    # Polish with SLSQP or L-BFGS-B
    # SLSQP is better for small problems with constraints, but we used penalty.
    # L-BFGS-B is faster.
    
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception as e:
            print(f"Polish failed: {e}")

    # Extract best solution
    positions = np.array(best_params[0:N*3:3]).reshape(-1, 2)
    angles = np.array(best_params[1:N*3:3])
    
    # Find optimal outer hexagon parameters
    def f_theta(t):
        return _calc_min_side(positions, angles, t)
        
    res_theta = minimize_scalar(f_theta, bounds=(0, 60), method='bounded', options={'xatol': 1e-6})
    theta_outer = res_theta.x
    side_length = res_theta.fun
    
    # The center of the outer hexagon
    # We need to calculate the center such that the hexagon of side S and angle theta contains the points.
    # The "width" logic implies the hexagon is centered at the midpoint of the min/max projections?
    # Actually, the minimal enclosing hexagon of a fixed orientation is centered at the point that minimizes the max distance to the sides.
    # For a regular hexagon, this is the center of the minimal bounding box in the 3 normal directions.
    # Let's calculate the center.
    
    all_verts = []
    for i in range(N):
        v = _get_hex_vertices(positions[i], angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    theta = np.deg2rad(theta_outer)
    normals = []
    for k in range(3):
        ang = theta + np.deg2rad(60 * k)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
        
    # The center (cx, cy) should be such that the projections are centered.
    # For each normal n, the range is [min, max]. The center of the range is (min+max)/2.
    # The hexagon center must project to this midpoint for all 3 directions.
    # This is a system of linear equations:
    # n1 . c = (min1 + max1) / 2
    # n2 . c = (min2 + max2) / 2
    # n3 . c = (min3 + max3) / 2
    
    A = np.array(normals)
    b = np.zeros(3)
    for i, n in enumerate(normals):
        proj = np.dot(all_verts, n)
        b[i] = (np.min(proj) + np.max(proj)) / 2.0
        
    # Solve A c = b
    c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    center = c
    
    # Construct inner list
    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], angles[i]])
        
    return inner, (center[0], center[1]), side_length, theta_outer


import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 11

# Lattice vectors
sqrt3 = np.sqrt(3)
u = np.array([sqrt3, 0])
v = np.array([sqrt3/2, 1.5])

def lattice_to_cart(i, j):
    return i * u + j * v

# Define candidate sets of 11 lattice points
# Base 7-core
core = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]

# Candidate 1: Cross
cand1 = core + [(2,0), (-2,0), (0,2), (0,-2)]

# Candidate 2: Diagonal extensions
cand2 = core + [(1,1), (-1,-1), (2,0), (-2,0)]

# Candidate 3: Side extensions
cand3 = core + [(2,0), (2,-1), (-2,1), (-2,0)]

# Candidate 4: Compact "blob"
# Remove (1,-1) and (-1,1) from core? No, keep 7.
# Try adding (0,2), (0,-2), (1,1), (-1,-1)
cand4 = core + [(0,2), (0,-2), (1,1), (-1,-1)]

# Candidate 5: 
cand5 = core + [(2,0), (-2,0), (1,1), (-1,-1)]

candidates = [cand1, cand2, cand3, cand4, cand5]

def get_hex_vertices(theta):
    # Vertices of unit hexagon, rotated by theta
    # Flat-top or pointy-top? 
    # Standard hexagon with side 1.
    # Vertices at angles 0, 60, ... if pointy-top?
    # Let's use vertices at 0, 60, ... relative to x-axis.
    angles = np.deg2rad(np.arange(0, 360, 60) + theta)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def calc_min_side(points_cart, theta):
    # points_cart: array of shape (11, 2)
    # theta: rotation angle in radians
    verts = get_hex_vertices(theta) # shape (6, 2)
    
    # Rotate centers
    c_rot = np.column_stack((
        points_cart[:, 0] * np.cos(theta) - points_cart[:, 1] * np.sin(theta),
        points_cart[:, 0] * np.sin(theta) + points_cart[:, 1] * np.cos(theta)
    ))
    
    # Rotate vertices
    v_rot = np.column_stack((
        verts[:, 0] * np.cos(theta) - verts[:, 1] * np.sin(theta),
        verts[:, 0] * np.sin(theta) + verts[:, 1] * np.cos(theta)
    ))
    
    # All vertices of all hexagons
    # Shape (11, 6, 2)
    all_verts = c_rot[:, np.newaxis, :] + v_rot[np.newaxis, :, :]
    
    # Unroll to (66, 2)
    all_verts = all_verts.reshape(-1, 2)
    
    # Container normals (for hexagon aligned with axes, side S)
    # Normals at 30, 90, 150, 210, 270, 330
    phi = np.deg2rad(np.arange(30, 360, 60))
    normals = np.column_stack((np.cos(phi), np.sin(phi)))
    
    # Project all vertices onto normals
    # Shape (66, 6)
    projections = all_verts @ normals.T
    
    # Max projection
    max_proj = np.max(projections)
    
    # S = max_proj / inradius_factor
    # inradius = S * sqrt(3) / 2
    # max_proj <= S * sqrt(3) / 2
    # S >= 2 * max_proj / sqrt(3)
    return 2 * max_proj / sqrt3

def optimize_cluster(points_lattice):
    points_cart = np.array([lattice_to_cart(i, j) for i, j in points_lattice])
    
    # Grid search
    thetas = np.linspace(0, np.pi/3, 1000)
    best_s = np.inf
    best_theta = 0
    
    for th in thetas:
        s = calc_min_side(points_cart, th)
        if s < best_s:
            best_s = s
            best_theta = th
            
    # Refine
    res = minimize_scalar(lambda th: calc_min_side(points_cart, th), 
                          bounds=(best_theta - 0.01, best_theta + 0.01), 
                          method='bounded', options={'xatol': 1e-6})
    
    if res.success:
        best_s = res.fun
        best_theta = res.x
        
    return best_s, best_theta, points_cart

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_overall_s = np.inf
    best_result = None
    
    for cand in candidates:
        if time.time() > deadline:
            break
            
        s, theta, points_cart = optimize_cluster(cand)
        
        if s < best_overall_s:
            best_overall_s = s
            # Construct inner list
            # inner rows: (x, y, angle_degrees)
            # angle_degrees is the rotation of the inner hexagon
            # Since all are aligned with cluster, angle = theta
            angle_deg = np.rad2deg(theta)
            inner = []
            for p in points_cart:
                inner.append([p[0], p[1], angle_deg])
            
            best_result = (inner, (0.0, 0.0), best_overall_s, 0.0)
            
    if best_result is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_result



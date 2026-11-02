import numpy as np
from scipy.optimize import minimize_scalar

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the centers of the 12 hexagons
    # Core 7
    sqrt3 = np.sqrt(3)
    centers = []
    centers.append(np.array([0.0, 0.0]))
    
    # 6 neighbors
    # v1 = (sqrt3, 0), v2 = (sqrt3/2, 3/2)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2, 1.5])
    
    neighbors = [
        v1,
        v2,
        v2 - v1, # (-sqrt3/2, 1.5)
        -v1,
        -v2,
        v1 - v2 # (sqrt3/2, -1.5)
    ]
    centers.extend(neighbors)
    
    # 5 side positions from the R=2 shell
    # Side positions are midpoints of the corners of the R=2 hexagon
    # Corners of R=2 shell:
    # c1 = 2*v1
    # c2 = 2*v2
    # c3 = 2*(v2-v1)
    # c4 = -2*v1
    # c5 = -2*v2
    # c6 = 2*(v1-v2)
    
    corners = [
        2*v1,
        2*v2,
        2*(v2-v1),
        -2*v1,
        -2*v2,
        2*(v1-v2)
    ]
    
    sides = []
    for i in range(6):
        # Midpoint between corner i and corner i+1
        c_i = corners[i]
        c_next = corners[(i+1)%6]
        mid = (c_i + c_next) / 2.0
        sides.append(mid)
        
    # We need 5 more. We take the first 5 sides.
    # This breaks symmetry, but minimizes max radial distance (3 vs 3.46)
    centers.extend(sides[:5])
    
    centers = np.array(centers)
    
    # Support function of a unit hexagon (side 1, circumradius 1)
    # Vertices at angles 0, pi/3, ...
    # h(theta) = max_k cos(theta - k*pi/3)
    def h_hex(theta):
        # theta is array
        angles = np.arange(6) * np.pi / 3
        # cos(theta - angles)
        # shape: (N_theta, 6)
        cos_vals = np.cos(theta[:, None] - angles[None, :])
        return np.max(cos_vals, axis=1)
    
    # Support function of the cluster
    # H_cluster(theta) = max_i ( h_hex(theta - inner_rot) + centers[i] . u_theta )
    # inner_rot is 0 for now
    def H_cluster(theta, inner_rot=0.0):
        # theta: array of angles
        # h_inner: support of single hexagon
        h_inner = h_hex(theta - inner_rot)
        
        # centers . u_theta
        # u_theta = [cos(theta), sin(theta)]
        u_theta = np.column_stack((np.cos(theta), np.sin(theta)))
        
        # dots: (N_theta, N_centers)
        dots = u_theta @ centers.T
        
        # H_cluster = max over i of (h_inner + dots[:, i])
        # h_inner is (N_theta, 1)
        # dots is (N_theta, N_centers)
        H = np.max(h_inner[:, None] + dots, axis=1)
        return H
    
    # Objective function to minimize
    # We want to minimize the scale factor S such that H_cluster(theta) <= S * h_hex(theta - phi)
    # S = max_theta H_cluster(theta) / h_hex(theta - phi)
    def objective(phi):
        # Sample theta densely
        thetas = np.linspace(0, 2*np.pi, 720, endpoint=False)
        
        H_c = H_cluster(thetas, inner_rot=0.0)
        H_outer = h_hex(thetas - phi)
        
        # Avoid division by zero (h_hex is always >= sqrt(3)/2)
        ratios = H_c / H_outer
        return np.max(ratios)
    
    # Optimize phi in [0, pi/3]
    res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-6})
    best_phi = res.x
    best_S = res.fun
    
    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    # center: (0.0, 0.0)
    center = (0.0, 0.0)
    
    # side: best_S
    side = float(best_S)
    
    # angle: best_phi (in degrees)
    angle = float(np.deg2rad(best_phi)) # Wait, the return interface asks for angle_degrees for inner, and outer rotation?
    # "outer hexagon center (x, y), side length and rotation"
    # The rotation is likely in degrees.
    angle_deg = float(np.rad2deg(best_phi))
    
    return inner, center, side, angle_deg

# EVOLVE-BLOCK-END

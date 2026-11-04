import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Lattice vectors for flat-top hexagons
    # Distance between centers is sqrt(3)
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    # Centers of the 13-cluster (7 center + 6 neighbors + 6 edge-midpoints)
    # We will use 12 of these.
    
    # Center
    c0 = np.array([0.0, 0.0])
    
    # 6 Neighbors
    n1 = u
    n2 = v
    n3 = v - u
    n4 = -u
    n5 = -v
    n6 = u - v
    
    # 6 Edge-midpoints (Ring 2, between corners)
    e1 = u + v
    e2 = 2*v - u
    e3 = v - 2*u
    e4 = -u - v
    e5 = -2*v + u
    e6 = -v + 2*u
    
    # Select 12 centers
    # Remove one neighbor (n1) to break symmetry and reduce max extent slightly?
    # Actually, removing any one is symmetric with respect to the optimization of theta.
    # Let's remove n1.
    centers = [c0, n2, n3, n4, n5, n6, e1, e2, e3, e4, e5, e6]
    
    # Vertices of a unit flat-top hexagon
    angles = np.deg2rad(np.arange(0, 360, 60))
    hex_verts = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # All vertices of all 12 hexagons
    all_verts = []
    for c in centers:
        all_verts.extend((hex_verts + c).tolist())
    all_verts = np.array(all_verts)
    
    # Objective function: minimize max projection
    def obj(theta):
        if time.time() > deadline:
            return 1e9
        k = np.arange(6)
        angles_n = theta + k * np.pi / 3
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles_n])
        dots = normals @ all_verts.T
        return np.max(dots)
    
    # Minimize
    res = minimize_scalar(obj, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-8})
    theta_opt = res.x
    max_proj = res.fun
    
    side_length = (2.0 / np.sqrt(3.0)) * max_proj
    
    # Prepare output
    inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
    center = (0.0, 0.0)
    
    return inner, center, float(side_length), float(np.rad2deg(theta_opt))
# EVOLVE-BLOCK-END

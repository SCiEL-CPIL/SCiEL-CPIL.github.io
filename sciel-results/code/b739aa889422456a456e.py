import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _hex_vertices(cx, cy, angle_deg=0.0):
    """
    Get vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg.
    Vertices are at distance 1 from center.
    """
    angle_rad = np.deg2rad(angle_deg)
    # Base vertices at angles 0, 60, 120, 180, 240, 300
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _min_outer_side(inner_centers, inner_angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length S of the outer hexagon (centered at 0,0, rotated by outer_angle_deg)
    that contains all inner hexagons.
    """
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = np.array([[np.cos(theta_rad + np.deg2rad(30 + k * 60)), 
                         np.sin(theta_rad + np.deg2rad(30 + k * 60))] for k in range(6)])
    
    max_proj = 0.0
    for (cx, cy), ang in zip(inner_centers, inner_angles):
        verts = _hex_vertices(cx, cy, ang)
        for (vx, vy) in verts:
            projs = vx * normals[:, 0] + vy * normals[:, 1]
            local_max = np.max(projs)
            if local_max > max_proj:
                max_proj = local_max
    
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _min_outer_side_vectorized(inner_centers, inner_angles, outer_angle_deg=0.0):
    """
    Vectorized version for speed.
    """
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = np.array([[np.cos(theta_rad + np.deg2rad(30 + k * 60)), 
                         np.sin(theta_rad + np.deg2rad(30 + k * 60))] for k in range(6)])
    
    max_proj = 0.0
    for i, ((cx, cy), ang) in enumerate(zip(inner_centers, inner_angles)):
        verts = _hex_vertices(cx, cy, ang)
        # verts is (6, 2), normals is (6, 2)
        projs = verts[:, 0] * normals[:, 0] + verts[:, 1] * normals[:, 1]
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
    
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _objective(x, outer_angle_deg=0.0):
    """
    Objective function for optimization: minimize outer side length.
    x = [x1, y1, a1, x2, y2, a2, ...] for N hexagons
    """
    n = len(x) // 3
    centers = [(x[3*i], x[3*i+1]) for i in range(n)]
    angles = [x[3*i+2] for i in range(n)]
    return _min_outer_side(centers, angles, outer_angle_deg)

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # Lattice basis vectors for hexagonal packing
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate candidate topologies (list of lattice indices (i,j))
    # Topology 1: 7 central + 4 at (2,0), (0,2), (-2,0), (0,-2) - cross pattern
    t1 = [(0,0)] + [(i,j) for i,j in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]] + [(2,0), (0,2), (-2,0), (0,-2)]
    # Topology 2: 7 central + 4 at (1,1), (1,-1), (-1,1), (-1,-1) - diagonal pattern
    t2 = [(0,0)] + [(i,j) for i,j in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]] + [(1,1), (1,-1), (-1,1), (-1,-1)]
    # Topology 3: 7 central + 4 consecutive: (2,0), (1,1), (0,2), (-1,1)
    t3 = [(0,0)] + [(i,j) for i,j in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]] + [(2,0), (1,1), (0,2), (-1,1)]
    # Topology 4: 7 central + 4 at (2,0), (1,1), (-1,1), (-2,0) - top arc
    t4 = [(0,0)] + [(i,j) for i,j in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]] + [(2,0), (1,1), (-1,1), (-2,0)]
    # Topology 5: 7 central + 4 at (0,2), (1,1), (1,-1), (0,-2) - right arc
    t5 = [(0,0)] + [(i,j) for i,j in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]] + [(0,2), (1,1), (1,-1), (0,-2)]
    # Topology 6: 7 central + 4 at (-2,0), (-1,1), (-1,-1), (0,-2) - left-bottom
    t6 = [(0,0)] + [(i,j) for i,j in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]] + [(-2,0), (-1,1), (-1,-1), (0,-2)]
    
    topologies = [t1, t2, t3, t4, t5, t6]
    
    best_S = float('inf')
    best_config = None
    
    for topo in topologies:
        # Convert indices to coordinates
        centers = []
        for i, j in topo:
            c = i * A + j * B
            centers.append(tuple(c))
        
        # Try rotations of the inner set: finer sampling
        for rot_deg in range(0, 180, 15):
            # Rotate centers around origin
            rot_rad = np.deg2rad(rot_deg)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            rotated_centers = []
            for (x, y) in centers:
                rx = x * cos_r - y * sin_r
                ry = x * sin_r + y * cos_r
                rotated_centers.append((rx, ry))
            
            # Inner hexagons can also be rotated. 
            # For optimal packing, try several inner angles
            for inner_ang in [0, 15, 30, 45, 60]:
                if time.time() > deadline - 30:
                    break
                S = _min_outer_side(rotated_centers, [inner_ang]*N, outer_angle_deg=0.0)
                if S < best_S:
                    best_S = S
                    best_config = (rotated_centers, [inner_ang]*N, 0.0)
    
    if best_config is None:
        # Fallback to seed-like behavior if something went wrong
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    centers, angles, outer_ang = best_config
    
    # Local optimization pass to refine positions
    if time.time() < deadline - 15:
        # Prepare initial x vector
        x0 = []
        for (cx, cy), ang in zip(centers, angles):
            x0.extend([cx, cy, np.deg2rad(ang)])
        
        # Try optimization
        try:
            result = minimize(_objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_S:
                # Extract optimized positions
                opt_centers = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(result.x[3*i+2]) for i in range(N)]
                best_S = result.fun
                best_config = (opt_centers, opt_angles, 0.0)
        except Exception:
            pass
    
    centers, angles, outer_ang = best_config
    # Format output
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    return inner, (0.0, 0.0), best_S, outer_ang

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

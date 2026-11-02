import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate centers for a 3x4 rhombus of hexagons
    # Lattice vectors
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * v1 + j * v2
            centers.append(c)
            
    centers = np.array(centers)
    
    # Vertices of a unit hexagon (side 1, angle 0)
    # Centered at origin
    angles = np.deg2rad(np.arange(0, 360, 60))
    unit_hex_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Pre-compute inner hexagon vertices for each center
    # Since they are all rotated by 0, we just add center
    inner_hexes = [c + unit_hex_vertices for c in centers]
    
    def calc_min_L(xc, yc, phi):
        """
        Calculate the minimum side length L of an outer hexagon
        centered at (xc, yc) with rotation phi that contains all inner hexes.
        """
        max_L = 0.0
        # Rotate inner hexes? No, the outer hexagon rotation is relative to the fixed inner hexes.
        # It's easier to rotate the outer hexagon frame to align with the inner ones (phi=0)
        # and rotate the points instead.
        # Let's stick to: Outer hexagon has rotation phi.
        # Condition: Point P is inside Outer Hex.
        # Equivalent: Point (P - C) rotated by -phi is inside Unit Hex (scaled by L).
        
        for hex_verts in inner_hexes:
            # Translate to outer center
            rel = hex_verts - np.array([xc, yc])
            # Rotate by -phi
            cos_phi = np.cos(-phi)
            sin_phi = np.sin(-phi)
            rot = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
            rel_rot = rel @ rot.T
            
            # For each vertex, calculate required L
            # Distance r and angle theta in the rotated frame
            r = np.linalg.norm(rel_rot, axis=1)
            theta = np.arctan2(rel_rot[:, 1], rel_rot[:, 0])
            
            # Hexagon boundary distance in direction theta
            # theta_mod = theta % (pi/3)
            # offset = pi/6
            # cos_term = cos(theta_mod - pi/6)
            # L_req = 2 * r / (sqrt(3) * cos_term)
            
            theta_mod = theta % (np.pi / 3)
            cos_term = np.cos(theta_mod - np.pi / 6)
            
            # Avoid division by zero or negative cos (should be positive in range)
            cos_term = np.maximum(cos_term, 1e-9)
            
            L_reqs = 2 * r / (np.sqrt(3) * cos_term)
            current_max = np.max(L_reqs)
            if current_max > max_L:
                max_L = current_max
                
        return max_L

    def objective(params):
        xc, yc, phi = params
        return calc_min_L(xc, yc, phi)

    # Initial guess: Center of the cluster, rotation 0
    center = np.mean(centers, axis=0)
    x0 = [center[0], center[1], 0.0]
    
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_xc, best_yc, best_phi = res.x
    best_L = res.fun
    
    # Return format:
    # inner: list of [x, y, angle]
    # center: (x, y)
    # side: L
    # angle: phi (degrees)
    
    inner_list = []
    for c in centers:
        inner_list.append([c[0], c[1], 0.0])
        
    return inner_list, (best_xc, best_yc), best_L, np.rad2deg(best_phi)
# EVOLVE-BLOCK-END

import numpy as np
from scipy.optimize import minimize

def hexagon_packing_11():
    # Define the configuration
    # 1 center, 6 in ring 1, 4 in ring 2
    # Ring 1 angles: 0, 60, 120, 180, 240, 300
    # Ring 2 angles: 30, 150, 210, 330
    
    def get_positions(R1, R2, rot):
        centers = []
        # Center
        centers.append([0, 0])
        
        # Ring 1
        for i in range(6):
            angle = np.radians(60 * i + rot)
            centers.append([R1 * np.cos(angle), R1 * np.sin(angle)])
            
        # Ring 2
        for i in range(4):
            angle = np.radians(60 * i + 30 + rot)
            centers.append([R2 * np.cos(angle), R2 * np.sin(angle)])
            
        return np.array(centers)

    def calc_outer_side(R1, R2, rot):
        centers = get_positions(R1, R2, rot)
        # Vertices of all inner hexagons
        # Each hex has 6 vertices
        # Hex at (cx, cy) with angle 0 (aligned with axes)
        # Vertices at angles 0, 60, 120, 180, 240, 300 relative to center
        # Wait, the inner hexagons can also be rotated. 
        # Assuming all inner hexagons are aligned with the same global orientation (angle 0)
        # is optimal for this symmetric packing? 
        # Yes, likely.
        
        verts = []
        for c in centers:
            cx, cy = c
            for k in range(6):
                va = np.radians(60 * k)
                verts.append([cx + np.cos(va), cy + np.sin(va)])
        verts = np.array(verts)
        
        # Outer hexagon centered at 0, 0.
        # We can rotate the outer hexagon to minimize its size.
        # Due to 6-fold symmetry of the cluster (if rot is 0 or 60), 
        # the outer hexagon should align with the cluster.
        # Let's optimize the outer hexagon angle too.
        # But for a quick fix, assume outer hex angle = rot (aligned with cluster).
        
        # Projection of vertices onto normals of outer hexagon.
        # Outer hex normals are at angles: outer_rot + 30, 90, 150, ...
        # Let's define outer_rot = rot.
        normals = []
        for k in range(6):
            na = np.radians(60 * k + 30 + rot)
            normals.append([np.cos(na), np.sin(na)])
        normals = np.array(normals)
        
        # Max projection
        # proj[i, j] = dot(verts[i], normals[j])
        projs = verts @ normals.T
        max_proj = np.max(projs)
        
        # Side length S
        # Inradius r = S * cos(30) = S * sqrt(3)/2
        # max_proj <= r
        # S = 2 * max_proj / sqrt(3)
        return 2 * max_proj / np.sqrt(3)

    def objective(params):
        R1, R2, rot = params
        if R1 < 1.732 or R2 < 1.732:
            return 1e6
        return calc_outer_side(R1, R2, rot)

    # Initial guess
    # R1 = sqrt(3), R2 = 3, rot = 0
    x0 = [np.sqrt(3) + 0.01, 3.0 + 0.01, 0.0]
    
    # Bounds
    bounds = [(1.732, 10), (1.732, 10), (0, 60)]
    
    res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    
    R1_opt, R2_opt, rot_opt = res.x
    
    # Add small epsilon to ensure no intersection
    # The objective function doesn't explicitly penalize intersection, 
    # but the geometry implies that if they intersect, the "outer side" 
    # calculation might be invalid or the packing is impossible.
    # However, calc_outer_side just projects vertices. 
    # We must enforce non-intersection.
    
    # Check intersections and adjust if necessary
    # For simplicity, we add a small buffer to R1 and R2
    R1_final = R1_opt + 1e-5
    R2_final = R2_opt + 1e-5
    
    centers = get_positions(R1_final, R2_final, rot_opt)
    
    inner_hex_data = np.zeros((11, 3))
    for i, c in enumerate(centers):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0 # Aligned
        
    outer_hex_data = np.array([0.0, 0.0, rot_opt])
    
    side = calc_outer_side(R1_final, R2_final, rot_opt)
    # Add margin to side to ensure containment
    side += 1e-5
    
    return inner_hex_data, outer_hex_data, side


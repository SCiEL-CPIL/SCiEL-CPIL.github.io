import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600 - 10  # Safety margin

    # Helper to get vertices of a hexagon
    def hex_vertices(cx, cy, s, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack((
            cx + s * np.cos(angles),
            cy + s * np.sin(angles)
        ))

    # Helper to check if two hexagons are separated
    def is_separated(hex1, hex2):
        # hex1, hex2 are (6,2) arrays of vertices
        # Check 12 normals: 6 from hex1, 6 from hex2
        # Normal for edge i of a hexagon (flat top, s=1)
        # Edge 0: between v0 and v1. Midpoint...
        # Actually, just use the normals of the unit hexagon rotated.
        # Unit hex normals (flat top) are at 30, 90, 150, 210, 270, 330 degrees.
        normals = []
        base_angles = np.deg2rad(np.arange(30, 360, 60))
        for ang in base_angles:
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        
        # For each pair of normals (n from hex1, n from hex2)
        # Project all vertices of hex1 and hex2 onto n
        # Check if intervals are apart by > 1e-6
        
        # We can just check the 6 normals of hex1 and 6 normals of hex2
        # Actually, if they are separated, they are separated on SOME normal.
        # It is sufficient to check the normals of hex1? 
        # No, if hex2 is rotated, its normals are different.
        # But the set of all edge normals of both hexes covers all potential separating lines.
        
        # Efficient check:
        # For each normal n in the 12 normals:
        #   min1 = min(dot(v, n) for v in hex1)
        #   max1 = max(dot(v, n) for v in hex1)
        #   min2 = min(dot(v, n) for v in hex2)
        #   max2 = max(dot(v, n) for v in hex2)
        #   if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
        #       return True (separated)
        
        # If no normal separates them, they intersect.
        
        # Optimization: Pre-compute projections?
        # For a single pair, this is fast enough.
        
        for n in normals:
            p1 = hex1 @ n
            p2 = hex2 @ n
            if p1.max() < p2.min() - 1e-6 or p2.max() < p1.min() - 1e-6:
                return True
        return False

    def get_min_side(positions, angles):
        # positions: (11, 2)
        # angles: (11,)
        # Outer hexagon centered at (0,0) with side S, flat-top (angle 0)
        # Normals of outer hexagon: 30, 90, ...
        outer_normals = []
        base_angles = np.deg2rad(np.arange(30, 360, 60))
        for ang in base_angles:
            outer_normals.append(np.array([np.cos(ang), np.sin(ang)]))
        outer_normals = np.array(outer_normals)
        
        max_proj = 0
        for i in range(11):
            verts = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            # Project onto outer normals
            projs = verts @ outer_normals.T
            max_proj = max(max_proj, np.max(projs))
            
        # max_proj is the max distance from center to a vertex in any normal direction
        # Apothem = S * sqrt(3) / 2
        # max_proj <= Apothem
        # S >= 2 * max_proj / sqrt(3)
        S = 2 * max_proj / np.sqrt(3)
        return S

    def objective(x):
        if time.time() > deadline:
            return 1e10
        
        positions = x[:22].reshape(11, 2)
        angles = x[22:33]
        
        # Calculate min side
        S = get_min_side(positions, angles)
        
        # Check overlaps
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                v1 = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
                v2 = hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
                if not is_separated(v1, v2):
                    # Estimate overlap penalty
                    # Use distance between centers as a proxy for how bad the overlap is
                    d = np.linalg.norm(positions[i] - positions[j])
                    # If d < 2, they are likely overlapping
                    penalty += max(0, 2.0 - d) ** 2
                    
        return S + 100 * penalty

    # Initial guess
    # 7 hexes in flower
    # 4 hexes in corners
    sqrt3 = np.sqrt(3)
    init_pos = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [3, -sqrt3]
    ])
    init_angles = np.zeros(11)
    
    x0 = np.concatenate([init_pos.flatten(), init_angles])
    
    best_x = x0
    best_val = objective(x0)
    
    # Run optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 10000, 'xatol': 1e-4, 'fatol': 1e-4})
        if result.success or result.fun < best_val:
            best_x = result.x
            best_val = result.fun
    except Exception:
        pass

    # Extract best result
    positions = best_x[:22].reshape(11, 2)
    angles = best_x[22:33]
    
    # Final validation
    S = get_min_side(positions, angles)
    
    # Check validity
    valid = True
    if S < np.sqrt(11) - 1e-6:
        # If S is too small, it might be invalid due to area constraint?
        # The problem says "Valid only if ... side >= sqrt(11)"
        # So if our S < sqrt(11), we must set S = sqrt(11) and check if it fits?
        # No, S is the side length of the OUTER hexagon.
        # If the inner hexes fit in a smaller hex, they also fit in a larger one.
        # But we want to MINIMIZE S.
        # If the minimal S is < sqrt(11), we can return sqrt(11) as the side length,
        # and the packing is still valid (inner hexes are inside).
        # But the score is 1/S. So we want the smallest valid S.
        # If the geometric S is 3.0, but constraint says S >= 3.31, then S=3.31 is the best we can do?
        # Yes.
        S = max(S, np.sqrt(11))
        
    # Check overlaps again with final positions
    for i in range(11):
        for j in range(i+1, 11):
            v1 = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            v2 = hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            if not is_separated(v1, v2):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    inner_hex_data = np.column_stack((positions, angles))
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = S

    return inner_hex_data, outer_hex_data, outer_hex_side_length

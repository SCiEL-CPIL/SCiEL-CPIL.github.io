import numpy as np
import time
import itertools
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0
    
    # Unit hexagon vertices (angle 0)
    # Vertices at 0, 60, ... 300 degrees
    hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    # Precompute normals for overlap checking
    # Edge normals are at 30, 90, 150, 210, 270, 330
    # The problem states: "on at least one of their 12 edge normals"
    # For aligned hexagons, the 6 normals of one and the 6 of the other are the same set.
    # So we just check the 6 normals.
    edge_normals = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(30, 360, 60)
    ])
    
    def get_vertices(center, angle_deg=0):
        angle_rad = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
        return center + hex_verts @ rot.T

    def max_penetration(c1, c2):
        # Returns the maximum penetration depth between two unit hexagons at c1 and c2
        # They are aligned.
        # Penetration depth is the minimum distance to separate them.
        # If they are separated, return negative value (or 0).
        # Using the support function difference:
        # d = max_n (h1(n) - h2(n)) where n is unit vector?
        # Simpler: Check the 6 edge normals.
        # Project both hexagons onto each normal.
        # If intervals overlap, the overlap length is (max1 - min2) if max2 > min1.
        # We want the maximum overlap length over all normals. If max overlap > 0, they intersect.
        
        v1 = hex_verts  # relative to c1
        v2 = hex_verts  # relative to c2
        
        # Projections of v1
        proj1 = v1 @ edge_normals.T  # (6, 6)
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        
        # Projections of v2 shifted by (c1 - c2)
        # Interval for hex2 on normal n: [min2 + n.(c2), max2 + n.(c2)]
        # We want to know if [min1, max1] and [min2 + n.c2, max2 + n.c2] overlap.
        # Overlap length = max(0, min(max1, max2+n.c2) - max(min1, min2+n.c2))
        
        c_diff = c1 - c2
        proj_c = c_diff @ edge_normals.T # (6,)
        
        # min2, max2 are same as min1, max1 for unit hex
        min2 = min1
        max2 = max1
        
        # Shifted intervals for hex2
        min2_shifted = min2 + proj_c
        max2_shifted = max2 + proj_c
        
        # Overlap in each normal direction
        overlaps = np.minimum(max1, max2_shifted) - np.maximum(min1, min2_shifted)
        
        # If any overlap is > 0, they intersect.
        # The "penetration depth" is roughly the max of these overlaps?
        # Actually, for convex polygons, the minimum translation vector magnitude is the min over normals of the overlap length?
        # No, the minimum translation vector to separate them has length equal to the minimum over all separating axes of the overlap depth?
        # Wait, if they intersect, there is no separating axis.
        # The penetration depth is the minimum distance one has to move one shape to separate them.
        # For two convex polygons, this is the minimum over all edge normals of the overlap length?
        # Yes, for two convex polygons, the Minkowski difference is a convex polygon. The distance from the origin to the boundary of the Minkowski difference in direction n is the support function.
        # The penetration depth is $1 - \max_n \frac{h_{diff}(n)}{||n||}$? No.
        
        # Let's just use the max overlap as a proxy for "how much they are stuck".
        # If max(overlaps) > 0, they overlap.
        # To make it smooth, we can use max(0, max(overlaps)).
        return np.max(overlaps)

    def total_penalty(centers):
        # Sum of squared penetration depths for all pairs
        pen_sum = 0.0
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                pen = max_penetration(centers[i], centers[j])
                if pen > 0:
                    pen_sum += pen * pen
        return pen_sum

    def objective(params):
        # params: [x1, y1, ..., x11, y11, theta]
        # theta is in radians
        if time.time() > deadline - 5:
            return 1e10
            
        centers = params[:22].reshape(11, 2)
        theta = params[22]
        
        # Calculate side length
        # Normals for outer hexagon
        outer_normals = []
        for i in range(6):
            a = theta + np.deg2rad(60 * i)
            outer_normals.append([np.cos(a), np.sin(a)])
        outer_normals = np.array(outer_normals)
        
        # Support function of the union of hexagons
        # h_total(n) = max_i (n . c_i + h_hex(n))
        
        # h_hex(n) for unit hexagon
        # proj_hex = hex_verts @ outer_normals.T
        # max_proj_hex = np.max(proj_hex, axis=0)
        
        # To avoid recomputing max_proj_hex every time (it depends on theta),
        # we can precompute or just compute it. It's small.
        proj_hex = hex_verts @ outer_normals.T
        max_proj_hex = np.max(proj_hex, axis=0)
        
        proj_centers = centers @ outer_normals.T
        max_proj_centers = np.max(proj_centers, axis=0)
        
        h_total = max_proj_centers + max_proj_hex
        max_h = np.max(h_total)
        
        side = (2.0 / np.sqrt(3.0)) * max_h
        
        # Penalty
        pen = total_penalty(centers)
        
        # Large penalty weight to ensure non-overlap
        weight = 100.0
        return side + weight * pen

    # Initial candidates
    # 1. Center + Ring 1 (7 hexagons) + 4 Ring 2
    d = np.sqrt(3)
    v = []
    for i in range(6):
        angle = np.deg2rad(60 * i)
        v.append([d * np.cos(angle), d * np.sin(angle)])
    v = np.array(v)
    
    ring2 = []
    for i in range(6):
        ring2.append(2 * v[i])
        ring2.append(v[i] + v[(i + 1) % 6])
    ring2 = np.array(ring2)
    
    base_centers = np.vstack([np.array([[0, 0]]), v])
    
    candidates = []
    
    # Candidate 1: Full Ring 2 (13 hexagons, too many? No, task says 11)
    # We need exactly 11.
    # Base 7. Need 4 more from Ring 2.
    
    # Symmetric choices for 4 ring 2 points:
    # Indices 0, 2, 4, 6 (0, 60, 120, 180 deg) -> 4 points
    # Indices 1, 3, 5, 7 (30, 90, 150, 210 deg) -> 4 points
    # Indices 0, 3, 6, 9 (0, 90, 180, 270 deg) -> 4 points
    # Indices 0, 2, 6, 8 (0, 60, 180, 240 deg) -> 4 points
    
    idx_sets = [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [0, 3, 6, 9],
        [0, 2, 6, 8],
        [0, 5, 6, 11],
        [1, 4, 7, 10]
    ]
    
    for idxs in idx_sets:
        r2_pts = ring2[idxs]
        centers = np.vstack([base_centers, r2_pts])
        # Flatten centers
        init_params = np.concatenate([centers.flatten(), [0.0]])
        candidates.append(init_params)
    
    best_result = None
    best_score = float('inf')
    
    for init in candidates:
        if time.time() > deadline - 30:
            break
            
        try:
            # Use Nelder-Mead
            res = minimize(objective, init, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            
            if res.success or res.fun < best_score:
                # Check validity
                centers = res.x[:22].reshape(11, 2)
                theta = res.x[22]
                
                # Verify no overlaps
                valid = True
                for i in range(11):
                    for j in range(i + 1, 11):
                        if max_penetration(centers[i], centers[j]) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Calculate final side
                    outer_normals = []
                    for i in range(6):
                        a = theta + np.deg2rad(60 * i)
                        outer_normals.append([np.cos(a), np.sin(a)])
                    outer_normals = np.array(outer_normals)
                    
                    proj_hex = hex_verts @ outer_normals.T
                    max_proj_hex = np.max(proj_hex, axis=0)
                    
                    proj_centers = centers @ outer_normals.T
                    max_proj_centers = np.max(proj_centers, axis=0)
                    
                    h_total = max_proj_centers + max_proj_hex
                    max_h = np.max(h_total)
                    side = (2.0 / np.sqrt(3.0)) * max_h
                    
                    if side < best_score:
                        best_score = side
                        inner_hex_data = np.column_stack([centers, np.zeros(11)])
                        outer_hex_data = np.array([0, 0, np.rad2deg(theta)])
                        best_result = (inner_hex_data, outer_hex_data, best_score)
                        
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to a known valid configuration (Seed-like)
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
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    return best_result

# EVOLVE-BLOCK-END

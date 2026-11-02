import numpy as np
import time
import math
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        # Flat-top hexagon: vertices at 0, 60, 120, 180, 240, 300 degrees
        # Pointy-top hexagon: vertices at 30, 90, 150, 210, 270, 330 degrees
        # Standard "unit hexagon" usually implies flat-top or pointy-top.
        # Let's use flat-top (vertex at 0 deg) for the base orientation.
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        # Ensure CCW order. hex_vertices generates CCW if angles increase.
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
        """
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        min_overlap = np.min(overlaps)
        
        return min_overlap

    def compute_min_side(hexes, outer_center, outer_angle_rad):
        """
        Calculate the minimum side length of a regular hexagon (given center and angle)
        that contains all given hexagons.
        """
        # The outer hexagon has center (cx, cy) and angle theta.
        # We need to find the smallest side length s such that all inner hexes are inside.
        # The supporting lines of the outer hexagon are at distance s * sqrt(3) / 2 from the center.
        # The normals of the outer hexagon are at angles: theta + 30, theta + 90, ...
        
        normal_angles = outer_angle_rad + np.radians(np.arange(30, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            # Translate hexes relative to outer center
            h_rel = h - outer_center
            proj = h_rel @ normals.T
            # The max projection onto any normal must be <= s * sqrt(3) / 2
            max_proj = max(max_proj, np.max(proj))
            
        # max_proj = s * sqrt(3) / 2 => s = max_proj * 2 / sqrt(3)
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def check_validity(hexes, tol=1e-9):
        """Check if hexagons are disjoint."""
        for i in range(len(hexes)):
            for j in range(i + 1, len(hexes)):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > tol:
                    return False
        return True

    # --- Topology Generation ---
    
    # Define the base honeycomb positions for 12 hexagons.
    # Center (0,0).
    # Ring 1: 6 hexagons at distance d1 = 2*sqrt(3) (tangent to center).
    # Ring 2: 5 hexagons at distance d2 = 4*sqrt(3) (tangent to ring 1).
    # This forms a "flower" with one petal missing.
    
    D1 = 2.0 * np.sqrt(3.0)
    D2 = 4.0 * np.sqrt(3.0)
    
    # Angles for Ring 1: 0, 60, 120, 180, 240, 300
    # Angles for Ring 2: 30, 90, 150, 210, 270 (missing 330 to make 5)
    # Wait, in a standard hex grid, the second ring positions are at angles 0, 60, ... 
    # But the distance is different.
    # Let's use axial coordinates to be precise.
    # Axial (q, r) to Cartesian (x, y) for pointy-top hexes (side 1):
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # Distance between centers: sqrt(3) * sqrt(1 + 1/4) * ... no.
    # Distance between (0,0) and (1,0) is sqrt(3).
    # But for tangent hexes (side 1), distance must be 2 * apothem = 2 * sqrt(3)/2 = sqrt(3)?
    # No, apothem is sqrt(3)/2. Distance between centers of tangent hexes is 2 * apothem = sqrt(3).
    # Wait, width of hex is 2. Height is sqrt(3).
    # If flat-top, width is 2, height is sqrt(3).
    # Tangent distance horizontally is 2. Vertically is sqrt(3).
    # So in a grid, adjacent centers are distance 2 apart horizontally?
    # Let's check.
    # Flat-top hex vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2).
    # Rightmost point: x=1. Leftmost of neighbor: x=-1+2=1.
    # So horizontal distance is 2.
    # Vertical distance?
    # Neighbor at (1, sqrt(3)).
    # Bottom of that hex: y = sqrt(3) - sqrt(3)/2 = sqrt(3)/2.
    # Top of center hex: y = sqrt(3)/2.
    # So they touch.
    # Distance: sqrt(1^2 + (sqrt(3))^2) = sqrt(1+3) = 2.
    # So the lattice spacing is 2.
    
    # Let's use a lattice with spacing 2.
    # Basis vectors:
    # u1 = (2, 0)
    # u2 = (1, sqrt(3))
    
    # Positions in lattice coordinates (i, j):
    # P = i*u1 + j*u2
    
    # 12 positions:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)  <- 6 neighbors
    # (2,0), (1,1), (0,2), (-1,2), (-2,1)  <- 5 more
    
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    u1 = np.array([2.0, 0.0])
    u2 = np.array([1.0, np.sqrt(3.0)])
    
    base_positions = np.array([i*u1 + j*u2 for i, j in lattice_coords])
    
    def generate_hexes(params):
        """
        params: [scale, shear, rotation, cx, cy, outer_angle]
        Returns list of 12 hex vertices.
        """
        scale, shear, rotation, cx, cy, outer_angle = params
        
        if scale <= 0.1:
            scale = 0.1
            
        # Affine deformation of the base positions
        # We want to allow "breathing" or shearing of the lattice.
        # Matrix M = R(theta) * A
        # A = [[scale, shear], [0, scale]]
        
        theta = rotation
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        A = np.array([
            [scale, shear],
            [0.0, scale]
        ])
        
        M = R @ A
        
        # Transform positions
        # base_positions is 12x2. We want to apply M to each row.
        # So we do base_positions @ M.T
        transformed_positions = base_positions @ M.T
        
        # Translate to center
        # We want the configuration to be centered around (0,0) for the inner hexes,
        # and the outer hexagon is at (cx, cy).
        # Actually, it's easier to fix inner hexes relative to each other,
        # and optimize the outer hexagon's position and angle.
        # But the directive says "Centroid-Minimization".
        # Let's center the inner hexes at (0,0).
        centroid = np.mean(transformed_positions, axis=0)
        transformed_positions -= centroid
        
        # Generate Hexagons
        # All inner hexagons have side 1.
        # Their angles: In a honeycomb, they are aligned.
        # So angle = 0 for all.
        # But wait, if we shear the lattice, the hexagons themselves don't rotate,
        # only their positions change. This is valid as long as they don't overlap.
        # However, if we rotate the whole configuration, the hexagons rotate too.
        # So inner hex angle = rotation.
        
        hexes = []
        for pos in transformed_positions:
            h = hex_vertices(pos[0], pos[1], 1.0, rotation)
            hexes.append(h)
            
        return hexes

    def objective(params):
        """
        Returns the side length of the outer hexagon + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = generate_hexes(params)
        
        # Check overlaps
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 100.0 # Strong penalty
            
        if penalty > 0:
            return 1e10 + penalty
            
        # Compute min side length
        cx, cy = params[3], params[4]
        outer_angle = params[5]
        s = compute_min_side(hexes, (cx, cy), outer_angle)
        
        return s

    # --- Optimization ---
    
    best_s = np.inf
    best_params = None
    best_hexes = None
    
    # Initial guesses
    # scale=1.0, shear=0, rotation=0, cx=0, cy=0, outer_angle=0
    starts = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.95, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [1.0, -0.1, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, np.pi/6, 0.0, 0.0, 0.0],
        [1.0, 0.0, -np.pi/6, 0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    
    bounds = [
        (0.5, 1.5),  # scale
        (-0.5, 0.5), # shear
        (-np.pi, np.pi), # rotation
        (-10, 10),   # cx
        (-10, 10),   # cy
        (-np.pi/6, np.pi/6) # outer_angle (symmetry)
    ]
    
    for start in starts:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x
                best_hexes = generate_hexes(res.x)
                
        except Exception:
            pass
            
    if best_params is None or best_s > 100.0:
        # Fallback
        best_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        best_hexes = generate_hexes(best_params)
        best_s = objective(best_params)
        
    # Final Polish with L-BFGS-B if time permits
    if time.time() < deadline - 10 and best_params is not None:
        try:
            res_polish = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x
                best_hexes = generate_hexes(res_polish.x)
        except Exception:
            pass

    # Verify validity of the final solution
    if best_hexes is not None and check_validity(best_hexes):
        inner = []
        for h in best_hexes:
            cx, cy = np.mean(h, axis=0)
            # Angle is best_params[2] for all
            inner.append([cx, cy, np.degrees(best_params[2])])
            
        outer_center = (best_params[3], best_params[4])
        outer_side = compute_min_side(best_hexes, outer_center, best_params[5])
        outer_angle_deg = np.degrees(best_params[5])
        
        return inner, outer_center, outer_side, outer_angle_deg
    else:
        # Fallback: Return a simple valid configuration
        # Use scale 1.2 to ensure no overlap
        fallback_params = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        fallback_hexes = generate_hexes(fallback_params)
        
        if not check_validity(fallback_hexes):
             fallback_params = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
             fallback_hexes = generate_hexes(fallback_params)
             
        inner = []
        for h in fallback_hexes:
            cx, cy = np.mean(h, axis=0)
            inner.append([cx, cy, 0.0])
            
        outer_center = (0.0, 0.0)
        outer_side = compute_min_side(fallback_hexes, outer_center, 0.0)
        outer_angle_deg = 0.0
        
        return inner, outer_center, outer_side, outer_angle_deg

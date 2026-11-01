import numpy as np

def construct():
    sqrt3 = np.sqrt(3)
    
    # Place 11 unit hexagons in a compact hexagonal cluster on triangular lattice
    # Axial coordinates for hex grid (pointy-top orientation)
    positions = [(0, 0)]
    
    # Ring 1: 6 hexagons around center
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    positions.extend(ring1)
    
    # Ring 2: add 4 more for compactness (symmetric selection)
    ring2_pick = [(2, 0), (0, 2), (-2, 2), (-2, 0)]
    positions.extend(ring2_pick)
    
    # Convert axial to Cartesian coordinates
    centers = []
    for (i, j) in positions:
        x = sqrt3 * (i + j / 2.0)
        y = 1.5 * j
        centers.append((x, y))
    
    centers_arr = np.array(centers)
    
    # Find minimum enclosing regular hexagon by optimizing rotation
    best_side = float('inf')
    best_angle = 0.0
    best_cx, best_cy = 0.0, 0.0
    
    # Try fine-grained rotations
    for angle_deg in np.arange(0.0, 180.0, 0.5):
        angle = np.radians(angle_deg)
        
        # 6 outward face normals of the enclosing hexagon
        normals = np.array([
            [np.cos(angle + k * np.pi / 3), np.sin(angle + k * np.pi / 3)]
            for k in range(6)
        ])
        
        # Project all centers onto each normal direction
        projs = centers_arr @ normals.T  # (11, 6)
        
        # Spread in each direction
        spreads = projs.max(axis=0) - projs.min(axis=0)  # (6,)
        
        # Required inradius = half the maximum spread
        r = spreads.max() / 2.0
        side = 2.0 * r / sqrt3
        
        if side < best_side:
            best_side = side
            best_angle = angle_deg
            # Compute optimal center: midpoint in the critical direction
            k_crit = int(np.argmax(spreads))
            proj_vals = projs[:, k_crit]
            c_proj = (proj_vals.max() + proj_vals.min()) / 2.0
            # The center in the critical normal direction is c_proj
            # For other directions, use centroid as approximation
            cx0, cy0 = centers_arr.mean(axis=0)
            best_cx, best_cy = cx0, cy0
    
    # Build inner hexagon list: (x, y, angle_degrees)
    inner = []
    for (x, y) in centers:
        inner.append([float(x), float(y), 0.0])
    
    return inner, (float(best_cx), float(best_cy)), float(best_side), float(best_angle)

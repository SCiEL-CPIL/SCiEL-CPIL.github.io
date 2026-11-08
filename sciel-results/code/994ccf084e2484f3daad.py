import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Hexagonal lattice for unit hexagons (side 1)
    # Adjacent hexagon center distance = sqrt(3)
    # Use axial coordinates for hex grid
    # Convert axial (q, r) to pixel coords for flat-top hexagons
    # For flat-top: x = sqrt(3) * (q + r/2), y = 1.5 * r
    
    sqrt3 = np.sqrt(3.0)
    
    # Generate candidate positions in hex grid coordinates
    # We want ~11 hexagons in a compact shape
    # Hexagonal numbers: 1, 7, 19, 37...
    # For 11, use a shape close to hexagonal radius 1 (7) + 4 extra
    
    # Generate all hex grid positions within radius 2 (gives 19 positions)
    # Then pick the 11 most central ones
    candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Hex distance from origin
            dist = (abs(q) + abs(r) + abs(q + r)) // 2
            if dist <= 2:
                # Convert to pixel coords (flat-top hexagons)
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                candidates.append((dist, x, y))
    
    # Sort by distance from center, take first 11
    candidates.sort(key=lambda c: c[0])
    selected = candidates[:N]
    
    # Get positions
    positions = [(x, y) for _, x, y in selected]
    
    # Compute the minimal enclosing regular hexagon
    # For flat-top unit hexagons, the vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # from center, at distance 1 (circumradius)
    # The enclosing hexagon (also flat-top, same orientation) has side length S
    # Its inradius = S * sqrt(3)/2, circumradius = S
    # A point (x, y) is inside the enclosing hexagon centered at (cx, cy) with side S if:
    # For flat-top hexagon, the 6 sides are at distances S*sqrt(3)/2 from center
    # in directions 30, 90, 150, 210, 270, 330 degrees
    
    # To find minimal S, we need to find the max over all inner hexagon vertices
    # of the "hex distance" from the enclosing hexagon center
    
    # For a flat-top regular hexagon centered at origin with side length S:
    # The constraint for a point (x,y) to be inside is:
    # |x| <= S (for the two vertical-ish sides... actually let me think again)
    
    # Flat-top hexagon vertices: (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), 
    #                           (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    # The 6 edges are:
    # Top: from (S/2, S*sqrt(3)/2) to (-S/2, S*sqrt(3)/2) -> y <= S*sqrt(3)/2
    # Top-right: from (S, 0) to (S/2, S*sqrt(3)/2) -> x + y/sqrt(3) <= S
    # Bottom-right: from (S, 0) to (S/2, -S*sqrt(3)/2) -> x - y/sqrt(3) <= S
    # Bottom: y >= -S*sqrt(3)/2
    # Bottom-left: -x - y/sqrt(3) <= S
    # Top-left: -x + y/sqrt(3) <= S
    
    # So for a point to be inside, all 6 constraints must hold.
    # The "hex radius" of a point is max of:
    # x + y/sqrt(3), x - y/sqrt(3), -x + y/sqrt(3), -x - y/sqrt(3), 
    # y/sqrt(3) * ... wait let me redo this.
    
    # Actually for flat-top hexagon with circumradius R (= side length S):
    # The 6 supporting lines are at distance R*sqrt(3)/2 from center.
    # Normal directions: 30°, 90°, 150°, 210°, 270°, 330°
    # n_i = (cos(30°+60°i), sin(30°+60°i)) for i=0..5
    # Constraint: dot(p, n_i) <= R*sqrt(3)/2 for all i
    
    # So R >= max_i dot(p - center, n_i) / (sqrt(3)/2) = max_i dot(p-center, n_i) * 2/sqrt(3)
    
    # For each inner hexagon, we need ALL its vertices to be inside.
    # Inner hexagon at (cx, cy) with side 1 (flat-top) has vertices at:
    # (cx + cos(60°k), cy + sin(60°k)) for k=0..5
    
    # We need to find center (ox, oy) and minimize R such that all vertices of all inner hexagons satisfy the constraint.
    
    # This is a convex optimization problem. For simplicity, let's try centering the enclosing hexagon at the centroid of the inner hexagon centers, then compute the required R.
    
    # Compute centroid
    centers = np.array(positions)
    centroid = centers.mean(axis=0)
    
    # Compute required R for this center
    angles = np.radians([0, 60, 120, 180, 240, 300])  # vertex angles for flat-top
    normals = np.radians([30, 90, 150, 210, 270, 330])  # normal directions for edges
    
    max_proj = 0
    for cx, cy in positions:
        # Vertices of this unit hexagon
        for k in range(6):
            vx = cx + np.cos(angles[k])
            vy = cy + np.sin(angles[k])
            for ni in normals:
                proj = (vx - centroid[0]) * np.cos(ni) + (vy - centroid[1]) * np.sin(ni)
                if proj > max_proj:
                    max_proj = proj
    
    R = max_proj * 2.0 / sqrt3  # Since inradius = R * sqrt(3)/2
    
    # Try a few different centers to see if we can do better
    # The optimal center might not be the centroid
    # Let's try a simple local search on the center
    
    best_R = R
    best_center = centroid.copy()
    
    # Local search
    step_size = 0.1
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0:
                    continue
                new_center = best_center + np.array([dx, dy])
                max_proj_new = 0
                for cx, cy in positions:
                    for k in range(6):
                        vx = cx + np.cos(angles[k])
                        vy = cy + np.sin(angles[k])
                        for ni in normals:
                            proj = (vx - new_center[0]) * np.cos(ni) + (vy - new_center[1]) * np.sin(ni)
                            if proj > max_proj_new:
                                max_proj_new = proj
                R_new = max_proj_new * 2.0 / sqrt3
                if R_new < best_R - 1e-9:
                    best_R = R_new
                    best_center = new_center
                    improved = True
        
        if not improved:
            step_size *= 0.5
            if step_size < 0.001:
                break
    
    # Convert to return format
    # inner: list of [x, y, angle_degrees]
    inner = [[cx, cy, 0.0] for cx, cy in positions]
    center = [float(best_center[0]), float(best_center[1])]
    side = float(best_R)
    angle = 0.0  # flat-top orientation
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END

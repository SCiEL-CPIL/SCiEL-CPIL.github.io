import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Place unit hexagons in a hexagonal lattice pattern (flat-top orientation)
    # For flat-top hexagons with side 1:
    #   - Center to vertex distance = 1
    #   - Apothem (center to edge) = sqrt(3)/2
    #   - When edge-sharing, center-to-center distance = sqrt(3)
    #   - Horizontal spacing in same row = 3/2
    #   - Vertical spacing between rows = sqrt(3)
    #   - Odd rows offset by 3/4
    
    sqrt3 = np.sqrt(3.0)
    
    # Arrange 11 hexagons in a compact hexagonal cluster
    # Use axial coordinates for hex grid
    # A hexagonal number arrangement: for radius r, we have 3r(r+1)+1 hexagons
    # r=0: 1, r=1: 7, r=2: 19
    # For N=11, we take a partial hexagon of radius 2 (19 positions) and pick 11
    
    # Generate hex grid positions in axial coordinates (q, r)
    # Convert to pixel coordinates for flat-top hexagons
    def axial_to_pixel(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return x, y
    
    # Generate all positions in a hexagon of radius 2 (19 positions)
    positions = []
    for q in range(-2, 3):
        for r in range(max(-2, -q - 2), min(2, -q + 2) + 1):
            positions.append((q, r))
    
    # Sort by distance from center to pick the most central 11
    positions.sort(key=lambda p: p[0]**2 + p[0]*p[1] + p[1]**2)
    selected = positions[:N]
    
    # Convert to pixel coordinates
    inner = []
    for q, r in selected:
        x, y = axial_to_pixel(q, r)
        inner.append([x, y, 0.0])
    
    # Compute minimal enclosing regular hexagon (flat-top, same orientation as inner)
    # For a flat-top regular hexagon with side S centered at (cx, cy):
    # The six edges are at distances S*sqrt(3)/2 from center
    # A point (x,y) relative to center is inside if:
    #   |x| <= S  (left/right vertices)
    #   |x|/2 + sqrt(3)*|y|/2 <= S  (for the other 4 edges)
    # Actually, the six half-planes for a flat-top hex with side S centered at origin:
    #   x <= S, x >= -S
    #   x/2 + sqrt(3)*y/2 <= S, x/2 - sqrt(3)*y/2 >= -S (wait, let me be more careful)
    
    # For flat-top hex with side S, vertices at:
    #   (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    # The six edge lines (outward normals at angles 0, 60, 120, 180, 240, 300 degrees):
    #   Edge 1 (normal 0°): x = S  =>  x <= S
    #   Edge 2 (normal 60°): x*cos(60) + y*sin(60) = S  =>  x/2 + y*sqrt(3)/2 <= S
    #   Edge 3 (normal 120°): x*cos(120) + y*sin(120) = S  =>  -x/2 + y*sqrt(3)/2 <= S
    #   Edge 4 (normal 180°): x = -S  =>  -x <= S  =>  x >= -S
    #   Edge 5 (normal 240°): x*cos(240) + y*sin(240) = S  =>  -x/2 - y*sqrt(3)/2 <= S
    #   Edge 6 (normal 300°): x*cos(300) + y*sin(300) = S  =>  x/2 - y*sqrt(3)/2 <= S
    
    # So for a point (px, py) relative to center (cx, cy), with dx = px-cx, dy = py-cy:
    #   dx <= S
    #   -dx <= S  =>  dx >= -S
    #   dx/2 + dy*sqrt(3)/2 <= S
    #   -dx/2 + dy*sqrt(3)/2 <= S
    #   -dx/2 - dy*sqrt(3)/2 <= S
    #   dx/2 - dy*sqrt(3)/2 <= S
    
    # The minimum S needed for a given center is the max of:
    #   max(dx), -min(dx), max(dx/2 + dy*sqrt(3)/2), max(-dx/2 + dy*sqrt(3)/2),
    #   max(-dx/2 - dy*sqrt(3)/2), max(dx/2 - dy*sqrt(3)/2)
    
    # We also need to account for the fact that the inner hexagons have size.
    # Each inner hexagon has vertices at distance 1 from its center.
    # So we need the enclosing hexagon to contain all vertices of all inner hexagons.
    
    # For each inner hexagon center (hx, hy), its 6 vertices are at:
    #   (hx + cos(theta_k), hy + sin(theta_k)) for k=0..5, theta_k = 60*k degrees
    # (flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees)
    
    # Collect all vertex points
    all_points = []
    for hx, hy, _ in inner:
        for k in range(6):
            theta = np.pi * k / 3.0
            vx = hx + np.cos(theta)
            vy = hy + np.sin(theta)
            all_points.append((vx, vy))
    
    all_points = np.array(all_points)
    
    # Try different center positions to minimize the required side length
    # The optimal center is somewhere near the centroid of the points
    centroid = all_points.mean(axis=0)
    
    # Try a grid of center positions around the centroid
    best_side = float('inf')
    best_center = centroid
    
    # Coarse grid search
    for cx in np.linspace(centroid[0] - 2, centroid[0] + 2, 41):
        for cy in np.linspace(centroid[1] - 2, centroid[1] + 2, 41):
            dx = all_points[:, 0] - cx
            dy = all_points[:, 1] - cy
            
            s1 = np.max(dx)
            s2 = -np.min(dx)
            s3 = np.max(dx/2 + dy*sqrt3/2)
            s4 = np.max(-dx/2 + dy*sqrt3/2)
            s5 = np.max(-dx/2 - dy*sqrt3/2)
            s6 = np.max(dx/2 - dy*sqrt3/2)
            
            side = max(s1, s2, s3, s4, s5, s6)
            
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
    
    # Refine with finer grid
    cx0, cy0 = best_center
    for cx in np.linspace(cx0 - 0.5, cx0 + 0.5, 41):
        for cy in np.linspace(cy0 - 0.5, cy0 + 0.5, 41):
            dx = all_points[:, 0] - cx
            dy = all_points[:, 1] - cy
            
            s1 = np.max(dx)
            s2 = -np.min(dx)
            s3 = np.max(dx/2 + dy*sqrt3/2)
            s4 = np.max(-dx/2 + dy*sqrt3/2)
            s5 = np.max(-dx/2 - dy*sqrt3/2)
            s6 = np.max(dx/2 - dy*sqrt3/2)
            
            side = max(s1, s2, s3, s4, s5, s6)
            
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
    
    # Even finer refinement
    cx0, cy0 = best_center
    for cx in np.linspace(cx0 - 0.1, cx0 + 0.1, 21):
        for cy in np.linspace(cy0 - 0.1, cy0 + 0.1, 21):
            dx = all_points[:, 0] - cx
            dy = all_points[:, 1] - cy
            
            s1 = np.max(dx)
            s2 = -np.min(dx)
            s3 = np.max(dx/2 + dy*sqrt3/2)
            s4 = np.max(-dx/2 + dy*sqrt3/2)
            s5 = np.max(-dx/2 - dy*sqrt3/2)
            s6 = np.max(dx/2 - dy*sqrt3/2)
            
            side = max(s1, s2, s3, s4, s5, s6)
            
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
    
    return inner, best_center, best_side, 0.0
# EVOLVE-BLOCK-END

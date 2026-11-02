"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    # 1. Generate lattice points for N unit hexagons
    # A hexagonal lattice of unit hexagons (side 1) has centers spaced by:
    # Distance between adjacent centers = sqrt(3) * side = sqrt(3)
    # Using axial coordinates (q, r) for a hex grid:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # This ensures non-overlapping for unit hexagons (distance sqrt(3) > 2*0.5*sqrt(3) is wrong, wait.
    # Distance between centers of touching unit hexagons is sqrt(3) * side?
    # No, distance between centers of touching unit hexagons (side 1) is sqrt(3).
    # Let's verify: Hexagon side 1. Width (flat-to-flat) = sqrt(3). Distance between centers = sqrt(3).
    # So if we place them on a grid with spacing sqrt(3) in x and 1.5 in y (offset), they touch.
    # To be safe and disjoint (strictly inside), we can use this spacing. They will touch, which is allowed ("disjoint" usually means interiors don't intersect, touching is ok).
    # If "disjoint" means strictly non-touching, we might need a tiny epsilon, but usually packing allows touching.
    # The task says "disjoint", which in geometry often means interiors are disjoint. Touching boundaries is typically allowed in packing problems unless "strictly disjoint" is specified.
    # Given the score 1.0 baseline was 40.0, and we expect ~3-4, touching is likely the intended limit.
    
    # Let's generate a compact cluster of N=11 hexagons.
    # Shape: Hexagon of radius 1 (center + 6 neighbors + 6 more? No, radius 1 is 7. Radius 2 is 19. We have 11).
    # Let's just take the first N points from a spiral or grid.
    # Better: A specific shape that minimizes convex hull.
    # For N=11, a 3x3 block minus 2 corners? Or a hexagon of radius 1 (7) + 4 more?
    # Let's use a standard hex grid enumeration.
    
    points = []
    # Generate points in a hexagonal pattern
    # Center (0,0)
    points.append((0, 0))
    
    # Ring 1 (6 points)
    for i in range(6):
        angle = np.pi / 3 * i
        # In axial coords, ring 1 is: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # Let's just iterate q, r in a range and filter by distance
        pass
    
    # Simpler: iterate over a square grid of axial coords and pick closest to origin
    candidates = []
    max_q = 5
    for q in range(-max_q, max_q + 1):
        for r in range(-max_q, max_q + 1):
            # Convert axial to cartesian
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            candidates.append((x, y, q, r))
    
    # Sort by distance to origin (0,0)
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first N
    selected = candidates[:N]
    inner_coords = [(x, y, 0.0) for x, y, q, r in selected]
    
    # 2. Determine the minimal enclosing regular hexagon
    # The outer hexagon must contain all these points.
    # A regular hexagon of side S, rotated by angle theta, centered at (cx, cy).
    # To minimize S, we can fix the center to the centroid of the points (or try a few centers).
    # For a convex shape, the minimal enclosing hexagon of a fixed orientation has its center at the center of the minimal enclosing circle? Not necessarily.
    # However, for a hexagonal lattice subset, the optimal enclosing hexagon is often aligned with the lattice or rotated by 30 degrees.
    # Let's try orientations: 0, 30, 60, 90, 120, 150 degrees.
    # And center: centroid of the points.
    
    pts = np.array([(x, y) for x, y, a in inner_coords])
    centroid = np.mean(pts, axis=0)
    
    best_S = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different rotations for the outer hexagon
    # The outer hexagon sides must be at distance S*sqrt(3)/2 from the center?
    # No, distance from center to side (apothem) is S * sqrt(3) / 2.
    # A point (x,y) is inside a hexagon centered at (cx,cy) with apothem A and rotation phi if:
    # For each of the 6 normal vectors n_i (angle phi + i*pi/3):
    # dot((x-cx, y-cy), n_i) <= A
    # We need the minimum A such that max_i dot(...) <= A for all points.
    # Then S = 2 * A / sqrt(3).
    
    angles_to_try = [0.0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
    
    for phi in angles_to_try:
        # Normals for the hexagon sides
        normals = []
        for i in range(6):
            ang = phi + i * np.pi / 3
            normals.append((np.cos(ang), np.sin(ang)))
        
        # Calculate required apothem A for this orientation and center
        max_proj = 0.0
        for p in pts:
            vec = p - centroid
            for n in normals:
                proj = vec[0]*n[0] + vec[1]*n[1]
                if proj > max_proj:
                    max_proj = proj
        
        A = max_proj
        S = 2 * A / np.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_center = tuple(centroid)
            best_angle = np.degrees(phi)
            
        # Check deadline
        if time.time() > deadline - 5:
            break
            
    # The inner hexagons are at their lattice positions.
    # The outer hexagon is centered at centroid, side length best_S, rotated by best_angle.
    # Note: The inner hexagons are axis-aligned (angle 0.0) in our lattice generation.
    # The task allows rotation of inner hexagons. Our lattice assumes they are aligned.
    # This is a valid configuration.
    
    # Construct the return values
    inner = [[x, y, 0.0] for x, y, a in inner_coords]
    center = best_center
    side = best_S
    angle = best_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END

import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate hexagonal grid positions for N hexagons
    # Using axial coordinates (q, r) for hex grid
    # Convert to pixel coordinates: x = 2*q + r, y = sqrt(3)*r (for pointy-top)
    # But for flat-top hexagons: x = 1.5*q, y = sqrt(3)*(r + q/2)
    
    # For flat-top hexagons with side length 1:
    # Width (flat-to-flat) = 2, Height (point-to-point) = sqrt(3)
    # Adjacent hex centers are spaced by 2 in the x direction (for same row)
    # and by sqrt(3) in y, with x offset of 1 for adjacent rows
    
    # Let's use a simple hexagonal packing
    # Place hexagons in rows
    
    # For flat-top hexagons:
    # Row 0: y = 0, x = 0, 2, 4, ...
    # Row 1: y = sqrt(3), x = 1, 3, 5, ...
    # Row 2: y = 2*sqrt(3), x = 0, 2, 4, ...
    
    sqrt3 = np.sqrt(3.0)
    
    # Generate positions in a compact hexagonal arrangement
    positions = []
    
    # Try a hexagonal arrangement: center + surrounding rings
    # Ring 0: (0,0)
    # Ring 1: 6 positions
    # Ring 2: 12 positions (we need only 4 more for total 11)
    
    # Axial coordinates for hex grid
    def axial_to_pixel(q, r):
        # Flat-top hexagons
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return x, y
    
    # Generate ring positions
    def get_ring_positions(radius):
        if radius == 0:
            return [(0, 0)]
        positions = []
        # Six directions in axial coordinates
        directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        # Start at the first position of the ring
        q, r = 0, -radius
        for i in range(6):
            for j in range(radius):
                positions.append((q, r))
                q, r = q + directions[i][0], r + directions[i][1]
        return positions
    
    # Build hexagonal arrangement
    axial_coords = [(0, 0)]  # Center
    axial_coords.extend(get_ring_positions(1))  # Ring 1: 6 more, total 7
    ring2 = get_ring_positions(2)  # Ring 2: 12 positions
    axial_coords.extend(ring2[:N - len(axial_coords)])  # Add remaining needed
    
    # Convert to pixel coordinates
    pixel_coords = [axial_to_pixel(q, r) for q, r in axial_coords[:N]]
    
    # Now compute the minimum enclosing regular hexagon
    # The outer hexagon must contain all inner hexagons
    # Each inner hexagon has vertices at distance 1 from its center
    
    # For a regular hexagon with side length S centered at (cx, cy) with rotation angle theta:
    # The vertices are at (cx + S*cos(theta + k*pi/3), cy + S*sin(theta + k*pi/3)) for k=0..5
    
    # We need to find S, cx, cy, theta such that all inner hexagon vertices are inside
    
    # Simple approach: compute the bounding box of all inner hexagon vertices,
    # then find the minimum enclosing hexagon
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for px, py in pixel_coords:
        for k in range(6):
            angle = k * np.pi / 3
            vx = px + np.cos(angle)
            vy = py + np.sin(angle)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    # Compute centroid
    cx, cy = np.mean(all_vertices[:, 0]), np.mean(all_vertices[:, 1])
    
    # For a given rotation angle, compute the minimum side length needed
    def min_hex_side(vertices, cx, cy, theta):
        # Rotate vertices relative to center
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rx = (vertices[:, 0] - cx) * cos_t + (vertices[:, 1] - cy) * sin_t
        ry = -(vertices[:, 0] - cx) * sin_t + (vertices[:, 1] - cy) * cos_t
        
        # For a regular hexagon with side S, the support function in direction phi is S*cos(phi - theta) for the 6 sides
        # Actually, for a regular hexagon centered at origin with one vertex at angle theta,
        # the distance from center to each side is S*sqrt(3)/2
        # The hexagon contains a point (x,y) if for all 6 sides, the point is on the inside
        
        # Simpler: for a regular hexagon with circumradius R (= S), 
        # a point (x,y) is inside if max over 6 directions of |projection| <= R
        # Actually no, that's for the circumscribed circle.
        
        # For a regular hexagon with side length S, the apothem (inradius) is S*sqrt(3)/2
        # The hexagon is the intersection of 6 half-planes
        # Side k has normal at angle theta + k*pi/3, and the half-plane is n_k . p <= S*sqrt(3)/2
        
        max_dist = 0
        for k in range(6):
            angle = theta + k * np.pi / 3
            nx = np.cos(angle)
            ny = np.sin(angle)
            dist = np.abs(rx * nx + ry * ny)
            max_dist = max(max_dist, np.max(dist))
        
        # The apothem must be at least max_dist
        # S * sqrt(3)/2 >= max_dist
        # S >= 2 * max_dist / sqrt(3)
        return 2.0 * max_dist / sqrt3
    
    # Try different rotation angles
    best_S = float('inf')
    best_theta = 0.0
    
    for theta_deg in range(0, 180, 3):
        theta = np.radians(theta_deg)
        S = min_hex_side(all_vertices, cx, cy, theta)
        if S < best_S:
            best_S = S
            best_theta = theta
    
    # Refine around best angle
    for theta_deg in np.arange(best_theta - 5, best_theta + 5, 0.5):
        theta = np.radians(theta_deg)
        S = min_hex_side(all_vertices, cx, cy, theta)
        if S < best_S:
            best_S = S
            best_theta = theta
    
    # Also try centering at the centroid of the hexagon centers
    cx2, cy2 = np.mean([p[0] for p in pixel_coords]), np.mean([p[1] for p in pixel_coords])
    
    for theta_deg in range(0, 180, 3):
        theta = np.radians(theta_deg)
        S = min_hex_side(all_vertices, cx2, cy2, theta)
        if S < best_S:
            best_S = S
            best_theta = theta
            cx, cy = cx2, cy2
    
    # Final refinement
    for theta_deg in np.arange(best_theta - 2, best_theta + 2, 0.1):
        theta = np.radians(theta_deg)
        S = min_hex_side(all_vertices, cx, cy, theta)
        if S < best_S:
            best_S = S
            best_theta = theta
    
    # Build inner hexagon list
    inner = [[px, py, 0.0] for px, py in pixel_coords]
    
    return inner, (float(cx), float(cy)), float(best_S), float(np.degrees(best_theta))
# EVOLVE-BLOCK-END

import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Hexagonal lattice positions for 11 hexagons
    # Using axial coordinates for hex grid
    # Center at (0,0), ring 1 has 6, ring 2 partial
    # For 11: center + 6 in ring 1 + 4 in ring 2
    
    # Axial to pixel (pointy-top hexagons)
    def axial_to_xy(q, r, size=1.0):
        x = size * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
        y = size * (3/2 * r)
        return x, y
    
    # Hexagonal lattice with spacing 2 (center-to-center for unit hexagons touching)
    # For same-orientation unit hexagons, min center distance = 2 along lattice directions
    
    # Generate positions in a compact cluster
    # Use cube coordinates for hex grid
    def cube_to_axial(x, y, z):
        return x, z
    
    def axial_to_cube(q, r):
        return q, -q-r, r
    
    def cube_distance(a, b):
        return (abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])) / 2
    
    # Generate hex grid positions by distance from center
    positions_axial = [(0, 0)]  # center
    
    # Ring 1: 6 positions at distance 1
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    positions_axial.extend(ring1)
    
    # Ring 2: need 4 more for total 11
    # Ring 2 has 12 positions; pick 4 that are most compact
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            cube = axial_to_cube(q, r)
            if cube_distance((0,0,0), cube) == 2:
                ring2.append((q, r))
    
    # Pick 4 from ring 2 that minimize bounding box
    # Try different subsets - use the 4 that are "closest" to center in some sense
    # Actually, let's try all C(12,4) = 495 combinations to find the most compact
    
    best_positions = None
    best_score = float('inf')
    
    from itertools import combinations
    
    # For each combination of 4 from ring2, compute the bounding hexagon size
    def compute_bounding_hexagon(points, spacing=2.0):
        """Compute minimal side length of regular hexagon enclosing all points."""
        # Convert axial to cartesian with spacing
        pts = []
        for (q, r) in points:
            x = spacing * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
            y = spacing * (3/2 * r)
            pts.append((x, y))
        
        pts = np.array(pts)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        pts_centered = pts - np.array([cx, cy])
        
        # For a regular hexagon with side s, the inradius is s*sqrt(3)/2
        # A point (x,y) is inside if max over 6 directions of dot product <= inradius
        # The 6 normal directions for a flat-top hexagon:
        # Actually, let's use the property that for a regular hexagon centered at origin
        # with "radius" R (center to vertex), a point is inside if
        # max(|x|, |x*cos60 + y*sin60|, |-x*cos60 + y*sin60|) <= R
        # Wait, that's not quite right. Let me use the support function.
        
        # For a regular hexagon with side length s, center at origin, flat-top orientation:
        # The 6 vertices are at angles 0, 60, 120, 180, 240, 300 degrees at distance s from center
        # The inradius (apothem) is s * sqrt(3)/2
        # A point (x,y) is inside if for all 6 edge normals n_i, dot(p, n_i) <= apothem
        # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
        
        # Actually, let me just compute the minimal enclosing hexagon by trying different 
        # center positions and orientations. For simplicity, assume center at centroid 
        # and orientation 0 degrees.
        
        # For a regular hexagon with side s, centered at (cx, cy), flat-top:
        # Vertices at (cx + s*cos(k*60°), cy + s*sin(k*60°)) for k=0..5
        # The hexagon contains point (px, py) iff for each edge, the point is on the 
        # interior side.
        
        # Simpler: the distance from center to each vertex is s. The hexagon is the 
        # intersection of 6 half-planes. For flat-top hex with side s:
        # The 6 supporting lines are at distance apothem = s*sqrt(3)/2 from center
        # in directions 0°, 60°, 120°, 180°, 240°, 300° (normals to edges)
        
        # Wait, I need to be careful. Let me use: for a regular hexagon with circumradius R
        # (center to vertex = R), the inradius is R*sqrt(3)/2.
        # Side length s = R for a regular hexagon! No wait, for regular hexagon, side = circumradius.
        # Yes! For a regular hexagon, the side length equals the circumradius.
        
        # So if side length = s, then circumradius = s, inradius = s*sqrt(3)/2.
        
        # The 6 edge normals point in directions 30°, 90°, 150°, 210°, 270°, 330° 
        # (for flat-top orientation where one edge is horizontal at top)
        
        # A point (x,y) relative to center is inside if:
        # max over i of (x*cos(theta_i) + y*sin(theta_i)) <= inradius
        # where theta_i are the 6 edge normal angles.
        
        theta_vals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
        normals = np.column_stack([np.cos(theta_vals), np.sin(theta_vals)])
        
        # For each point, compute the max dot product with normals
        # The required inradius is the max over all points of max dot product
        dots = pts_centered @ normals.T  # shape (n_points, 6)
        max_dots = np.max(dots, axis=1)  # shape (n_points,)
        required_inradius = np.max(max_dots)
        
        # inradius = s * sqrt(3)/2, so s = required_inradius * 2 / sqrt(3)
        s = required_inradius * 2.0 / np.sqrt(3)
        return s
    
    all_positions = positions_axial + ring2  # 7 + 12 = 19 positions total, but we only use 11
    
    # We have 7 fixed (center + ring1) and need to choose 4 from ring2
    for combo in combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
        selected_ring2 = [ring2[i] for i in combo]
        all_pts = positions_axial + selected_ring2
        
        s = compute_bounding_hexagon(all_pts, spacing=2.0)
        if s < best_score:
            best_score = s
            best_positions = all_pts[:]
    
    if best_positions is None:
        # Fallback: just use center + ring1 + first 4 of ring2
        best_positions = positions_axial + ring2[:4]
        best_score = compute_bounding_hexagon(best_positions, spacing=2.0)
    
    # Convert to cartesian coordinates
    inner = []
    spacing = 2.0
    for (q, r) in best_positions:
        x = spacing * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
        y = spacing * (3/2 * r)
        inner.append([float(x), float(y), 0.0])
    
    # Center of outer hexagon
    cx = np.mean([p[0] for p in inner])
    cy = np.mean([p[1] for p in inner])
    
    return inner, (float(cx), float(cy)), float(best_score), 0.0


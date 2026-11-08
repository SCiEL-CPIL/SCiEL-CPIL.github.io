"""
Packing unit regular hexagons inside a regular hexagon.

Strategy:
  1. Place unit hexagons on a honeycomb lattice (two offset rows per "layer").
  2. For a given number of hexagons n, enumerate hexagonal-shaped clusters
     centered at the origin (all hexagons within a given hexagonal radius r
     in the axial coordinate system).
  3. For each candidate cluster, compute the vertices of each unit hexagon,
     then find the minimal regular hexagon (axis-aligned, rotation 0) that
     encloses all vertices. The outer hexagon is centered at the origin.
  4. Return the configuration with the smallest outer side length.

  A unit regular hexagon with vertices at distance 1 from its center has
  vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top) or
  30, 90, 150, 210, 270, 330 degrees (pointy-top).

  For a flat-top hexagon centered at origin with side length s:
    vertices at (s*cos(60*k), s*sin(60*k)) for k=0..5

  The honeycomb lattice of flat-top unit hexagons:
    - Row spacing (vertical): sqrt(3)
    - Column spacing (horizontal): 1.5
    - Offset for odd rows: 0.75 in x (or use axial coordinates)

  Actually, let's use the standard axial coordinate system for hex grids.
  For a flat-top hexagon of side 1:
    - Horizontal distance between adjacent hex centers in the same row: 1.5
    - Vertical distance between rows: sqrt(3)
    - Odd rows offset by 0.75 in x

  Axial coordinates (q, r) to pixel (x, y) for flat-top:
    x = 1.5 * q
    y = sqrt(3) * (r + q/2)

  A hexagonal cluster of radius R (in axial coords) contains all hexagons
  with max(|q|, |r|, |q+r|) <= R.
  The count is 1 + 3*R*(R+1).

  For n=11, R=1 gives 7, R=2 gives 19. So n=11 doesn't fit a perfect
  hexagonal cluster. We need to find the best arrangement.

  Alternative approach: For small n, just try all hexagonal clusters up to
  radius R and pick the smallest that contains at least n hexagons, then
  remove the farthest ones to get exactly n.

  But actually, the problem says "n unit hexagons" - looking at the seed,
  N=11 and it places 11 hexagons in a row. So we need to pack exactly 11.

  Let me think about what's optimal for 11 unit hexagons.

  For a hexagonal cluster of radius 1 (7 hexagons), we can add 4 more.
  The 7-hex cluster has outer side length... let me compute.

  Actually, let me just implement a general approach:
  1. Generate hexagonal cluster of radius R.
  2. If we need fewer hexagons, remove the outermost ones (those at maximum
     axial distance) to reduce count.
  3. Compute the enclosing hexagon.

  The enclosing regular hexagon for a set of points:
  - A regular hexagon with flat-top orientation has 6 sides.
  - Side 0 (top): y = c_y + s (horizontal line)
  - Side 1 (upper-right): -sqrt(3)*x + y = c_x*(-sqrt(3)) + c_y + s*(1+sqrt(3)) ... 
  
  Actually, let me think about this differently. A regular hexagon centered at
  (cx, cy) with side length s and flat-top orientation has vertices at:
    (cx + s*cos(theta), cy + s*sin(theta)) for theta = 0, 60, 120, 180, 240, 300

  The 6 half-planes (interior of hexagon) are:
    For each vertex direction, the edge is perpendicular.
    Edge between vertex k and k+1:
    The normal points outward.

  For a flat-top hexagon centered at origin with side s:
    Vertices: (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    
    Edges and outward normals:
    Edge 0 (right): from (s,0) to (s/2, s*sqrt(3)/2). Normal: (sqrt(3), -1)/2 ... 
    This is getting complex. Let me use a simpler approach.

  A regular hexagon can be defined as the intersection of 6 half-planes.
  For a hexagon centered at (cx,cy) with circumradius R_circ = s (for regular hexagon, side = circumradius):
  
  The 6 supporting lines (outward normal directions at 0, 60, 120, 180, 240, 300):
    n_k . (p - c) <= R_circ * cos(30 deg) = R_circ * sqrt(3)/2
    
  Wait, for a regular hexagon with side length s, the apothem (distance from center to edge) is s*sqrt(3)/2.
  The circumradius (distance from center to vertex) is s.

  The 6 edge normals are at angles 30, 90, 150, 210, 270, 330 degrees for flat-top.
  Actually for flat-top, the flat sides are at top and bottom, so normals are vertical:
    Top edge normal: (0, 1)
    Bottom edge normal: (0, -1)
    Upper-right edge normal: (sqrt(3)/2, 1/2)
    Lower-right edge normal: (sqrt(3)/2, -1/2)
    Upper-left edge normal: (-sqrt(3)/2, 1/2)
    Lower-left edge normal: (-sqrt(3)/2, -1/2)

  For each point p in the set, and for each normal n_k:
    n_k . (p - c) <= apothem = s * sqrt(3)/2

  So s >= 2 * max_k |n_k . (p - c)| / sqrt(3) for all points p.

  Therefore, the minimum s for a given center c is:
    s = 2 * max_p max_k |n_k . (p - c)| / sqrt(3)

  We can minimize over c as well. For simplicity, let's first try c = (0,0)
  since our cluster is centered at origin.

  Let me implement this.
"""

import math
import time

def construct():
    t_start = time.time()
    DEADLINE = 110.0  # seconds, safety margin before 120s limit
    
    n = 11  # number of unit hexagons to pack
    
    sqrt3 = math.sqrt(3.0)
    
    # Axial to pixel for flat-top hexagons with side 1
    def axial_to_pixel(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y
    
    # Generate hexagonal cluster of radius R
    def hex_cluster(R):
        cells = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                if abs(q + r) <= R:
                    cells.append((q, r))
        return cells
    
    # Get all vertices of a unit flat-top hexagon centered at (cx, cy)
    def hex_vertices(cx, cy, s=1.0):
        verts = []
        for k in range(6):
            angle = math.radians(60.0 * k)
            verts.append((cx + s * math.cos(angle), cy + s * math.sin(angle)))
        return verts
    
    # Compute minimum enclosing hexagon side length for a set of points,
    # with hexagon centered at (cx, cy), flat-top orientation
    def min_hex_side(points, cx=0.0, cy=0.0):
        # 6 outward normals for flat-top hexagon
        normals = [
            (0.0, 1.0),        # top
            (sqrt3/2, 0.5),    # upper-right
            (sqrt3/2, -0.5),   # lower-right
            (0.0, -1.0),       # bottom
            (-sqrt3/2, -0.5),  # lower-left
            (-sqrt3/2, 0.5),   # upper-left
        ]
        max_proj = 0.0
        for (px, py) in points:
            for (nx, ny) in normals:
                proj = nx * (px - cx) + ny * (py - cy)
                if abs(proj) > max_proj:
                    max_proj = abs(proj)
        # apothem = s * sqrt(3)/2, so s = 2 * apothem / sqrt(3)
        return 2.0 * max_proj / sqrt3
    
    # Try different hexagonal cluster sizes
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try clusters of radius 1 (7 hex), 2 (19 hex), etc.
    # For n=11, radius 1 gives 7 (too few), radius 2 gives 19 (too many)
    # So we need radius 2 cluster and remove 8 outermost hexagons
    
    for R in range(1, 5):
        if time.time() - t_start > DEADLINE:
            break
        
        cells = hex_cluster(R)
        
        if len(cells) < n:
            continue
        
        # If too many, remove outermost cells to get exactly n
        if len(cells) > n:
            # Sort by axial distance from center (use hex distance)
            def hex_dist(c):
                q, r = c
                return max(abs(q), abs(r), abs(q + r))
            
            # Group by distance
            cells_sorted = sorted(cells, key=hex_dist)
            # Remove from the farthest first
            # We want to keep the n closest ones
            # But we should try removing different sets for optimization
            # For now, just keep the n closest
            cells = cells_sorted[:n]
        
        # Get pixel coordinates
        pixels = [axial_to_pixel(q, r) for (q, r) in cells]
        
        # Collect all vertices
        all_points = []
        for (px, py) in pixels:
            verts = hex_vertices(px, py)
            all_points.extend(verts)
        
        # Compute min enclosing hex side (centered at origin)
        side = min_hex_side(all_points, 0.0, 0.0)
        
        if side < best_side:
            best_side = side
            best_center = (0.0, 0.0)
            best_angle = 0.0
            # Build inner list: (x, y, angle_degrees) for each hex
            best_inner = [[px, py, 0.0] for (px, py) in pixels]
    
    # Also try the seed approach (row of 11) as fallback
    seed_side = 40.0
    if best_inner is None or best_side > seed_side:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(n // 2), n - n // 2)]
        return inner, (0.0, 0.0), seed_side, 0.0
    
    # Try to optimize the center position
    # The cluster is symmetric, so origin should be optimal, but let's verify
    # by trying a few offsets
    
    if time.time() - t_start < DEADLINE:
        # Try small offsets
        for dx in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            if time.time() - t_start > DEADLINE:
                break
            for dy in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                side_test = min_hex_side(all_points, dx, dy)
                if side_test < best_side:
                    best_side = side_test
                    best_center = (dx, dy)
    
    return best_inner, best_center, best_side, best_angle


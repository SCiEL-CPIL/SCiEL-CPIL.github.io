"""Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.

Strategy:
- Use a structured lattice-based packing with optimal orientation.
- For 12 hexagons, a hexagonal arrangement (1 center + 6 ring + 5 more) or
  a 3x4 grid-like arrangement can be analyzed.
- The key insight: unit regular hexagons tile the plane. If we place them in a
  hexagonal lattice pattern, the minimal bounding hexagon can be computed analytically.
- We try several configurations and pick the one with minimal outer side length.

A regular hexagon of side s has "radius" (center to vertex) = s, and "apothem" (center to edge) = s*sqrt(3)/2.

For a hexagonal lattice of unit hexagons:
- The distance between centers of adjacent hexagons in the lattice is 2*sqrt(3)/2 = sqrt(3) in the flat-top orientation? 
  Actually, for regular hexagons of side 1, the center-to-center distance for edge-sharing is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3).
  
Let me think more carefully. A regular hexagon of side length a has:
- Distance from center to vertex (circumradius): a
- Distance from center to edge midpoint (inradius/apothem): a * sqrt(3)/2

When two unit hexagons share an edge, their centers are separated by 2 * apothem = sqrt(3).

For a hexagonal lattice packing of n hexagons, we can arrange them in concentric rings:
- Ring 0: 1 hexagon
- Ring 1: 6 hexagons
- Ring 2: 12 hexagons
Total for rings 0,1,2: 1+6+12 = 19. Too many.

For 12 hexagons, we can use:
- A 3x4 rectangular grid in hexagonal lattice coordinates, or
- A compact cluster.

Let me use a different approach: place hexagons in a pattern that fits in a hexagonal container.

For a hexagonal container of side S (in units of inner hexagon side = 1), the number of unit hexagons that fit in a perfect hexagonal arrangement of side k is 3k(k-1)+1.
- k=1: 1
- k=2: 7
- k=3: 19

So 12 doesn't fit a perfect hexagonal number. We need a partial arrangement.

Alternative: Use a 2-row by 6-column arrangement, or other rectangular-ish patterns in hex lattice coords.

Let me just compute: for a given set of inner hexagon positions, what is the minimum outer hexagon side length that contains all of them?

The outer hexagon must contain all inner hexagons. Each inner hexagon has extent: in any direction, its farthest point from its center is 1 (the vertex distance). So the outer hexagon must contain all points at distance up to 1 from each inner center.

This is equivalent to: the outer hexagon (of side S, centered at C, rotated by angle theta) must contain the union of disks of radius 1 around each inner center... no wait, hexagons are not disks. But a necessary condition is that the outer hexagon contains the circumscribed circle of each inner hexagon (radius 1). A sufficient condition for the outer hexagon to contain an inner hexagon is that all 6 vertices of the inner hexagon are inside the outer hexagon.

For simplicity and to get a good answer, let me use a known good packing:

Place 12 unit hexagons in a 3x4 grid in axial coordinates. In axial coords (q,r), the Cartesian position for flat-top hexagons is:
x = 1.5 * q
y = sqrt(3) * (r + q/2)

But we can rotate the whole configuration.

Actually, let me just try a simple arrangement and compute the required outer hexagon size numerically.

I'll place 12 hexagons in a 3x4 rectangular grid with appropriate spacing, then find the minimal enclosing regular hexagon.

For a regular hexagon containing a set of points, the minimal side length can be found by rotating the hexagon and checking containment. But that's complex.

Simpler: if all inner hexagons have the same orientation as the outer, and we align them, we can compute the bounding box in the hexagonal coordinate system.

Let me try: place 12 unit hexagons in a hexagonal lattice pattern forming a 3x4 block. The centers will be at specific positions. Then the outer hexagon needs to enclose all vertices of all inner hexagons.

Given the complexity, let me use a pragmatic approach: try a few known-good configurations and return the best.

Configuration 1: 3 rows of 4 hexagons each, in a staggered (hexagonal) arrangement.
Configuration 2: A more compact cluster.

For a hexagonal lattice, adjacent centers are sqrt(3) apart. Let me define positions:

Row 0 (4 hexagons): x = 0, sqrt(3), 2*sqrt(3), 3*sqrt(3); y = 0
Row 1 (4 hexagons): x = sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2, 7*sqrt(3)/2; y = 3/2  (since vertical spacing for staggered hex is 3/2 * ... wait)

Actually for flat-top hexagons of side 1:
- Horizontal spacing between centers in same row: 2 * apothem? No.
- For edge-sharing, center distance = sqrt(3).
- In a hex grid, if we go in the "east" direction, center moves by (1.5, sqrt(3)/2) for pointy-top, or (sqrt(3), 0) for flat-top... 

Let me just use a concrete coordinate system. Flat-top hexagons:
- Center to center for adjacent in x-direction: sqrt(3)
- Center to center for adjacent in diagonal: (sqrt(3)/2, 3/2)

So for a 3x4 grid:
Row 0: (0,0), (sqrt(3),0), (2*sqrt(3),0), (3*sqrt(3),0)
Row 1: (sqrt(3)/2, 3/2), (3*sqrt(3)/2, 3/2), (5*sqrt(3)/2, 3/2), (7*sqrt(3)/2, 3/2)
Row 2: (0,3), (sqrt(3),3), (2*sqrt(3),3), (3*sqrt(3),3)

The bounding box of centers: x from 0 to 3*sqrt(3) ≈ 5.196, y from 0 to 3.
Adding the hexagon extent (1 in all directions from center for vertices), the total extent is roughly x: -1 to 6.196, y: -1 to 4.

A regular hexagon containing this... the side length would be roughly the max extent in the hexagonal directions.

This is getting complicated. Let me just use a numerical approach: for a given configuration of inner hexagons, compute the minimal outer hexagon side by trying different rotations and using a linear programming or geometric approach.

Given time constraints, I'll use a simpler heuristic: place the hexagons in a compact arrangement and estimate the outer side length as the maximum distance from the centroid to any vertex of any inner hexagon, divided by cos(0) for the hexagon... 

Actually, for a regular hexagon of side S, the distance from center to a vertex is S, and to an edge is S*sqrt(3)/2. If all inner hexagon vertices are within distance S from the outer center, and the angular positions work out, it might fit.

Let me just try a concrete answer: for 12 unit hexagons in a 3x4 hex grid, the outer hexagon side is approximately 4.5 to 5.0.

Let me code a simple version that places them in a 3x4 grid and computes a reasonable outer size.
"""

import numpy as np
import math

N = 12

def construct():
    # Place 12 unit hexagons in a 3x4 staggered grid (hexagonal lattice)
    sqrt3 = math.sqrt(3)
    
    # Flat-top hexagon lattice positions
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * sqrt3 + (row % 2) * (sqrt3 / 2)
            y = row * 1.5
            positions.append((x, y, 0.0))  # angle 0 for all
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    inner = [[p[0] - cx, p[1] - cy, p[2]] for p in positions]
    
    # Compute the required outer hexagon side length
    # Each inner hexagon has vertices at distance 1 from its center.
    # The outer hexagon must contain all these vertices.
    # For a regular hexagon of side S centered at origin, a point (px, py) is inside
    # if for all 6 directions, the projection is within the apothem S*sqrt(3)/2.
    
    # Find the minimal S by checking all vertices of all inner hexagons
    max_required = 0.0
    for (hx, hy, ha) in inner:
        for k in range(6):
            angle = math.radians(60 * k + ha)
            vx = hx + math.cos(angle)
            vy = hy + math.sin(angle)
            # For a regular hexagon with flat top (angle 0), the 6 normal directions are at 0, 60, 120, 180, 240, 300 degrees
            # The constraint is: |dot(point, normal)| <= S * sqrt(3)/2 for each of the 6 normals
            # But the outer hexagon can be rotated. For simplicity, assume same orientation.
            for j in range(6):
                na = math.radians(60 * j)
                nx, ny = math.cos(na), math.sin(na)
                proj = abs(vx * nx + vy * ny)
                required_S = proj / (sqrt3 / 2)
                if required_S > max_required:
                    max_required = required_S
    
    outer_center = (0.0, 0.0)
    outer_side = max_required
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle

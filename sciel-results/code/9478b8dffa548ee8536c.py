"""Program for placing disjoint unit regular hexagons inside a regular hexagon."""
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START

def construct():
    """
    Returns (inner, center, side, angle) where:
      inner: list of [x, y, angle_degrees] for N unit hexagons
      center: (x, y) of outer hexagon center
      side: side length of outer hexagon
      angle: rotation angle (degrees) of outer hexagon
    """
    # Use a hexagonal lattice arrangement.
    # Unit hexagons with side 1: distance between centers in a tight hex lattice is sqrt(3) (for flat-top) or 2 for pointy-top?
    # Let's use flat-top hexagons (angle 0 means flat top/bottom).
    # Actually, for hexagons with side s, the distance from center to vertex is s, and center to edge is s*sqrt(3)/2.
    # For two non-overlapping unit hexagons aligned, the minimum center distance is 2 * (s*sqrt(3)/2) = sqrt(3) * s = sqrt(3) for s=1.
    # Hexagonal lattice vectors for flat-top:
    # d1 = (sqrt(3), 0)
    # d2 = (sqrt(3)/2, 3/2)  -- wait, let me think.
    # Actually, for a regular hexagon with flat top, the neighbors are at angles 0, 60, 120, 180, 240, 300 degrees from center?
    # No. Let's use the standard honeycomb lattice.
    # Centers of touching flat-top hexagons:
    # Horizontal neighbors: distance sqrt(3)
    # The 6 neighbors are at:
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    # So the lattice basis vectors can be:
    # a1 = (sqrt(3), 0)
    # a2 = (sqrt(3)/2, 1.5)
    
    sqrt3 = math.sqrt(3)
    
    # We need 11 hexagons. A hexagonal cluster with 1 in center and 10 around? No, 1+6=7, next ring is 12, total 19.
    # 11 is not a perfect hex number. 
    # Let's do a compact cluster. 
    # Center at (0,0).
    # Ring 1: 6 hexagons.
    # That's 7. We need 4 more.
    # Add them adjacent to the ring.
    # Let's generate centers in a hex lattice.
    
    centers = [(0.0, 0.0)]
    
    # Generate all lattice points up to a certain radius, then pick the 11 closest to origin.
    max_k = 5
    points = []
    for i in range(-max_k, max_k + 1):
        for j in range(-max_k, max_k + 1):
            # Lattice coordinate to Cartesian
            x = i * sqrt3 + j * sqrt3 / 2
            y = j * 1.5
            dist = math.sqrt(x*x + y*y)
            points.append((dist, x, y, i, j))
    
    # Sort by distance
    points.sort(key=lambda t: t[0])
    
    # Take the first 11
    selected = points[:11]
    inner = []
    for _, x, y, _, _ in selected:
        inner.append([x, y, 0.0])
    
    # Now find the minimum enclosing regular hexagon (axis-aligned, angle 0).
    # For a regular hexagon centered at (0,0) with side S and angle 0 (flat top),
    # the boundary in direction theta is: r(theta) = S * cos(theta mod (pi/3) - pi/6)  ??? 
    # Actually, let's use the support function.
    # A regular hexagon with circumradius R (which equals side length S for a regular hexagon) 
    # Wait, no. For a regular hexagon, side length = circumradius.
    # So S = R.
    # The vertices are at (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2).
    # The edges connect these vertices.
    # To find the minimal S such that all inner hexagons are inside, 
    # we need to check each vertex of each inner hexagon.
    # An inner hexagon at (cx, cy) with rotation 0 has vertices at:
    # (cx + S_i*cos(k*60), cy + S_i*sin(k*60)) for k=0..5, where S_i = 1 (side length = circumradius for unit hex).
    # Wait, the inner hexagons have side length 1, so their circumradius is 1.
    # So vertices of inner hex i are at distance 1 from (cx, cy).
    
    # We need to find S such that for all inner hex vertices (vx, vy), they are inside the outer hexagon.
    # The outer hexagon (angle 0, center 0) contains point (vx, vy) if:
    # For each of the 6 edges, the point is on the correct side.
    # Edge 0: from (S,0) to (S/2, S*sqrt(3)/2). Normal is (1, sqrt(3)/2) ? 
    # Let's use the half-plane representation.
    # The 6 outward normals for a flat-top hexagon (angle 0) are at angles 30, 90, 150, 210, 270, 330 degrees? 
    # No. The vertices are at 0, 60, 120, 180, 240, 300.
    # The edges are between consecutive vertices.
    # Edge between 0 and 60 deg vertices: midpoint angle 30 deg. The outward normal is at 30 deg.
    # The distance from center to this edge is S * cos(30) = S * sqrt(3)/2.
    # So the constraint is: for all points (vx, vy) in the cluster, and for each edge normal direction theta_k (k=0..5, theta_k = 30 + 60k),
    # vx * cos(theta_k) + vy * sin(theta_k) <= S * sqrt(3)/2.
    
    # So S >= max over all points and all k of (vx * cos(theta_k) + vy * sin(theta_k)) / (sqrt(3)/2).
    
    max_val = 0.0
    for cx, cy, _ in inner:
        # Vertices of this inner hexagon
        for k in range(6):
            ang = k * math.pi / 3
            vx = cx + math.cos(ang)
            vy = cy + math.sin(ang)
            for m in range(6):
                theta_m = math.pi / 6 + m * math.pi / 3
                val = vx * math.cos(theta_m) + vy * math.sin(theta_m)
                if val > max_val:
                    max_val = val
    
    S = max_val / (math.sqrt(3) / 2)
    # Add a small epsilon to be safe
    S += 1e-9
    
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END

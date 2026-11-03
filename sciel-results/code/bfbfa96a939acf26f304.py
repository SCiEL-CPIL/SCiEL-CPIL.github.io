"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 150s limit, keep 10s margin
    
    # 1. Define the inner hexagons using a hexagonal lattice.
    # For N=12, a 4x3 rectangle in axial coordinates (or similar compact shape) works well.
    # Let's use a 4x3 grid in axial coordinates (q, r) where q is column, r is row.
    # Distance between centers in a hex grid with side length 1 (flat-top) is sqrt(3).
    # However, the problem allows rotation. The most efficient packing of hexagons is the lattice packing.
    # Let's align the lattice with the outer hexagon to minimize bounding box.
    
    # Flat-top hexagon side length 1.
    # Vertices relative to center at angle 0:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # "Flat-top" usually means top edge is horizontal.
    # Standard flat-top hexagon vertices (center 0,0, side 1):
    # (1, 0), (0.5, h), (-0.5, h), (-1, 0), (-0.5, -h), (0.5, -h) where h = sqrt(3)/2.
    # Wait, standard definition:
    # Pointy-top: vertices at 30, 90, 150...
    # Flat-top: vertices at 0, 60, 120...
    # Let's stick to one convention. Let's use "pointy-top" for the lattice calculation often, but let's just define coordinates directly.
    
    # Lattice vectors for hexagonal packing of unit hexagons:
    # If hexagons are flat-top (flat edge on top), the distance between centers horizontally is sqrt(3) * side?
    # No. For flat-top hexagons (width = 2*side, height = sqrt(3)*side):
    # Horizontal distance between centers in same row: sqrt(3) * side? No.
    # Let's visualize.
    # Flat-top hex:
    #   ____
    #  /    \
    # |      |
    #  \____/
    # Width (left to right) = 2 * side.
    # Height (top to bottom) = sqrt(3) * side.
    # Distance between centers in same row (horizontal): sqrt(3) * side?
    # If I place two flat-top hexes side by side, they share a vertical edge? No, they share a slanted edge?
    # Actually, flat-top hexes tile with horizontal spacing of sqrt(3)*side and vertical spacing of 1.5*side (offset rows).
    # Let's verify.
    # Center (0,0). Right neighbor center (sqrt(3), 0)?
    # Vertex at (1,0). Next vertex (0.5, sqrt(3)/2).
    # If I put another hex at (sqrt(3), 0), its left vertex is (sqrt(3)-1, 0).
    # Distance from (1,0) to (sqrt(3)-1, 0) is sqrt(3)-2 < 0. Overlap.
    # Correct horizontal spacing for flat-top hexes to touch:
    # The width is 2. The "shoulders" are at x=0.5.
    # The distance between centers of two adjacent flat-top hexes is sqrt(3).
    # Let's check:
    # Hex 1 center (0,0). Hex 2 center (sqrt(3), 0).
    # Hex 1 rightmost point x=1. Hex 2 leftmost point x=sqrt(3)-1 ≈ 0.732.
    # 1 > 0.732. They overlap.
    # Ah, for flat-top, the "pointy" parts are left and right.
    # So they don't tile side-by-side simply.
    # For flat-top, you tile by shifting rows.
    # Row 0: y=0. x = 0, sqrt(3), 2*sqrt(3)...
    # Row 1: y=sqrt(3)/2 * 1.5? No.
    # Let's use the standard axial coordinate system for hex grids.
    # Axial (q, r).
    # Pixel coordinates for flat-top:
    # x = size * 3/2 * q
    # y = size * sqrt(3) * (r + q/2)
    # Distance between (0,0) and (1,0) is 3/2 * size?
    # Let's check distance.
    # If size=1.
    # (0,0) -> (0,0)
    # (1,0) -> (1.5, sqrt(3)/2).
    # Dist = sqrt(1.5^2 + 0.75) = sqrt(2.25+0.75) = sqrt(3) ≈ 1.732.
    # This is the distance between centers of touching flat-top hexagons.
    # So, we can generate 12 positions in a compact cluster.
    # A 4x3 rectangle in axial coords: q in [-2, 1], r in [-1, 1]?
    # Let's just take the first 12 points from a spiral or a rectangular block.
    # A 4x3 block of hexes is a good shape.
    # Axial coords for a 4x3 block centered at 0:
    # q: -2, -1, 0, 1
    # r: -1, 0, 1
    # Total 12 points.
    
    positions = []
    for q in range(-2, 2):
        for r in range(-1, 2):
            # Convert axial to pixel
            x = 1.5 * q
            y = math.sqrt(3) * (r + q/2.0)
            positions.append((x, y))
            
    # These positions are for flat-top hexagons of side 1.
    # The inner hexagons should be aligned with this lattice to be disjoint.
    # So inner_angle = 0 (if 0 corresponds to flat-top in our vertex generation).
    
    # Let's define the vertex generation to match the lattice.
    # Flat-top hexagon vertices (angle 0):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # This matches the x,y calculation above.
    
    inner = []
    for x, y in positions:
        inner.append([x, y, 0.0])
        
    # 2. Determine the minimal outer hexagon.
    # The outer hexagon must contain all inner hexagons.
    # Since the inner hexagons are on a lattice, the convex hull of the union of inner hexagons
    # must be inside the outer hexagon.
    # However, the outer hexagon is also a regular hexagon.
    # We can try to align the outer hexagon with the lattice (angle 0) and find the minimal side length.
    # Or we can rotate the outer hexagon.
    # Given the symmetry of the 4x3 block, aligning it with the lattice is likely optimal or near-optimal.
    # Let's calculate the bounding box of the vertices of the inner hexagons.
    
    all_vertices = []
    for x, y, a in inner:
        # Generate vertices of this inner hexagon
        for i in range(6):
            theta = math.radians(a) + i * math.pi / 3
            vx = x + 1.0 * math.cos(theta)
            vy = y + 1.0 * math.sin(theta)
            all_vertices.append((vx, vy))
            
    # Find the extent of these vertices.
    # If the outer hexagon is also flat-top (angle 0), its boundaries are defined by:
    # |x| <= S
    # |x/2 + sqrt(3)/2 * y| <= S * sqrt(3)/2 ? No.
    # Let's derive the inequalities for a flat-top regular hexagon centered at origin with side S.
    # Vertices: (S,0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2).
    # The hexagon is the intersection of 6 half-planes.
    # The normals to the edges are at angles 30, 90, 150, 210, 270, 330 degrees?
    # Edge 1: from (S,0) to (S/2, S*sqrt(3)/2).
    # Midpoint: (3S/4, S*sqrt(3)/4).
    # Normal vector pointing out: (sqrt(3)/2, 1/2)?
    # Let's check dot product with normal (sqrt(3)/2, 1/2) and vertex (S,0): S*sqrt(3)/2.
    # Distance from origin to edge = S * sqrt(3)/2.
    # So the condition is:
    # For each of the 6 directions (angles 30, 90, 150, 210, 270, 330),
    # the projection of any point (x,y) onto the normal vector must be <= S * sqrt(3)/2.
    # Wait, the "width" of a flat-top hexagon is 2S. The "height" is S*sqrt(3).
    # The distance from center to edge (apothem) is S*sqrt(3)/2.
    # So, max over all vertices v of (v . n_i) <= S * sqrt(3)/2 for all 6 normals n_i.
    # This implies S >= max( max(v . n_i) / (sqrt(3)/2) ).
    
    normals = []
    for i in range(6):
        # Normals are perpendicular to edges.
        # Edge angles are 0, 60, 120...? No.
        # Vertex angles are 0, 60, 120...
        # Edge midpoints are at 30, 90, 150...
        angle = math.radians(30 + i * 60)
        nx = math.cos(angle)
        ny = math.sin(angle)
        normals.append((nx, ny))
        
    max_proj = 0.0
    for vx, vy in all_vertices:
        for nx, ny in normals:
            proj = vx * nx + vy * ny
            if proj > max_proj:
                max_proj = proj
                
    # S * sqrt(3)/2 >= max_proj
    # S >= max_proj * 2 / sqrt(3)
    side_length = max_proj * 2.0 / math.sqrt(3)
    
    # We should also check if rotating the outer hexagon helps.
    # But for now, this gives a valid upper bound.
    # The 4x3 block is symmetric.
    # Let's refine the position of the outer hexagon center.
    # The bounding box of the inner hexagons might not be centered at (0,0) if the block is asymmetric?
    # Our block q in [-2, 1], r in [-1, 1] is NOT symmetric around 0.
    # q: -2, -1, 0, 1. Center of q is -0.5.
    # Let's center the block properly.
    # 12 hexes.
    # A better shape: a hexagon of radius 2 has 1 + 6 + 12 = 19 hexes. Too many.
    # Radius 1 has 7 hexes.
    # We need 12.
    # 7 + 5?
    # Let's use a 4x3 grid but centered.
    # Axial coords for 4x3 centered:
    # q: -1, 0, 1, 2 ? No.
    # Let's just take the 12 closest points to origin in a hex grid.
    # Points:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 points (radius 1)
    # Next ring (radius 2):
    # (2,0), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,0) ...
    # We need 5 more from the second ring to make 12.
    # The second ring has 12 points.
    # To keep it compact and symmetric, we should pick a subset.
    # However, a rectangular 4x3 block is also compact.
    # Let's re-center the 4x3 block.
    # q: -1, 0, 1, 2 is not centered.
    # Let's use q: -2, -1, 0, 1 and r: -1, 0, 1.
    # Sum of x:
    # q=-2: x=-3, y=sqrt(3)(r-1) -> r=-1: y=-2sqrt(3), r=0: y=-sqrt(3), r=1: y=0
    # q=-1: x=-1.5, y=sqrt(3)(r-0.5) -> r=-1: y=-1.5sqrt(3), r=0: y=-0.5sqrt(3), r=1: y=0.5sqrt(3)
    # q=0: x=0, y=sqrt(3)r -> r=-1: y=-sqrt(3), r=0: y=0, r=1: y=sqrt(3)
    # q=1: x=1.5, y=sqrt(3)(r+0.5) -> r=-1: y=-0.5sqrt(3), r=0: y=0.5sqrt(3), r=1: y=1.5sqrt(3)
    
    # This shape is roughly centered but slightly skewed.
    # Let's just compute the bounding box center and shift the inner hexes so their "center of mass" is at (0,0) or the bounding box is centered.
    # Actually, the outer hexagon center can be anywhere.
    # We calculated `side_length` assuming outer center is (0,0).
    # We should find the optimal outer center (cx, cy) that minimizes the required side length.
    # This is a convex optimization problem (minimize max of linear functions).
    # But we can approximate by centering the bounding box of the vertices.
    
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    
    # Shift inner hexes so their bounding box is centered at (0,0) for the outer hex calc?
    # No, the outer hexagon center is (cx, cy).
    # Let's re-calculate side length with outer center (cx, cy).
    # The vertices of inner hexes are fixed in world coords.
    # The outer hexagon is centered at (cx, cy).
    # Condition: for all inner vertices v, and all normals n:
    # (v - c) . n <= S * sqrt(3)/2
    # (v . n - c . n) <= S * sqrt(3)/2
    # v . n - c . n <= S * sqrt(3)/2
    # S >= (v . n - c . n) * 2 / sqrt(3)
    
    max_proj_centered = 0.0
    for vx, vy in all_vertices:
        for nx, ny in normals:
            # Project relative to center
            rel_x = vx - cx
            rel_y = vy - cy
            proj = rel_x * nx + rel_y * ny
            if proj > max_proj_centered:
                max_proj_centered = proj
                
    side_length = max_proj_centered * 2.0 / math.sqrt(3)
    
    # Add a small epsilon to ensure strict containment if needed, though <= is fine for disjointness if touching is allowed.
    # The problem says "disjoint", which usually allows touching.
    
    return inner, (cx, cy), side_length, 0.0
# EVOLVE-BLOCK-END

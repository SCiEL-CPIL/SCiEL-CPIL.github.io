import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    
    # Define the 3-4-3 hexagonal lattice configuration
    # Coordinates are in units of side length 1.
    # Hexagon with side 1 has width sqrt(3) and height 2.
    # For a "pointy-top" hexagon (vertex at top), neighbors are at distance sqrt(3) horizontally?
    # No, let's use the standard axial coordinates for a hex grid.
    # Let's place them in a triangular arrangement.
    
    # A regular hexagon of side 1 can be tiled by unit hexagons.
    # The 3-4-3 arrangement is a hexagon of "radius" 1 in hex grid terms (center + 6 neighbors + 4 more? No, 1+6+12=19).
    # 3-4-3 is a hexagon of side length 2 in terms of number of hexes per side?
    # A hexagon of side length k has 3k(k-1)+1 hexes.
    # k=1: 1
    # k=2: 7
    # k=3: 19
    # So 3-4-3 is not a full hexagon of side 3.
    # 3-4-3 is a subset of the k=3 hexagon.
    # Let's construct the 3-4-3 shape explicitly.
    
    # Using axial coordinates (q, r) for a hex grid where pointy-top hexes are used.
    # Distance between centers of adjacent hexes is sqrt(3) * 1 = sqrt(3).
    # Let's define positions in Cartesian coordinates directly for a "flat-top" or "pointy-top" grid.
    
    # Let's use a pointy-top hexagon for the inner hexagons (angle 90 deg for vertex up? or 0?).
    # The code in the seed used `hex_vertices` with `theta = a + 60*i`.
    # If a=0, vertices are at 0, 60, 120... degrees. This is "pointy-right" or "flat-top"?
    # Vertex at 0 deg is (S, 0). Vertex at 60 is (S/2, S*sqrt(3)/2).
    # This is a hexagon with a vertex pointing to the right.
    
    # Let's define the 3-4-3 cluster centers.
    # Row 1 (top): 3 hexes
    # Row 2 (middle): 4 hexes
    # Row 3 (bottom): 3 hexes
    
    # Spacing:
    # Horizontal spacing between centers in same row: sqrt(3)
    # Vertical spacing between rows: 1.5 (which is 3/2 * 1)
    # Offset between rows: sqrt(3)/2
    
    # Let's center the cluster at (0,0).
    # Total width of 4 hexes: 3 * sqrt(3) + sqrt(3) = 4*sqrt(3)? No.
    # Centers of 4 hexes in a row: x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3).
    # Total span of centers: 3*sqrt(3).
    # Plus hex width (sqrt(3)) -> Total width 4*sqrt(3).
    
    # Total height of 3 rows:
    # y = -1.5, 0, 1.5.
    # Span of centers: 3.
    # Plus hex height (2) -> Total height 5.
    
    # Let's generate the points.
    points = []
    
    # Row 0 (top, y = 1.5): 3 hexes
    # x offsets: -sqrt(3), 0, sqrt(3)
    for i in range(3):
        x = (i - 1) * math.sqrt(3)
        y = 1.5
        points.append((x, y))
        
    # Row 1 (middle, y = 0): 4 hexes
    # x offsets: -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    for i in range(4):
        x = (i - 1.5) * math.sqrt(3)
        y = 0.0
        points.append((x, y))
        
    # Row 2 (bottom, y = -1.5): 3 hexes
    # x offsets: -sqrt(3), 0, sqrt(3)
    for i in range(3):
        x = (i - 1) * math.sqrt(3)
        y = -1.5
        points.append((x, y))
        
    # Total 11 points.
    # Inner hexagons have side 1.
    # We will rotate the entire cluster by an angle `alpha`.
    # The outer hexagon is fixed at angle 0 (or we can rotate it, but relative rotation is what matters).
    # Let's assume outer hexagon is "flat-top" or "pointy-top" consistent with inner?
    # The seed program allowed independent rotation.
    # To minimize the bounding box, we should align the "wide" parts of the cluster with the "narrow" parts of the hexagon if possible, or optimize the angle.
    
    # Let's compute the minimal side length S of the outer hexagon (angle 0) containing the rotated cluster.
    # Outer hexagon (angle 0) has normals at 30, 90, 150, 210, 270, 330 degrees.
    # Apothem a = S * sqrt(3) / 2.
    # Constraint: max projection of any vertex of any inner hexagon onto normal n <= a.
    
    # Vertices of an inner hexagon at center (cx, cy) with rotation `alpha`:
    # v_k = (cx, cy) + R(alpha) * R(0) * v_0_k
    # Where v_0_k are vertices of unit hexagon at angle 0.
    # v_0_k = (cos(60*k), sin(60*k)).
    
    # We need to check all vertices of all 11 hexagons.
    
    def get_cluster_vertices(alpha):
        verts = []
        # Precompute unit hex vertices at angle 0
        base_verts = []
        for k in range(6):
            theta = math.radians(60 * k)
            base_verts.append((math.cos(theta), math.sin(theta)))
            
        ca = math.cos(math.radians(alpha))
        sa = math.sin(math.radians(alpha))
        
        for (cx, cy) in points:
            for (vx, vy) in base_verts:
                # Rotate vertex
                rx = ca * vx - sa * vy
                ry = sa * vx + ca * vy
                # Translate
                verts.append((cx + rx, cy + ry))
        return verts
        
    def min_outer_side(alpha):
        verts = get_cluster_vertices(alpha)
        max_proj = 0.0
        # Normals of outer hexagon (angle 0)
        for i in range(6):
            theta = math.radians(30 + 60 * i)
            nx = math.cos(theta)
            ny = math.sin(theta)
            for (px, py) in verts:
                proj = px * nx + py * ny
                if proj > max_proj:
                    max_proj = proj
        # S * sqrt(3)/2 >= max_proj
        return max_proj / (math.sqrt(3) / 2)
        
    # Optimize alpha.
    # The function is periodic with period 60 degrees.
    # We can do a fine grid search.
    best_S = float('inf')
    best_alpha = 0
    
    # Search alpha from 0 to 60 degrees
    # Step size 0.1 degrees? 600 steps.
    # Each step calculates 66 vertices and 6 projections. Very fast.
    for i in range(600):
        alpha = i * 0.1
        S = min_outer_side(alpha)
        if S < best_S:
            best_S = S
            best_alpha = alpha
            
    # Refine around best_alpha
    # Golden section search or just finer grid
    start = max(0, best_alpha - 0.1)
    end = min(60, best_alpha + 0.1)
    if start == 0 and best_alpha == 0:
        start = 59.9
        end = 60.1 # handle wrap? No, just check 0 to 0.1 again
    
    # Actually, just do a finer grid over the whole range if it's fast enough.
    # 6000 steps (0.01 deg) is still fast.
    for i in range(6000):
        alpha = i * 0.01
        S = min_outer_side(alpha)
        if S < best_S:
            best_S = S
            best_alpha = alpha
            
    # Construct the result
    # Inner hexagons: (x, y, angle_degrees)
    # We rotated the cluster by best_alpha.
    # The centers (cx, cy) were defined for alpha=0.
    # We need to rotate the centers by best_alpha as well.
    # And the angle of each inner hexagon is best_alpha.
    
    ca = math.cos(math.radians(best_alpha))
    sa = math.sin(math.radians(best_alpha))
    
    inner = []
    for (cx, cy) in points:
        # Rotate center
        rx = ca * cx - sa * cy
        ry = sa * cx + ca * cy
        inner.append([rx, ry, best_alpha])
        
    # Outer hexagon
    # Center (0,0)
    # Side length best_S
    # Angle 0 (since we optimized for outer angle 0)
    
    # Verify validity quickly?
    # The calculation `min_outer_side` guarantees that all vertices are within the apothem distance.
    # Since the hexagon is convex and defined by these half-planes, all vertices inside implies the whole hexagon is inside.
    # Disjointness is guaranteed by the lattice construction (spacing sqrt(3) > sum of radii? No, they touch).
    # In a hex lattice, hexagons of side 1 touch each other. They are disjoint (interiors do not overlap).
    # The problem says "disjoint", which usually means interiors are disjoint. Touching is allowed.
    
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END

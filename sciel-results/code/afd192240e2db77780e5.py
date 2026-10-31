import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # The strategy is to use a perfect hexagonal lattice packing.
    # Regular hexagons of side length 1 tile the plane.
    # If we align the inner hexagons with the outer hexagon,
    # we can place them on a triangular grid.
    
    # Spacing between centers of adjacent non-overlapping hexagons
    # in a hexagonal lattice is 2 * apothem = sqrt(3).
    sp = math.sqrt(3)
    
    # We want to pack 11 hexagons.
    # A central hexagon surrounded by 6 (total 7) fits in a hexagon of side 2.
    # Let's verify:
    # Center at (0,0).
    # 6 neighbors at distance sp = sqrt(3) at angles 0, 60, 120, 180, 240, 300.
    # The farthest points of these neighbors from the origin are at distance sp + 1.
    # For a regular hexagon of side S centered at origin, the apothem is S*sqrt(3)/2.
    # The distance from center to vertex is S.
    # The distance from center to edge midpoint is S*sqrt(3)/2.
    
    # Let's define the positions for 11 hexagons.
    # Positions in a hex grid (axial coordinates q, r):
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 6 hexes
    # Total 7.
    # We need 4 more. We can add them in Ring 2.
    # Ring 2 positions: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-1)... wait.
    # Let's just pick 4 from Ring 2 that are symmetric or compact.
    # To minimize the bounding box, we should pick positions that fill the corners or sides evenly.
    
    # Axial to Cartesian conversion for hex grid with pointy-top orientation (angle 0 is right):
    # x = sp * (q + r/2)
    # y = sp * (r * sqrt(3)/2)
    # Wait, standard hex grid spacing:
    # If hexagons are pointy-top (vertex at top), the distance between centers horizontally is sqrt(3).
    # The distance between centers vertically is 1.5.
    # Let's stick to the rotation angle 0 for simplicity, which usually implies flat-top or pointy-top depending on definition.
    # In the previous code, angle 0 was pointy-top (vertex at 0 deg? No, 0 deg was used in hex_vertices with cos/sin, so vertex at (S,0)).
    # If vertex is at (S,0), it's "pointy-right".
    # Distance between centers of adjacent hexes sharing a vertical edge (left/right) is 2 * apothem = sqrt(3).
    # Distance between centers of adjacent hexes sharing a slanted edge is sqrt(3) as well.
    
    # Let's define the 11 positions explicitly in Cartesian coordinates.
    # We will use a "flower" shape.
    # Center: (0, 0)
    # 6 neighbors:
    #   (sqrt(3), 0)
    #   (sqrt(3)/2, 1.5)
    #   (-sqrt(3)/2, 1.5)
    #   (-sqrt(3), 0)
    #   (-sqrt(3)/2, -1.5)
    #   (sqrt(3)/2, -1.5)
    
    # This cluster of 7 has a "width" (max x - min x) of 2*sqrt(3) + 2 (diameter of hex).
    # Max x: sqrt(3) + 1. Min x: -sqrt(3) - 1.
    # Max y: 1.5 + sqrt(3)/2. Min y: -1.5 - sqrt(3)/2.
    
    # We need 4 more. Let's add them at the "corners" of the second ring to keep it compact.
    # The second ring neighbors of the center are at distance 2*sqrt(3) or 2*1.5 etc.
    # Actually, let's just extend the linear chain or add to the sides.
    # A very compact arrangement for 11 is a hexagon of side 2 (which holds 1+6+12=19) is too big.
    # A hexagon of side 1.5 doesn't exist in integer lattice.
    
    # Let's try a specific configuration known to be dense:
    # 1 in center.
    # 6 around it.
    # 4 more placed adjacent to the 6, specifically at positions (2,0), (0,2), (-2,0), (0,-2) in axial coords?
    # Let's use Cartesian coords directly.
    
    positions = []
    # Center
    positions.append((0.0, 0.0))
    
    # Ring 1
    for i in range(6):
        theta = math.radians(60 * i)
        x = sp * math.cos(theta)
        y = sp * math.sin(theta)
        positions.append((x, y))
        
    # We have 7. Need 4 more.
    # Let's add them at the "points" of the hexagon formed by the centers of Ring 1.
    # The centers of Ring 1 form a hexagon of side sp.
    # The next ring out would be at distance 2*sp from center along the axes? No.
    # In a hex lattice, the neighbors of (1,0) [which is (sp, 0)] are (2,0), (1,1), (0,1), (0,-1), (1,-1), (2,-1).
    # (2,0) in axial is x = sp*(2+0) = 2sp, y = 0.
    # (1,1) in axial is x = sp*(1+0.5) = 1.5sp, y = sp*sqrt(3)/2 * 1 = 1.5.
    
    # Let's add 4 positions from Ring 2 that are symmetric.
    # (2, 0) -> (2*sp, 0)
    # (-2, 0) -> (-2*sp, 0)
    # (0, 2) -> (sp, 1.5*sqrt(3)) ? No.
    # Axial (0,2): x = sp*(0+1) = sp, y = sp*sqrt(3)/2 * 2 = sp*sqrt(3) = 3.
    # Wait, sp = sqrt(3). So y = 3.
    # Let's check distance from center (0,0) to (sp, 3).
    # sp^2 + 9 = 3 + 9 = 12. sqrt(12) = 2*sqrt(3) = 2*sp. Correct.
    
    # So let's add:
    # (2*sp, 0)
    # (-2*sp, 0)
    # (sp, 3.0)  # Which is (0,2) in axial
    # (-sp, -3.0) # Which is (0,-2) in axial?
    # Axial (0,-2): x = sp*(0-1) = -sp, y = sp*sqrt(3)/2 * (-2) = -3.
    
    # Let's verify these 4 are disjoint from the existing 7.
    # (2*sp, 0) is neighbor to (sp, 0). Distance sp. OK.
    # Is (2*sp, 0) disjoint from others?
    # Distance to (1.5*sp, 1.5) [which is (1,1) axial? No, (1,1) axial is x=1.5sp, y=1.5].
    # Wait, my Ring 1 points were:
    # i=0: (sp, 0)
    # i=1: (sp/2, 1.5)
    # i=2: (-sp/2, 1.5)
    # i=3: (-sp, 0)
    # i=4: (-sp/2, -1.5)
    # i=5: (sp/2, -1.5)
    
    # Let's check distance from (2*sp, 0) to (sp/2, 1.5).
    # dx = 1.5*sp. dy = -1.5.
    # dist^2 = (1.5*sqrt(3))^2 + 1.5^2 = 2.25*3 + 2.25 = 6.75 + 2.25 = 9.
    # dist = 3.
    # Required distance for disjoint hexes is sp = sqrt(3) ~ 1.732.
    # 3 > 1.732. So they are disjoint.
    
    # So the 4 extra points are:
    # (2*sp, 0)
    # (-2*sp, 0)
    # (sp, 3.0)
    # (-sp, -3.0)
    
    # This creates a somewhat asymmetric shape (long diagonal).
    # Can we do better?
    # What if we add (1.5*sp, 1.5) and (-1.5*sp, -1.5) and (1.5*sp, -1.5) and (-1.5*sp, 1.5)?
    # These are (1,1), (-1,-1), (1,-1), (-1,1) in axial.
    # Let's check (1,1) -> (1.5*sp, 1.5).
    # Distance to (sp, 0) [axial 1,0]: dx=0.5sp, dy=1.5.
    # dist^2 = 0.75 + 2.25 = 3. dist = sqrt(3) = sp. OK.
    # Distance to (sp/2, 1.5) [axial 0,1]: dx=sp, dy=0.
    # dist = sp. OK.
    # Distance to (0,0): dist^2 = 3*2.25 + 2.25 = 6.75+2.25=9. dist=3. OK.
    
    # So adding the 4 "diagonal" neighbors of the center (in terms of grid) is valid.
    # Positions:
    # (1.5*sp, 1.5)
    # (-1.5*sp, -1.5)
    # (1.5*sp, -1.5)
    # (-1.5*sp, 1.5)
    
    # This shape is more compact (diamond-like).
    # Let's calculate the bounding box for this configuration.
    
    all_pos = []
    # Center
    all_pos.append((0.0, 0.0))
    # Ring 1
    for i in range(6):
        theta = math.radians(60 * i)
        x = sp * math.cos(theta)
        y = sp * math.sin(theta)
        all_pos.append((x, y))
    # Ring 2 (4 selected)
    all_pos.append((1.5*sp, 1.5))
    all_pos.append((-1.5*sp, -1.5))
    all_pos.append((1.5*sp, -1.5))
    all_pos.append((-1.5*sp, 1.5))
    
    # Now we need to find the minimum outer hexagon side length S.
    # We can rotate the outer hexagon to minimize S.
    # Due to symmetry of the inner packing (it has 180 degree rotational symmetry, but not 60),
    # we might need to check a few angles.
    # However, the inner packing is aligned with the lattice.
    # The "width" of the inner packing in any direction d is max(p . d) - min(p . d) + 2 (diameter of hex).
    # Actually, the condition for a hexagon of side S centered at origin to contain a point p is:
    # |p . n_i| <= S * sqrt(3)/2 for all 6 normals n_i.
    # So S >= max_i (max_p (p . n_i)) / (sqrt(3)/2).
    # Note: We should center the outer hex at the centroid of the inner hexes?
    # The configuration is centrally symmetric (point reflection), so centroid is (0,0).
    
    # Let's compute S for angle 0 (outer hex aligned with lattice).
    # Normals for angle 0 (pointy-right, vertex at (S,0)):
    # The sides are vertical? No.
    # If vertex is at (S,0), the edges are at angles 30, 90, 150, 210, 270, 330.
    # The normals point outwards at 0, 60, 120, 180, 240, 300.
    # Wait, if vertex is at 0 deg, the edge connecting vertex 0 and 1 is at 30 deg.
    # The normal to that edge is at 30 - 90 = -60 or 30 + 90 = 120?
    # Let's use the property: Apothem is distance from center to side.
    # For a hexagon with vertex at (S, 0), the rightmost point is (S, 0).
    # The rightmost side is not vertical.
    # Let's stick to the code's definition: hex_vertices(cx, cy, S, angle_deg).
    # Angle 0: vertex 0 at (S, 0).
    # The 6 vertices are at 0, 60, 120, 180, 240, 300.
    # The edges connect 0-60, 60-120, etc.
    # Edge 0-60 midpoint angle is 30. Normal is 30+90=120? Or 30-90=-60?
    # The normal pointing outwards from the center for the edge at 30 deg is at 30 deg? No.
    # The radius to the midpoint of the edge is the apothem.
    # The angle of the midpoint of edge (0,60) is 30.
    # So the normal is at 30 degrees.
    # So the 6 normals are at 30, 90, 150, 210, 270, 330.
    
    # Let's verify.
    # If normal is at 90 (up), the edge is horizontal at y = apothem.
    # Vertices at 60 and 120 have y = S*sin(60) = S*sqrt(3)/2 = apothem. Correct.
    # So normals are at 30 + 60*k.
    
    # We want to minimize S = max_k ( max_p (p . n_k) ) / (sqrt(3)/2).
    # We can rotate the outer hexagon by angle alpha.
    # Then normals are at 30 + alpha + 60*k.
    
    # Let's sample alpha from 0 to 30 degrees (symmetry period is 60, but inner shape has 180 sym, so 30 is enough? No, inner shape is not 60 sym).
    # Actually, let's just calculate for alpha = 0.
    
    alpha = 0.0
    apothem_factor = math.sqrt(3) / 2.0
    max_proj = 0.0
    
    # Generate vertices for all inner hexagons to be precise?
    # No, for a regular hexagon, the max projection of the hexagon itself onto a normal n is S_inner * (projection of vertex).
    # But we are given centers.
    # The max projection of a unit hexagon centered at c onto normal n is (c . n) + 1 (if n aligns with a vertex direction) or (c . n) + sqrt(3)/2 (if n aligns with edge normal).
    # Actually, the support function of a regular hexagon of side 1 is h(n) = max(1, sqrt(3)/2) ? No.
    # h(n) depends on the angle of n relative to the hexagon.
    # For a unit hexagon, the max distance from center to boundary in direction n is:
    # If n is parallel to a vertex radius, dist = 1.
    # If n is parallel to an apothem, dist = sqrt(3)/2.
    # It varies sinusoidally? No, it's the support function of a polygon.
    
    # To be safe and simple, let's just compute the max projection of the vertices of all inner hexagons.
    # There are 11 * 6 = 66 vertices. This is very fast.
    
    inner_vertices = []
    for cx, cy in all_pos:
        for i in range(6):
            theta = math.radians(60 * i) # Inner hexagons are aligned with angle 0
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            inner_vertices.append((vx, vy))
    
    S_min = float('inf')
    # Try a few angles for the outer hexagon
    for alpha_deg in [0, 15, 30, 45, 60]:
        a = math.radians(alpha_deg)
        current_S = 0.0
        for k in range(6):
            theta_n = a + math.radians(30 + 60 * k)
            nx = math.cos(theta_n)
            ny = math.sin(theta_n)
            max_p = 0.0
            for vx, vy in inner_vertices:
                proj = vx * nx + vy * ny
                if proj > max_p:
                    max_p = proj
            # S * apothem_factor >= max_p
            S_needed = max_p / apothem_factor
            if S_needed > current_S:
                current_S = S_needed
        if current_S < S_min:
            S_min = current_S
    
    # The result S_min is the side length.
    # We need to return:
    # inner: list of (x, y, angle_degrees)
    # center: (0, 0)
    # side: S_min
    # angle: alpha_deg that gave S_min
    
    # Let's find the best angle again to return it
    best_alpha = 0
    best_S = float('inf')
    for alpha_deg in [0, 15, 30, 45, 60]:
        a = math.radians(alpha_deg)
        current_S = 0.0
        for k in range(6):
            theta_n = a + math.radians(30 + 60 * k)
            nx = math.cos(theta_n)
            ny = math.sin(theta_n)
            max_p = 0.0
            for vx, vy in inner_vertices:
                proj = vx * nx + vy * ny
                if proj > max_p:
                    max_p = proj
            S_needed = max_p / apothem_factor
            if S_needed > current_S:
                current_S = S_needed
        if current_S < best_S:
            best_S = current_S
            best_alpha = alpha_deg
            
    # Construct the return value
    inner = []
    for cx, cy in all_pos:
        inner.append((cx, cy, 0.0))
        
    return inner, (0.0, 0.0), best_S, best_alpha

# EVOLVE-BLOCK-END

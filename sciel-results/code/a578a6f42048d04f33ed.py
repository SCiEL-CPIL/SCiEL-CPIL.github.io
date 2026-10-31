"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

# EVOLVE-BLOCK-START
def construct():
    # Use a hexagonal close-packing pattern.
    # A hexagonal number H_k = 3k(k-1)+1 represents the number of hexagons
    # in a hexagonal arrangement with k rings (including center).
    # We need to choose n such that the packing is dense and fits well.
    # The seed had N=11. Let's try to use a similar count but arranged hexagonally.
    # H_2 = 7, H_3 = 19, H_4 = 37.
    # 11 is not a hexagonal number.
    # Let's try to pack 11 hexagons optimally.
    # A central hexagon surrounded by 6, plus 4 more?
    # Or a row of 11? The seed was a row.
    # A hexagonal cluster of 19 is too many.
    # A cluster of 7 (1 center + 6 around) is compact.
    # Let's try to pack 11 by taking a 7-cluster and adding 4 more on the perimeter.
    # Or simply use a known optimal packing for small n.
    # For n=11, a common dense packing is a hexagon of side 2 (7 hexes) 
    # plus 4 more attached to the flat sides?
    
    # Actually, let's just generate a hexagonal grid of points and pick the first 11.
    # The distance between centers of adjacent unit hexagons is sqrt(3).
    # Let d = sqrt(3).
    # Axial coordinates (q, r) to pixel coordinates:
    # x = d * (q + r/2)
    # y = d * (3/4) * r
    
    # Generate points in a hexagonal pattern centered at origin.
    # We want to minimize the outer hexagon side length.
    # The outer hexagon must contain all these centers plus the radius of the inner hexagons.
    # The "radius" (center to vertex) of a unit hexagon is 1.
    # The "inradius" (center to side) is sqrt(3)/2.
    
    # Let's place 11 hexagons.
    # Order by distance from center.
    # (0,0) -> dist 0
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1) -> 6 neighbors. Total 7.
    # Next ring: (2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2) and (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2)
    # Let's just generate a grid and sort by distance.
    
    d = math.sqrt(3)
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = d * (q + r/2)
            y = d * (3/4) * r
            dist = math.sqrt(x*x + y*y)
            points.append((dist, x, y))
            
    points.sort()
    
    # Take the first 11 points
    selected = points[:11]
    inner = []
    for dist, x, y in selected:
        # Angle is 0 for all, since the lattice is aligned with the outer hexagon
        inner.append([x, y, 0.0])
        
    # Now determine the outer hexagon size.
    # The outer hexagon is centered at (0,0).
    # It must contain all inner hexagons.
    # An inner hexagon at (x,y) with rotation 0 has vertices at distance 1 from (x,y).
    # The farthest point of an inner hexagon from the origin is at distance sqrt(x^2+y^2) + 1.
    # However, the outer hexagon is a regular hexagon.
    # The distance from center to a vertex of the outer hexagon is S (side length).
    # The distance from center to the midpoint of a side is S * sqrt(3)/2.
    # We need to find the minimum S such that all inner hexagons are inside.
    
    # For a regular hexagon with side S centered at origin, the condition for a point (x,y) to be inside is:
    # max(|x|, |y|/sqrt(3) + |x|/2, -y/sqrt(3) + |x|/2) <= S * sqrt(3)/2 ? 
    # Actually, the support function of a regular hexagon with "radius" R (center to vertex) is R.
    # The inradius is R * sqrt(3)/2.
    # Let's use the property that a point (x,y) is inside a regular hexagon of side S (centered at origin, flat top) if:
    # |x| <= S * sqrt(3)/2 * cos(theta) + ... this is getting complicated.
    
    # Simpler: The outer hexagon must cover the convex hull of the inner hexagons.
    # The vertices of the inner hexagons are the critical points.
    # For each inner hexagon at (cx, cy), its 6 vertices are at:
    # (cx + cos(a), cy + sin(a)) for a in [0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3]
    
    # We need the smallest S such that all these vertices are inside the outer hexagon.
    # The outer hexagon (flat top, side S) has vertices at:
    # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    
    # A point (px, py) is inside the outer hexagon if:
    # 1. |py| <= S * sqrt(3)/2
    # 2. |px| <= S - |py| / sqrt(3)  (for the slanted sides)
    
    # So for a given (px, py), the required S is:
    # S >= |py| / sqrt(3) + |px| / 2  ... wait.
    # The line equation for the side in the first quadrant is:
    # y = -sqrt(3)(x - S) => y = -sqrt(3)x + sqrt(3)S => sqrt(3)x + y = sqrt(3)S
    # So x + y/sqrt(3) <= S
    # Similarly for other quadrants.
    # The condition is: max( |x|, |x|/2 + |y|/sqrt(3) ) <= S ?
    # Let's check.
    # If x>0, y>0: x + y/sqrt(3) <= S
    # If x>0, y<0: x - y/sqrt(3) <= S
    # So max(x, x + y/sqrt(3), x - y/sqrt(3)) <= S? No.
    # The constraints are:
    # x + y/sqrt(3) <= S
    # x - y/sqrt(3) <= S
    # -x + y/sqrt(3) <= S
    # -x - y/sqrt(3) <= S
    # Which simplifies to:
    # |x| + |y|/sqrt(3) <= S ? No, that's for a diamond.
    # For a hexagon:
    # max( x + y/sqrt(3), x - y/sqrt(3), -x + y/sqrt(3), -x - y/sqrt(3) ) <= S
    # This is equivalent to:
    # |x| + |y|/sqrt(3) <= S is WRONG.
    # Let's test (S, 0): S + 0 = S. OK.
    # Test (S/2, S*sqrt(3)/2): S/2 + (S*sqrt(3)/2)/sqrt(3) = S/2 + S/2 = S. OK.
    # Test (0, S*sqrt(3)/2): 0 + S/2 = S/2 <= S. OK.
    # So the condition is indeed: |x| + |y|/sqrt(3) <= S is NOT correct for all points.
    # Wait, the support function of a regular hexagon with circumradius R (S) is h(u) = R * max(|u_x|, |u_x/2 + u_y*sqrt(3)/2|, |-u_x/2 + u_y*sqrt(3)/2|) ... no.
    
    # Let's stick to the line equations.
    # The 6 sides are:
    # 1. y = sqrt(3)/2 * S  (top) -> y <= sqrt(3)/2 * S
    # 2. y = -sqrt(3)/2 * S (bottom) -> y >= -sqrt(3)/2 * S
    # 3. x = S (right) -> x <= S
    # 4. x = -S (left) -> x >= -S
    # 5. y = -sqrt(3)(x - S) -> sqrt(3)x + y <= sqrt(3)S
    # 6. y = -sqrt(3)(x + S) -> -sqrt(3)x + y >= -sqrt(3)S -> sqrt(3)x - y <= sqrt(3)S
    
    # So for a point (px, py), the required S is the maximum of:
    # 2*py/sqrt(3)
    # -2*py/sqrt(3)
    # px
    # -px
    # (sqrt(3)*px + py)/sqrt(3) = px + py/sqrt(3)
    # (sqrt(3)*px - py)/sqrt(3) = px - py/sqrt(3)
    # (sqrt(3)*(-px) + py)/sqrt(3) = -px + py/sqrt(3)
    # (sqrt(3)*(-px) - py)/sqrt(3) = -px - py/sqrt(3)
    
    # So S_req = max( 2*|py|/sqrt(3), |px|, |px + py/sqrt(3)|, |px - py/sqrt(3)| )
    # Actually, |px + py/sqrt(3)| and |px - py/sqrt(3)| cover the slanted sides.
    # And 2*|py|/sqrt(3) covers the top/bottom.
    # And |px| covers the left/right.
    
    max_s = 0.0
    for x, y, angle in inner:
        # Generate 6 vertices of the inner hexagon
        for k in range(6):
            a = math.radians(60 * k)
            vx = x + math.cos(a)
            vy = y + math.sin(a)
            
            s1 = abs(vx)
            s2 = 2 * abs(vy) / math.sqrt(3)
            s3 = abs(vx + vy / math.sqrt(3))
            s4 = abs(vx - vy / math.sqrt(3))
            
            current_s = max(s1, s2, s3, s4)
            if current_s > max_s:
                max_s = current_s
                
    # Add a tiny epsilon to ensure strict containment if needed, though <= is usually fine
    outer_side = max_s + 1e-9
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END

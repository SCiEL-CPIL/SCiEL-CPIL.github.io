"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # For packing unit hexagons in a hexagonal container:
    # Use hexagonal lattice arrangement
    # Axial coordinates (q, r) -> Cartesian:
    #   x = sqrt(3) * (q + r/2)
    #   y = 1.5 * r
    # 
    # A hexagonal cluster of "radius" R contains 3R(R-1)+1 hexagons
    # The outer hexagon side length needed: R*sqrt(3) + 1
    #
    # For N=11: R=3 gives 19 hexagons capacity, side = 3*sqrt(3)+1 ≈ 6.196
    # We place 11 hexagons in a centered pattern.
    
    sqrt3 = math.sqrt(3)
    
    # Generate hexagonal lattice positions in order of distance from center
    # Use axial coordinates
    def hex_to_cart(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    # Generate all positions in a hexagonal cluster of radius R
    # and sort by distance from center to pick the closest N
    best_result = None
    
    for R in range(1, 10):
        if time.time() > deadline:
            break
        
        # Generate all axial coordinates within hexagonal radius R
        positions = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                if abs(q + r) <= R:
                    x, y = hex_to_cart(q, r)
                    dist = math.sqrt(x * x + y * y)
                    positions.append((dist, q, r, x, y))
        
        # Sort by distance from center
        positions.sort(key=lambda p: p[0])
        
        # Take the first N positions
        if len(positions) < N:
            continue
        
        selected = positions[:N]
        
        # Calculate required outer hexagon side length
        # The outer hexagon (centered at origin, with flat sides parallel to lattice)
        # needs to contain all selected hexagons
        # For a hexagon with side s, the constraint is that for each inner hex center
        # at (x,y), the point must satisfy the hexagon inequality.
        # 
        # A regular hexagon centered at origin with "radius" (center to vertex) = s:
        # The six sides are at distance s*sqrt(3)/2 from center.
        # For a point (x,y) to be inside, we need:
        #   |x| <= s*sqrt(3)/2  (for the two vertical-ish sides)
        #   |y| <= s*sqrt(3)/2  (wait, this isn't right for all orientations)
        #
        # Let me use the standard form. A regular hexagon with side s, oriented with
        # flat top/bottom (pointy left/right), centered at origin:
        # The vertices are at angles 0, 60, 120, 180, 240, 300 degrees at distance s.
        # 
        # Actually, let me use the half-width approach.
        # For a hexagon with circumradius s (center to vertex = s):
        # The inradius (center to side midpoint) = s * sqrt(3) / 2
        #
        # The hexagon contains point (x,y) if and only if:
        # For each of the 6 normal directions (at angles 0, 60, 120, 180, 240, 300),
        # the projection of (x,y) onto that normal is <= inradius = s*sqrt(3)/2
        #
        # But we also need to account for the inner hexagon's extent. Each inner hex
        # has circumradius 1, so the farthest point from the outer center is at
        # distance (dist_to_center + 1) in the direction away from center.
        #
        # More precisely, for the outer hexagon to contain an inner hex centered at
        # (x,y) with circumradius 1, we need that the entire inner hex is inside.
        # The worst case is the vertex of the inner hex that is farthest from the
        # outer hexagon's center.
        #
        # A simpler sufficient condition: the distance from outer center to the
        # farthest point of the inner hex <= the inradius of the outer hex in that
        # direction. But this is complex.
        #
        # Let's use a simpler approach: the outer hexagon with circumradius S contains
        # the inner hex if the Minkowski sum condition is met. For hexagons, if both
        # are aligned, the condition is that the outer hex's inradius >= inner hex's
        # inradius + distance between centers projected appropriately.
        #
        # For simplicity and correctness, let's compute: for each inner hex center (x,y),
        # the maximum distance from origin to any point of that inner hex is 
        # sqrt(x^2+y^2) + 1 (upper bound). So we need S >= max(sqrt(x^2+y^2) + 1).
        # But this gives a circle-like bound. For a hexagon, we can do better.
        #
        # Let's use the exact hexagon containment test.
        # Outer hexagon: circumradius S, flat-top orientation (vertices at 0,60,120,180,240,300)
        # The six side normals are at angles 30, 90, 150, 210, 270, 330 degrees.
        # Inradius = S * sqrt(3) / 2
        #
        # For point (x,y) to be inside: for each normal direction n_i, dot((x,y), n_i) <= inradius
        #
        # For the inner hex centered at (x,y) with circumradius 1 to be fully inside,
        # we need: for each normal direction n_i, dot((x,y), n_i) + max_dot <= inradius
        # where max_dot is the maximum projection of any point in the inner hex onto n_i.
        # For a regular hexagon with circumradius 1, the maximum projection onto any
        # direction is 1 (at the vertex in that direction). Actually, it's the support
        # function: for a regular hexagon with circumradius 1, h(theta) = max over vertices.
        #
        # The vertices of a unit hexagon (circumradius 1) are at angles 0, 60, 120, 180, 240, 300.
        # The support function h(theta) = max_i cos(theta - alpha_i) where alpha_i are vertex angles.
        # For a regular hexagon, h(theta) = cos(theta mod 60°) for theta in [0, 60°], etc.
        # Actually h(theta) = cos((theta mod 60°) - 30°) * ... no.
        #
        # Let me just compute: vertices at angles 0, 60, 120, 180, 240, 300, each at distance 1.
        # h(theta) = max(cos(theta), cos(theta-60), cos(theta-120), cos(theta-180), cos(theta-240), cos(theta-300))
        #
        # For the outer hexagon normals at 30, 90, 150, 210, 270, 330:
        # h(30) = max(cos30, cos(-30), cos(-90), cos(-150), cos(-210), cos(-270))
        #       = max(0.866, 0.866, 0, -0.866, 0.5, 0) = 0.866 = sqrt(3)/2
        # h(90) = max(cos90, cos30, cos(-30), cos(-90), cos(-150), cos(-210))
        #       = max(0, 0.866, 0.866, 0, -0.866, 0.5) = 0.866
        # By symmetry, h(30+60k) = sqrt(3)/2 for all k.
        #
        # So the condition is: for each normal direction n_i (at angle 30+60k),
        #   dot((x,y), n_i) + sqrt(3)/2 <= S * sqrt(3) / 2
        #   => dot((x,y), n_i) <= S * sqrt(3)/2 - sqrt(3)/2 = (S-1) * sqrt(3)/2
        #
        # So we need: max_i |dot((x,y), n_i)| <= (S-1) * sqrt(3)/2 for all inner hex centers
        # Wait, it's not absolute value, it's just the projection in the outward normal direction.
        #
        # For the outer hexagon centered at origin with normals at 30,90,150,210,270,330:
        # n_0 = (cos30, sin30) = (sqrt(3)/2, 1/2)
        # n_1 = (cos90, sin90) = (0, 1)
        # n_2 = (cos150, sin150) = (-sqrt(3)/2, 1/2)
        # n_3 = (cos210, sin210) = (-sqrt(3)/2, -1/2)
        # n_4 = (cos270, sin270) = (0, -1)
        # n_5 = (cos330, sin330) = (sqrt(3)/2, -1/2)
        #
        # Condition: dot((x,y), n_i) + sqrt(3)/2 <= S * sqrt(3)/2 for all i
        # => dot((x,y), n_i) <= (S-1) * sqrt(3)/2
        #
        # The maximum S needed is: S = 1 + (2/sqrt(3)) * max_i dot((x,y), n_i) over all inner hexes and all i
        # But we take the max over all (x,y) and all i.
        
        # Compute the required S
        max_proj = 0.0
        for (_, _, _, x, y) in selected:
            # Compute projections onto the 6 normal directions
            projs = [
                x * sqrt3 / 2.0 + y * 0.5,      # n_0
                y,                                 # n_1
                -x * sqrt3 / 2.0 + y * 0.5,       # n_2
                -x * sqrt3 / 2.0 - y * 0.5,       # n_3
                -y,                                # n_4
                x * sqrt3 / 2.0 - y * 0.5,        # n_5
            ]
            max_proj = max(max_proj, max(projs))
        
        S = 1.0 + (2.0 / sqrt3) * max_proj
        
        if best_result is None or S < best_result[2]:
            # Build the inner list
            inner = []
            for (_, q, r, x, y) in selected:
                inner.append([x, y, 0.0])  # angle 0 (no rotation needed for aligned hexagons)
            
            best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Return the best result found
    if best_result is not None:
        return best_result
    else:
        # Fallback: simple linear arrangement
        inner = [[sqrt3 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END

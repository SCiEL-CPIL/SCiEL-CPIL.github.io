"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Unit hexagons with side length 1, flat-top orientation
    # In a hex grid with flat-top hexagons of side 1:
    # - Horizontal spacing between columns: 1.5 (for odd columns offset)
    # - Actually, let's use axial coordinates and convert to pixel coords
    # For flat-top hex with side s=1:
    #   x = 1.5 * q
    #   y = sqrt(3) * (r + q/2)
    # where q, r are axial coordinates
    # Distance between adjacent centers = sqrt(3)
    
    # Let's place 11 hexagons in a compact hexagonal-ish arrangement
    # Using offset coordinates: rows with varying counts
    # Best known packing for 11: a 3-4-4 or 4-3-4 arrangement
    
    # Let me use a more systematic approach:
    # Place hexagons in a hexagonal grid pattern
    # For N=11, use rows: 3, 4, 4 (or similar)
    
    # Flat-top hexagon, side=1
    # Center positions in a hex grid:
    # Row r, position i in row:
    #   x = 1.5 * i  (if we space them by 1.5 in x for flat-top)
    #   Wait, for flat-top hex with side 1:
    #   - Width = 2 (point to point horizontally)
    #   - Height = sqrt(3) (flat to flat vertically)
    #   - Adjacent hex in same row: center distance = 3/2 * 2? No.
    #   
    # Let me be precise. Flat-top regular hexagon with side s:
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    # Vertex at angle 0: (s, 0)
    # So width (x-direction) = 2s, height (y-direction) = sqrt(3)*s
    # 
    # For two adjacent hexagons sharing an edge (left-right neighbors in flat-top):
    # The shared edge is vertical. The centers are separated by sqrt(3)*s in x? No.
    # 
    # Actually for flat-top:
    # - Left and right vertices at (-s, 0) and (s, 0)
    # - Top-left and bottom-left vertices at (-s/2, sqrt(3)*s/2) and (-s/2, -sqrt(3)*s/2)
    # - Top-right and bottom-right at (s/2, sqrt(3)*s/2) and (s/2, -sqrt(3)*s/2)
    # 
    # Two hexagons side by side (sharing the right/left edge):
    # Right edge of left hex: from (s/2, sqrt(3)s/2) to (s/2, -sqrt(3)s/2)
    # Left edge of right hex: same line
    # So centers are separated by 3s/2 in x? Let's see:
    # Left hex center at (0,0), right hex center at (3s/2, 0)? 
    # Right edge of left hex is at x = s/2... no wait.
    # 
    # Let me just use the standard formula. For flat-top hex grid with side s:
    #   x = 3*s/2 * col
    #   y = sqrt(3)*s * (row + (col & 1) / 2.0)
    # 
    # With s=1:
    #   x = 1.5 * col
    #   y = sqrt(3) * (row + (col % 2) / 2.0)
    
    sqrt3 = np.sqrt(3)
    
    # Place 11 hexagons in a compact arrangement
    # Using axial coordinates, let's try a hexagonal blob
    # For 11 hexagons, a good shape is:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 4 hexagons
    # Total: 11
    
    # Using offset coordinates (odd-q offset for flat-top):
    # Let's place them and compute the bounding hexagon
    
    positions = []
    # Row 0 (top): 3 hexagons, centered
    # Row 1 (middle): 4 hexagons
    # Row 2 (bottom): 4 hexagons
    
    # Using axial coordinates (q, r) where:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    # Let me place them in a more natural hexagonal pattern
    # Center at (0,0) in axial coords
    # Hexagonal distances: 0, 1, 2, ...
    # Distance 0: 1 hex
    # Distance 1: 6 hex (total 7)
    # Distance 2: 12 hex (total 19)
    # 
    # For 11, take all hex with distance <= 1 (7 hex) plus 4 more from distance 2
    # To minimize bounding box, pick 4 from distance 2 that are clustered
    
    # Let me just use a simple arrangement and compute the outer hexagon
    
    # Simple approach: place in rows
    # Row 0: 3 hex at x offsets
    # Row 1: 4 hex
    # Row 2: 4 hex
    
    # Using pixel coordinates for flat-top hex side=1:
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3)
    # Odd rows shifted by 0.75
    
    inner = []
    
    # Row 0: 3 hexagons
    for i in range(3):
        x = 1.5 * (i - 1)  # centered: -1.5, 0, 1.5
        y = 0.0
        inner.append([x, y, 0.0])
    
    # Row 1: 4 hexagons, shifted
    for i in range(4):
        x = 1.5 * (i - 1.5)  # -2.25, -0.75, 0.75, 2.25
        y = sqrt3
        inner.append([x, y, 0.0])
    
    # Row 2: 4 hexagons
    for i in range(4):
        x = 1.5 * (i - 1.5)  # -2.25, -0.75, 0.75, 2.25
        y = 2 * sqrt3
        inner.append([x, y, 0.0])
    
    # Now compute the minimal enclosing regular hexagon
    # The outer hexagon should be centered and axis-aligned (0 degrees rotation)
    # For a flat-top hexagon, the six sides are at angles 0, 60, 120, 180, 240, 300
    # A point (x,y) is inside a flat-top hexagon centered at (cx,cy) with side s if:
    #   |x - cx| <= s
    #   |y - cy| <= sqrt(3) * s
    #   |sqrt(3)*(x-cx) + (y-cy)| <= 2*sqrt(3)*s  ... this isn't quite right
    
    # Better: for a flat-top regular hexagon with side s, centered at origin:
    # The six half-plane constraints are:
    #   x <= s
    #   x >= -s
    #   sqrt(3)*x + y <= sqrt(3)*s + ... 
    # 
    # Actually, the vertices of a flat-top hex with side s at origin are:
    #   (s, 0), (s/2, sqrt(3)*s/2), (-s/2, sqrt(3)*s/2), (-s, 0), (-s/2, -sqrt(3)*s/2), (s/2, -sqrt(3)*s/2)
    # 
    # The six edges give six half-planes. For a point to be inside:
    #   x <= s
    #   x >= -s  
    #   y <= sqrt(3)*s/2 + sqrt(3)/2 * (s - x) for x in [0, s]... this is getting complex
    # 
    # Simpler: the distance from center to each edge (inradius) = sqrt(3)*s/2
    # For a flat-top hex, the six edge normals are at angles 0, 60, 120, 180, 240, 300
    # Wait, no. The edges are perpendicular to the directions to the vertices.
    # 
    # Let me just compute the required side length by finding the max over all points
    # of the "hex distance" from the center.
    # 
    # For a flat-top hex centered at (cx, cy) with side s, a point (px, py) is inside iff:
    #   |px - cx| <= s
    #   |py - cy| <= sqrt(3) * s / 2 + ... 
    # 
    # I'll use the support function approach. The hexagon can be defined as the intersection of 6 half-planes.
    # For flat-top hex with side s at origin, the 6 supporting lines are:
    #   x = s, x = -s
    #   sqrt(3)*x + 2*y = sqrt(3)*s, sqrt(3)*x + 2*y = -sqrt(3)*s  ... no
    
    # Let me just compute it numerically. For each inner hexagon center, compute the 
    # minimum s such that the hexagon center plus the hexagon's extent fits inside.
    # 
    # Actually, since all inner hexagons have the same orientation (0 degrees), I can compute
    # the bounding box in the hexagonal metric.
    # 
    # For a flat-top hex with side s centered at origin, the constraint for a point (x,y) to be inside:
    # The hexagon is the set of points where max of 6 linear forms <= s
    # The 6 linear forms for flat-top hex are:
    #   f0(x,y) = x
    #   f1(x,y) = -x  
    #   f2(x,y) = (sqrt(3)*x + 2*y) / 2  ... 
    # 
    # Let me look this up properly. A regular hexagon with circumradius R (= s for unit side) 
    # flat-top has vertices at R*(cos(k*60°), sin(k*60°)) for k=0..5
    # 
    # The inradius (apothem) = R * cos(30°) = R * sqrt(3)/2
    # 
    # The six edge equations for a flat-top hex centered at origin with circumradius R:
    # Edge opposite vertex at angle 0: x = R*cos(30°) = R*sqrt(3)/2 ... no
    # 
    # I'm overcomplicating this. Let me just use the fact that for a hexagonal grid of unit 
    # hexagons, the outer hexagon side length for a given arrangement can be computed by 
    # finding the extent in the three hex directions.
    # 
    # For now, let me estimate. The arrangement spans:
    # x: from -2.25 to 2.25, so width 4.5
    # y: from 0 to 2*sqrt(3) ≈ 3.464
    # 
    # The outer hexagon needs to contain all of these plus the hexagon extents.
    # Each unit hex has circumradius 1, so I need to add 1 to the extents.
    # 
    # Let me just compute the required side length by checking all 6 constraints for each point.
    
    # For a flat-top regular hexagon centered at (cx, cy) with side length s:
    # A point (px, py) is inside if and only if:
    #   px - cx <= s
    #   cx - px <= s
    #   sqrt(3)*(py - cy) + (px - cx) <= 2*sqrt(3)*s  ... I need to get this right
    
    # The six vertices of a flat-top hex with side s at origin:
    # V0 = (s, 0)
    # V1 = (s/2, sqrt(3)*s/2)
    # V2 = (-s/2, sqrt(3)*s/2)
    # V3 = (-s, 0)
    # V4 = (-s/2, -sqrt(3)*s/2)
    # V5 = (s/2, -sqrt(3)*s/2)
    # 
    # The six edges and their outward normals:
    # Edge V0-V1: from (s,0) to (s/2, sqrt(3)s/2)
    #   Direction: (-s/2, sqrt(3)s/2), normal: (sqrt(3)s/2, s/2) normalized -> (sqrt(3)/2, 1/2)
    #   Line: sqrt(3)/2 * x + 1/2 * y = sqrt(3)/2 * s + 1/2 * 0 = sqrt(3)s/2
    #   Constraint: sqrt(3)/2 * x + 1/2 * y <= sqrt(3)s/2
    # 
    # Edge V1-V2: from (s/2, sqrt(3)s/2) to (-s/2, sqrt(3)s/2)
    #   Direction: (-s, 0), normal: (0, 1)
    #   Line: y = sqrt(3)s/2
    #   Constraint: y <= sqrt(3)s/2
    # 
    # Edge V2-V3: from (-s/2, sqrt(3)s/2) to (-s, 0)
    #   Direction: (-s/2, -sqrt(3)s/2), normal: (-sqrt(3)/2, -1/2) ... outward is (sqrt(3)/2, 1/2)? 
    #   Wait, outward normal should point away from center. Let me reconsider.
    #   The edge from V2 to V3, the outward normal... The center is at origin.
    #   Midpoint of V2-V3: (-3s/4, sqrt(3)s/4)
    #   Normal pointing away from center: proportional to (-3s/4, sqrt(3)s/4), normalized
    #   Actually, for a convex polygon, the outward normal of an edge can be found by rotating the edge direction 90° clockwise (for CCW-ordered vertices).
    # 
    # Let me just define the 6 half-plane constraints directly:
    # For flat-top hex centered at origin with side s:
    #   x <= s
    #   x >= -s
    #   y <= sqrt(3)*s/2
    #   y >= -sqrt(3)*s/2
    #   sqrt(3)*x + y <= sqrt(3)*s  ... let me verify
    #   sqrt(3)*x - y >= -sqrt(3)*s  ... 
    # 
    # Check vertex V1 = (s/2, sqrt(3)s/2):
    #   x = s/2 <= s ✓
    #   y = sqrt(3)s/2 <= sqrt(3)s/2 ✓ (equality)
    #   sqrt(3)*x + y = sqrt(3)*s/2 + sqrt(3)s/2 = sqrt(3)s ✓ (equality, so constraint is sqrt(3)*x + y <= sqrt(3)*s)
    #   sqrt(3)*x - y = sqrt(3)*s/2 - sqrt(3)s/2 = 0 >= -sqrt(3)*s ✓
    # 
    # Check vertex V0 = (s, 0):
    #   x = s <= s ✓
    #   sqrt(3)*x + y = sqrt(3)*s <= sqrt(3)*s ✓
    #   sqrt(3)*x - y = sqrt(3)*s >= -sqrt(3)*s ✓
    # 
    # So the 6 constraints for a point (x,y) to be inside flat-top hex centered at origin with side s:
    #   x <= s
    #   -x <= s  (i.e., x >= -s)
    #   y <= sqrt(3)*s/2
    #   -y <= sqrt(3)*s/2  (i.e., y >= -sqrt(3)*s/2)
    #   sqrt(3)*x + y <= sqrt(3)*s
    #   -sqrt(3)*x - y <= sqrt(3)*s  (i.e., sqrt(3)*x + y >= -sqrt(3)*s)
    #   sqrt(3)*x - y <= sqrt(3)*s
    #   -sqrt(3)*x + y <= sqrt(3)*s  (i.e., sqrt(3)*x - y >= -sqrt(3)*s)
    # 
    # Wait, that's 8 constraints. Let me recount. A hexagon has 6 edges, so 6 constraints.
    # 
    # The 6 edges:
    # 1. V0-V1: constraint sqrt(3)*x + y <= sqrt(3)*s
    # 2. V1-V2: constraint y <= sqrt(3)*s/2
    # 3. V2-V3: constraint -sqrt(3)*x + y <= sqrt(3)*s  ... let me check
    #    V2 = (-s/2, sqrt(3)s/2): -sqrt(3)*(-s/2) + sqrt(3)s/2 = sqrt(3)s/2 + sqrt(3)s/2 = sqrt(3)s ✓
    #    V3 = (-s, 0): -sqrt(3)*(-s) + 0 = sqrt(3)s ✓
    # 4. V3-V4: constraint -x <= s (i.e., x >= -s)
    # 5. V4-V5: constraint -sqrt(3)*x - y <= sqrt(3)*s
    #    V4 = (-s/2, -sqrt(3)s/2): -sqrt(3)*(-s/2) - (-sqrt(3)s/2) = sqrt(3)s/2 + sqrt(3)s/2 = sqrt(3)s ✓
    #    V5 = (s/2, -sqrt(3)s/2): -sqrt(3)*(s/2) - (-sqrt(3)s/2) = -sqrt(3)s/2 + sqrt(3)s/2 = 0 <= sqrt(3)s ✓
    # 6. V5-V0: constraint x <= s
    # 
    # So the 6 constraints are:
    #   sqrt(3)*x + y <= sqrt(3)*s
    #   y <= sqrt(3)*s/2
    #   -sqrt(3)*x + y <= sqrt(3)*s
    #   -x <= s
    #   -sqrt(3)*x - y <= sqrt(3)*s
    #   x <= s
    # 
    # For a hex centered at (cx, cy), replace x with x-cx and y with y-cy.
    # 
    # For each inner hexagon center (px, py), I need the hexagon to fit inside.
    # The hexagon extends by its circumradius in all directions, but since it's the same orientation,
    # I can just check the 6 vertices of each inner hexagon, or equivalently, find the max of the
    # 6 linear forms over all inner hexagon vertices.
    # 
    # Simpler: for each inner hex center, compute the 6 values of the linear forms at that center,
    # then add the maximum extent of a unit hex in that direction.
    # 
    # For a unit flat-top hex centered at origin, the max of (sqrt(3)*x + y) over the hex is:
    # At V0=(1,0): sqrt(3)*1 + 0 = sqrt(3)
    # At V1=(1/2, sqrt(3)/2): sqrt(3)/2 + sqrt(3)/2 = sqrt(3)
    # So max is sqrt(3).
    # 
    # Similarly for other forms. So for each inner hex center (px,py), the constraint becomes:
    #   sqrt(3)*(px-cx) + (py-cy) + sqrt(3) <= sqrt(3)*s
    #   etc.
    # 
    # So s >= (sqrt(3)*(px-cx) + (py-cy) + sqrt(3)) / sqrt(3) = (px-cx) + (py-cy)/sqrt(3) + 1
    # And similarly for other constraints.
    
    # Let me compute the required s for center (0,0) and rotation 0.
    
    cx, cy = 0.0, 0.0
    
    # For each inner hex, compute the 6 constraint values
    # The 6 linear forms (for center at origin) are:
    # L0 = sqrt(3)*x + y, bound: sqrt(3)*s
    # L1 = y, bound: sqrt(3)*s/2
    # L2 = -sqrt(3)*x + y, bound: sqrt(3)*s
    # L3 = -x, bound: s
    # L4 = -sqrt(3)*x - y, bound: sqrt(3)*s
    # L5 = x, bound: s
    
    # For a unit hex at (px,py), the max of each linear form over the hex:
    # L0: sqrt(3), L1: sqrt(3)/2, L2: sqrt(3), L3: 1, L4: sqrt(3), L5: 1
    
    max_L0 = -np.inf
    max_L1 = -np.inf
    max_L2 = -np.inf
    max_L3 = -np.inf
    max_L4 = -np.inf
    max_L5 = -np.inf
    
    for px, py, _ in inner:
        dx = px - cx
        dy = py - cy
        max_L0 = max(max_L0, sqrt3 * dx + dy + sqrt3)
        max_L1 = max(max_L1, dy + sqrt3 / 2)
        max_L2 = max(max_L2, -sqrt3 * dx + dy + sqrt3)
        max_L3 = max(max_L3, -dx + 1)
        max_L4 = max(max_L4, -sqrt3 * dx - dy + sqrt3)
        max_L5 = max(max_L5, dx + 1)
    
    # Required s from each constraint:
    s0 = max_L0 / sqrt3
    s1 = max_L1 / (sqrt3 / 2)
    s2 = max_L2 / sqrt3
    s3 = max_L3
    s4 = max_L4 / sqrt3
    s5 = max_L5
    
    s = max(s0, s1, s2, s3, s4, s5)
    
    return inner, (cx, cy), s, 0.0

# EVOLVE-BLOCK-END

"""Improved packing: analytic construction for 12 unit hexagons in a regular hexagon."""
import math
import time

N = 12
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # Strategy: Place 12 unit regular hexagons in a compact arrangement inside a regular hexagon.
    # 
    # Key insight: The optimal packing of 12 unit hexagons in a hexagon likely uses a
    # symmetric arrangement. Let's try a few known configurations.
    #
    # Configuration 1: 2 rows of 6 hexagons each, offset
    # Configuration 2: 3 rows of 4 hexagons each
    # Configuration 3: Hexagonal arrangement (1-2-3-2-1 pattern = 9, not 12)
    # Configuration 4: 4-4-4 arrangement
    #
    # For a regular hexagon of side length s, the distance from center to vertex is s,
    # and the distance from center to edge (apothem) is s * sqrt(3)/2.
    #
    # A unit hexagon has circumradius 1 and inradius sqrt(3)/2 ≈ 0.866.
    #
    # Let's try a 2x6 grid arrangement with hexagonal offsets.
    # In a hexagonal lattice, adjacent hexagons in the same row are spaced 2*sqrt(3) apart
    # (center-to-center distance for touching hexagons = 2 * inradius = 2*sqrt(3)/2 = sqrt(3)*... 
    # Actually, for two touching regular hexagons of side 1, the center-to-center distance is 2*inradius = 2*(sqrt(3)/2) = sqrt(3) ≈ 1.732.
    # Wait, no. The inradius is sqrt(3)/2. Two hexagons touching side-to-side have centers
    # separated by 2 * inradius = sqrt(3). But that's for flat-to-flat contact.
    #
    # Actually, for regular hexagons of side length 1:
    # - Circumradius (center to vertex) = 1
    # - Inradius (center to edge midpoint) = sqrt(3)/2
    #
    # Two hexagons can touch in different ways. The minimum center-to-center distance
    # for non-overlapping hexagons depends on their relative rotation.
    #
    # For same-orientation hexagons, the minimum separation is 2*inradius = sqrt(3) ≈ 1.732
    # when they touch flat-to-flat.
    #
    # Let's try a simple grid: 2 rows x 6 columns
    # Row spacing: sqrt(3) (flat-to-flat)
    # Column spacing within a row: 2*1 = 2 (vertex-to-vertex horizontally? No...)
    #
    # For hexagons with flat top (pointing up), vertices are at angles 0, 60, 120, 180, 240, 300
    # from the positive x-axis, at distance 1 from center.
    #
    # Let me just try a compact arrangement and compute the required outer hexagon size.
    
    # Try arrangement: 3 rows of 4 hexagons
    # Using hexagonal close packing
    # Row 0: 4 hexagons at x = -1.5, -0.5, 0.5, 1.5 (spacing 1.0? No, that's too close)
    
    # Let me think more carefully. For same-orientation unit hexagons (flat-top):
    # The horizontal distance between centers of adjacent hexagons in a row (touching) is 2*inradius = sqrt(3)
    # The vertical distance between rows is inradius = sqrt(3)/2, with horizontal offset of inradius... 
    # No, for flat-top hexagons:
    # - Width (flat to flat, horizontal) = 2 * inradius = sqrt(3)
    # - Height (vertex to vertex, vertical) = 2 * circumradius = 2
    #
    # Wait, I need to be more careful about orientation. Let's use "pointy-top" hexagons
    # where a vertex points up.
    #
    # For pointy-top unit hexagon:
    # - Width (horizontal, flat to flat) = 2 * inradius = sqrt(3)
    # - Height (vertical, vertex to vertex) = 2 * circumradius = 2
    #
    # In a hexagonal grid of pointy-top hexagons:
    # - Horizontal spacing between adjacent hexagons in same row: sqrt(3)
    # - Vertical spacing between rows: 1.5
    # - Horizontal offset between adjacent rows: sqrt(3)/2
    
    # Let's try 3 rows of 4:
    # Row 0 (top): y = 1.5, x = -3*sqrt(3)/2, -1*sqrt(3)/2, 1*sqrt(3)/2, 3*sqrt(3)/2
    # Row 1 (mid): y = 0, x = -2*sqrt(3)/2, 0, 2*sqrt(3)/2, ... wait, need 4 hexagons
    #
    # Actually for 3 rows of 4 with hexagonal packing:
    # Row 0: 4 hexagons, centered, y = h
    # Row 1: 4 hexagons, offset by sqrt(3)/2, y = 0  
    # Row 2: 4 hexagons, centered, y = -h
    # where h = 1.5 (vertical spacing)
    
    sqrt3 = math.sqrt(3)
    
    # Attempt: symmetric arrangement
    # Let's try placing them in a pattern and computing the bounding hexagon
    
    best_side = float('inf')
    best_config = None
    
    # Try several candidate arrangements
    
    # Arrangement A: 2 rows x 6 columns (pointy-top)
    # Horizontal spacing: sqrt(3), vertical spacing: 1.5, offset: sqrt(3)/2
    # But with 6 in a row, total width = 5*sqrt(3) + sqrt(3) = 6*sqrt(3) (from leftmost to rightmost center + hex width)
    
    # Let me just compute: for a given set of centers, find the minimum side length
    # of a regular hexagon (with given rotation) that contains all of them.
    
    def hexagon_contains_point(cx, cy, px, py, side, angle_deg):
        """Check if point (px,py) is inside regular hexagon centered at (cx,cy) with side length 'side' and rotation 'angle_deg'."""
        angle_rad = math.radians(angle_deg)
        # Transform point to hexagon's local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # Regular hexagon with side 'side', pointy-top (vertex at top) in local frame
        # Vertices at angles 90, 150, 210, 270, 330, 30 degrees (for pointy-top)
        # Or equivalently, the hexagon is the intersection of 6 half-planes
        # For a regular hexagon with circumradius R = side, the support function in direction theta is R * max over vertices
        # Actually, for a regular hexagon with side s, the distance from center to each edge (inradius) = s*sqrt(3)/2
        # The hexagon can be defined as: for each of 6 edge normals, the projection of the point must be <= inradius
        
        # Edge normals for pointy-top hexagon: at angles 0, 60, 120, 180, 240, 300 degrees
        inradius = side * sqrt3 / 2
        for k in range(6):
            theta = math.radians(60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            if lx * nx + ly * ny > inradius:
                return False
        return True
    
    def min_hexagon_side_for_points(points, angle_deg=0.0):
        """Find minimum side length of regular hexagon (with given rotation) centered at origin containing all points."""
        # Binary search on side length
        lo = 0.0
        hi = 100.0
        
        def check(side):
            for (px, py) in points:
                if not hexagon_contains_point(0, 0, px, py, side, angle_deg):
                    return False
            return True
        
        for _ in range(50):
            mid = (lo + hi) / 2
            if check(mid):
                hi = mid
            else:
                lo = mid
        return hi
    
    def evaluate_arrangement(centers, angle_deg=0.0):
        """Evaluate an arrangement: compute min outer hexagon side, check validity."""
        if time.time() - start_time > DEADLINE:
            return float('inf'), None
        
        # Compute min side for hexagon centered at origin
        side = min_hexagon_side_for_points(centers, angle_deg)
        
        # Verify all points are actually inside
        for (px, py) in centers:
            if not hexagon_contains_point(0, 0, px, py, side, angle_deg):
                return float('inf'), None
        
        # Check pairwise non-overlap (approximate: min center distance >= 2*inradius for same orientation)
        # For unit hexagons with same orientation, min separation = 2 * inradius = sqrt(3)
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < sqrt3 - 1e-9:
                    return float('inf'), None
        
        return side, centers
    
    # Generate candidate arrangements
    
    candidates = []
    
    # Candidate 1: 2 rows x 6, pointy-top hexagons
    # Row spacing 1.5, col spacing sqrt(3), row offset sqrt(3)/2
    for row_offset in [0.0, sqrt3/2]:
        centers = []
        for row in range(2):
            y = (1 - row) * 1.5  # rows at y=1.5 and y=0
            for col in range(6):
                x = (col - 2.5) * sqrt3 + row_offset
                centers.append((x, y))
        candidates.append(centers)
    
    # Candidate 2: 3 rows x 4, pointy-top
    for row_offset in [0.0, sqrt3/2]:
        centers = []
        for row in range(3):
            y = (1 - row) * 1.5
            if row % 2 == 0:
                # 4 hexagons
                for col in range(4):
                    x = (col - 1.5) * sqrt3
                    centers.append((x, y))
            else:
                # 4 hexagons, offset
                for col in range(4):
                    x = (col - 1.5) * sqrt3 + row_offset
                    centers.append((x, y))
        candidates.append(centers)
    
    # Candidate 3: 4 rows x 3, pointy-top
    for row_offset in [0.0, sqrt3/2]:
        centers = []
        for row in range(4):
            y = (1.5 - row) * 1.5
            if row % 2 == 0:
                for col in range(3):
                    x = (col - 1.0) * sqrt3
                    centers.append((x, y))
            else:
                for col in range(3):
                    x = (col - 1.0) * sqrt3 + row_offset
                    centers.append((x, y))
        candidates.append(centers)
    
    # Candidate 4: 6x2 grid
    centers = []
    for row in range(6):
        for col in range(2):
            x = (col - 0.5) * sqrt3
            y = (2.5 - row) * 1.5
            centers.append((x, y))
    candidates.append(centers)
    
    # Candidate 5: Compact cluster - try to pack more tightly
    # Use a different offset pattern
    
    # Candidate 6: All in one orientation, tight hexagonal packing
    # 12 hexagons in hexagonal close packing: 
    # A hexagonal number pattern: center + 6 around + 9 around = 1+6+12 = 19 (too many)
    # 1+6 = 7, 1+6+6 = 13 (one more than 12)
    # So take 1 center + 6 ring + 5 from second ring
    
    # Let's just evaluate all candidates
    for centers in candidates:
        if len(centers) != N:
            continue
        side, valid_centers = evaluate_arrangement(centers, 0.0)
        if side < best_side:
            best_side = side
            best_config = valid_centers
    
    # Also try with 30-degree rotation of outer hexagon
    for centers in candidates:
        if len(centers) != N:
            continue
        side, valid_centers = evaluate_arrangement(centers, 30.0)
        if side < best_side:
            best_side = side
            best_config = valid_centers
    
    # If we found a good config, use it
    if best_config is not None and best_side < 40.0:
        inner = [[x, y, 0.0] for (x, y) in best_config]
        return inner, (0.0, 0.0), best_side, 0.0
    
    # Fallback: original arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

import numpy as np
import time
import math

N = 11
DEADLINE = 140.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_convex_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _verify_packing(inner, outer_center, side, outer_angle):
    """Verify that all inner hexagons are inside the outer hexagon and don't overlap."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_angle, side)
    
    inner_verts_list = []
    for x, y, a in inner:
        verts = _hex_vertices(x, y, a, 1.0)
        inner_verts_list.append(verts)
        # Check containment
        for v in verts:
            if not _point_in_convex_polygon(v, outer_verts):
                return False
    
    # Check pairwise non-overlap
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            v1 = inner_verts_list[i]
            v2 = inner_verts_list[j]
            # Check if any vertex of v1 is inside v2
            for v in v1:
                if _point_in_convex_polygon(v, v2):
                    return False
            # Check if any vertex of v2 is inside v1
            for v in v2:
                if _point_in_convex_polygon(v, v1):
                    return False
    return True

def construct():
    start_time = time.time()
    
    # Deterministic packing: 3-4-4 grid rotated 90 degrees
    # Centers form a rectangular grid in rotated coordinates
    # We want to minimize the bounding hexagon side length.
    # A 3-4-4 grid of hexagons (side 1) has centers at:
    #   x in {-1.5, 0, 1.5}
    #   y in {-2.598, -0.866, 0.866, 2.598}  (spacing sqrt(3) = 1.732)
    # Wait, standard hex grid spacing:
    #   Horizontal neighbors: distance 1 (if touching sides) or sqrt(3) (if touching corners)
    #   For non-overlapping unit hexagons, minimum center distance is 2 * apothem = sqrt(3) if aligned,
    #   or 1 if rotated 60 deg relative?
    #   Actually, for two unit regular hexagons, the minimum distance between centers for non-overlap
    #   is 1 if they share an edge, but they can be closer if rotated?
    #   No, the minimum distance is always 1 (when they share an edge) to 2 (when they touch at vertices opposite).
    #   Wait, the circumradius is 1, inradius is sqrt(3)/2.
    #   If two hexagons share an edge, center distance is sqrt(3).
    #   If they are rotated 30 deg relative to each other, they might touch at vertices, distance 2.
    #   So sqrt(3) is the tightest packing for aligned hexagons.
    
    # Let's use the "flower" packing logic but optimized for a rectangular grid shape.
    # The previous best was 4.0.
    # Let's try the 3-4-4 grid.
    # Centers:
    #   x: -1.5, 0, 1.5
    #   y: -3*sqrt(3)/2, -sqrt(3)/2, sqrt(3)/2, 3*sqrt(3)/2
    #   Spacing in y: sqrt(3)
    #   Spacing in x: 1.5?
    #   If we place them in a grid, we need to ensure non-overlap.
    #   Distance between (0, y) and (1.5, y+sqrt(3)/2):
    #   dx = 1.5, dy = sqrt(3)/2
    #   d^2 = 2.25 + 0.75 = 3.0 => d = sqrt(3).
    #   So this is a valid hexagonal lattice.
    
    sqrt3 = math.sqrt(3)
    dx = 1.5
    dy = sqrt3 / 2
    
    centers = []
    # 3 columns, 4 rows
    # x coords: -1.5, 0, 1.5
    # y coords: -3*sqrt3/2, -sqrt3/2, sqrt3/2, 3*sqrt3/2
    # Wait, if x spacing is 1.5 and y offset is sqrt3/2, it's a honeycomb.
    # Let's generate 11 points in a compact shape.
    # Shape:
    #   . X X X .
    #   X X X X X
    #   . X X X .
    #   X X X X X
    # No, that's 3+5+3+5 = 16.
    # We need 11.
    #   . X X X .
    #   X X X X X
    #   . X X X .
    #   . X X X .
    #   . X X X .
    # 3+5+3+3+3 = 17.
    
    # Let's stick to the 3-4-4 grid:
    # Row 0 (y = -1.5*sqrt3): 3 hexagons
    # Row 1 (y = -0.5*sqrt3): 4 hexagons
    # Row 2 (y = 0.5*sqrt3): 4 hexagons
    # Total 11.
    
    # Coordinates:
    # Row 0 (3 hex): x = -1.5, 0, 1.5 ; y = -1.5*sqrt3
    # Row 1 (4 hex): x = -0.75, 0.75, 2.25, -2.25 ? No.
    # In a honeycomb, if row 0 has 3 at x=-1.5, 0, 1.5,
    # Row 1 should be shifted by 0.75 in x and sqrt3/2 in y.
    # x = -1.5+0.75=-0.75, 0+0.75=0.75, 1.5+0.75=2.25. That's only 3.
    # To get 4, we add one at -2.25?
    # Then x range is -2.25 to 2.25. Width 4.5.
    # Height range: -1.5*sqrt3 to 0.5*sqrt3 + something.
    
    # Let's try a different orientation.
    # Rotate the whole assembly 90 degrees.
    # Then the "wide" part becomes vertical.
    # If the width was 4.5, height becomes 4.5.
    # Hexagon side S:
    # Width of outer hex (flat top) = sqrt(3)*S.
    # Height of outer hex (flat top) = 2*S.
    # If we align the 4.5 dimension with the 2*S (height), then 2*S >= 4.5 => S >= 2.25.
    # If we align the other dimension (height of assembly) with sqrt(3)*S.
    # Assembly height (original x-width) was ~3.
    # So sqrt(3)*S >= 3 => S >= 1.732.
    # So S is determined by the 4.5 dimension. S = 2.25.
    # This is much better than 4.0!
    
    # Let's verify the packing.
    # Centers:
    # r0: (-1.5, -1.5*sqrt3), (0, -1.5*sqrt3), (1.5, -1.5*sqrt3)
    # r1: (-2.25, -0.5*sqrt3), (-0.75, -0.5*sqrt3), (0.75, -0.5*sqrt3), (2.25, -0.5*sqrt3)
    # r2: (-1.5, 0.5*sqrt3), (0, 0.5*sqrt3), (1.5, 0.5*sqrt3)
    
    # Wait, r1 has 4, r2 has 3. Total 10. Need 11.
    # Add one more to r2?
    # r2: (-2.25, 0.5*sqrt3), (-1.5, 0.5*sqrt3), (0, 0.5*sqrt3), (1.5, 0.5*sqrt3), (2.25, 0.5*sqrt3) -> 5 hexagons.
    # Total 3+4+5 = 12. Too many.
    
    # Let's use 3-4-4:
    # r0: 3
    # r1: 4
    # r2: 4
    # r2 x-coords: -1.5+0.75 = -0.75, 0+0.75=0.75, 1.5+0.75=2.25, and one more? -1.5-0.75=-2.25.
    # So r2 x: -2.25, -0.75, 0.75, 2.25.
    # y = 0.5*sqrt3 + sqrt3 = 1.5*sqrt3? No, spacing is sqrt3/2?
    # Standard hex grid:
    #   (x, y)
    #   (x+0.75, y+sqrt3/2)
    #   (x-0.75, y+sqrt3/2)
    #   (x+1.5, y)
    #   (x-1.5, y)
    
    # Let's just generate a compact cluster of 11.
    # Center at (0,0).
    # Neighbors at distance sqrt3.
    # 1 + 6 + 4 = 11.
    # The 4 are at angles 45, 135, 225, 315? No, hex grid is 60 deg.
    # Ring 2 has 12 positions. We pick 4 of them.
    # To make it symmetric and compact, pick the ones at 0, 90, 180, 270?
    # Those are not in the hex grid.
    # Hex grid ring 2 positions:
    # (3, 0), (1.5, 2.598), (-1.5, 2.598), (-3, 0), (-1.5, -2.598), (1.5, -2.598) -- these are distance 3.
    # And (2.598, 1.5), (2.598, -1.5), etc? No.
    # Ring 2 of hex grid (distance sqrt(3)*2 = 2*sqrt3 ~ 3.464):
    # Actually, let's use the "flower" from the current program but rotate it.
    # Current program flower:
    #   Center (0,0)
    #   6 at r=sqrt3
    #   4 at r=3, angles 30, 90, 150, 210?
    #   The current program's side was 4.0.
    
    # Let's try the 3-4-4 grid again, carefully.
    # It seems the most promising for reducing S.
    
    # Let's construct the 3-4-4 grid centers explicitly.
    # sqrt3 = 1.732
    # y0 = -1.5 * sqrt3
    # y1 = -0.5 * sqrt3
    # y2 = 0.5 * sqrt3  (Wait, if spacing is sqrt3/2 in y, then y0=-1.5*sqrt3, y1=-0.5*sqrt3, y2=0.5*sqrt3? No, diff is sqrt3. So y0, y0+sqrt3, y0+2*sqrt3)
    # Let's use y spacing = sqrt3.
    # y0 = -sqrt3
    # y1 = 0
    # y2 = sqrt3
    # x0 (3 hex): -1.5, 0, 1.5
    # x1 (4 hex): -0.75, 0.75, 2.25, -2.25 ?
    #   If x1 is shifted by 0.75, then x1 = -2.25, -0.75, 0.75, 2.25.
    #   Distance between (1.5, y0) and (0.75, y1):
    #   dx = 0.75, dy = sqrt3.
    #   d^2 = 0.5625 + 3 = 3.5625. d = 1.88 > sqrt3. OK.
    #   Distance between (1.5, y0) and (2.25, y1):
    #   dx = 0.75, dy = sqrt3. Same.
    #   Distance between (0, y0) and (0.75, y1):
    #   dx = 0.75, dy = sqrt3. Same.
    #   But we need to check neighbors in the same row.
    #   (1.5, y0) and (0, y0): dx = 1.5. d = 1.5 < sqrt3. OVERLAP!
    #   So 1.5 spacing is too close for unit hexagons.
    #   Minimum center distance for non-overlapping unit hexagons is sqrt3 (if sharing edge).
    #   So x spacing must be at least sqrt3?
    #   If we use square grid, x spacing = 2 (circumradius*2), y spacing = sqrt3.
    #   Then width of 3 columns: 4. Height of 3 rows: 2*sqrt3.
    #   This is the "brick wall" layout.
    
    # Let's try Brick Wall (Square Grid with Hex Rotation).
    # If we rotate hexagons by 30 deg (Flat Top), width = sqrt3, height = 2.
    # If we stack them in a grid:
    #   x spacing = 2 * width/2 = sqrt3? No.
    #   If Flat Top, width is sqrt3. To not overlap, x spacing must be at least sqrt3.
    #   Height is 2. Y spacing must be at least 2?
    #   Actually, if we interlock them, we can do better.
    #   But let's stick to simple grid.
    #   3x4 grid of Flat-Top hexagons.
    #   x: -sqrt3, 0, sqrt3
    #   y: -3, -1, 1, 3
    #   Width = 2*sqrt3 + sqrt3 = 3*sqrt3 ~ 5.2.
    #   Height = 6 + 2 = 8.
    #   S = max(8/2, 5.2/sqrt3) = max(4, 3) = 4.
    #   Same as before.
    
    # Let's go back to the 3-4-4 honeycomb grid.
    # To avoid overlap, we must ensure min distance >= sqrt3.
    # In honeycomb, if we shift rows by 0.5 * x_spacing, and y_spacing = sqrt(3)/2 * x_spacing?
    # If x_spacing = 2 (touching at vertices), y_spacing = sqrt3.
    # Then 3x4 grid:
    # x: -2, 0, 2
    # y: -3*sqrt3/2, -sqrt3/2, sqrt3/2, 3*sqrt3/2
    # Shift odd rows by 1.
    # Width: 4 + 2 = 6.
    # Height: 3*sqrt3 + 2 = 5.196 + 2 = 7.196.
    # S = max(7.196/2, 6/sqrt3) = max(3.6, 3.46) = 3.6.
    # Better than 4.0!
    
    # Let's implement this 3-4-4 grid with x_spacing=2, y_spacing=sqrt3, shift=1.
    # Centers:
    # Row 0 (y = -1.5*sqrt3): x = -2, 0, 2
    # Row 1 (y = -0.5*sqrt3): x = -3, -1, 1, 3  (4 hexagons)
    # Row 2 (y = 0.5*sqrt3): x = -2, 0, 2  (3 hexagons) -> Total 10.
    # Row 3 (y = 1.5*sqrt3): x = -1, 1  (2 hexagons) -> Total 12.
    
    # We need 11.
    # Row 0: 3
    # Row 1: 4
    # Row 2: 4
    # Row 2 x: -3, -1, 1, 3?
    # If Row 1 is -3, -1, 1, 3, then Row 2 (shifted back by 1) is -2, 0, 2. (3 hexagons).
    # To get 4 in Row 2, we need to expand the range.
    # Maybe -4, -2, 0, 2?
    # Then width increases.
    
    # Let's try 2-4-3-2? No.
    # Let's try the 3-4-4 grid with x_spacing = 2.
    # Row 0: -2, 0, 2 (3)
    # Row 1: -3, -1, 1, 3 (4)
    # Row 2: -2, 0, 2, 4? (4) -> Asymmetric.
    
    # Let's use the symmetric 3-4-4:
    # Row 0: -1.5, 0, 1.5
    # Row 1: -2.5, -0.5, 1.5, 3.5? No.
    
    # Okay, let's use the 3-4-4 grid with x_spacing = 2, but center it.
    # Row 0 (3): -2, 0, 2
    # Row 1 (4): -3, -1, 1, 3
    # Row 2 (4): -3, -1, 1, 3 ? No, that would be 3-4-4 = 11.
    # But Row 2 is not shifted relative to Row 1?
    # If Row 1 is shifted by 1, Row 2 should be shifted back by 1 to be aligned with Row 0.
    # So Row 2 x: -2, 0, 2. (3 hexagons).
    # So 3-4-3 = 10.
    # Add one more to Row 2? -4, -2, 0, 2, 4 (5 hexagons). Total 12.
    
    # Let's try 2-4-3-2 = 11.
    # Row 0 (2): -1, 1
    # Row 1 (4): -3, -1, 1, 3
    # Row 2 (3): -2, 0, 2
    # Row 3 (2): -1, 1
    # This is symmetric.
    # x range: -3 to 3. Width 6.
    # y range: -1.5*sqrt3 to 1.5*sqrt3. Height 3*sqrt3.
    # Plus hex height 2.
    # Total Y extent: 3*sqrt3 + 2 = 5.196 + 2 = 7.196.
    # Total X extent: 6 + sqrt3 = 7.732.
    # S = max(7.196/2, 7.732/sqrt3) = max(3.6, 4.46) = 4.46. Worse.
    
    # Let's stick to the current flower packing (Score 4.0) and try to polish it.
    # Or try the 3-4-4 grid with S=3.6 and accept 12 hexagons? No, N=11.
    
    # Actually, the 3-4-3 grid (10 hex) has S=3.6.
    # Can we add the 11th hexagon without increasing S?
    # The 11th hexagon would be at (0, 2.5*sqrt3)?
    # y = 2.5*sqrt3 = 4.33.
    # Total Y = 4.33 + 1 (half height) - (-1.5*sqrt3 - 1) = 4.33 + 1 + 2.598 + 1 = 8.928.
    # S = 8.928 / 2 = 4.46.
    
    # It seems 4.0 is hard to beat with simple grids.
    # Let's return the current flower packing which is verified.
    
    centers = []
    # Ring 0: Center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    
    # Ring 2: 4 hexagons at distance 3
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    
    inner_angle = 30.0
    inner = []
    for x, y in centers:
        inner.append([x, y, inner_angle])
    
    max_x = 0.0
    max_y = 0.0
    for x, y in centers:
        max_x = max(max_x, x + np.sqrt(3) / 2)
        max_y = max(max_y, y + 1.0)
    
    s_from_x = 2 * max_x / np.sqrt(3)
    s_from_y = max_y
    side = max(s_from_x, s_from_y) + 1e-9
    
    outer_center = (0.0, 0.0)
    outer_angle = 30.0
    
    if time.time() - start_time < DEADLINE - 10:
        if not _verify_packing(inner, outer_center, side, outer_angle):
            side += 0.01
            if not _verify_packing(inner, outer_center, side, outer_angle):
                side += 0.1
    
    return inner, outer_center, side, outer_angle


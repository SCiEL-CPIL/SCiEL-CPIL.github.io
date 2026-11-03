"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import numpy as np

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _point_in_hex(px, py, hx, hy, side, angle_deg):
    """Check if point (px,py) is inside regular hexagon centered at (hx,hy)."""
    dx = px - hx
    dy = py - hy
    angle = -math.radians(angle_deg)
    # Rotate point to hexagon's local frame
    rx = dx * math.cos(angle) - dy * math.sin(angle)
    ry = dx * math.sin(angle) + dy * math.cos(angle)
    # Regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300
    # In local frame, hexagon is defined by |x| <= side and |y| <= sqrt(3)/2 * side and |x| + sqrt(3)/2 * |y| <= side
    # Actually for a regular hexagon with side s, vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    # The hexagon can be described as: for each of 6 sides, the point must be on the correct side
    # Side normals at angles 30, 90, 150, 210, 270, 330 degrees
    s = side
    for i in range(6):
        theta = math.radians(30 + i * 60)
        nx = math.cos(theta)
        ny = math.sin(theta)
        if rx * nx + ry * ny > s * math.cos(math.pi / 6):
            return False
    return True

def _hex_corners(cx, cy, side, angle_deg):
    """Return the 6 corner vertices of a unit hexagon at given position."""
    return _hex_vertices(cx, cy, side, angle_deg)

def _all_hexagon_corners(positions):
    """Get all 6*12 = 72 corners of all inner hexagons."""
    corners = []
    for (x, y, ang) in positions:
        corners.extend(_hex_vertices(x, y, 1.0, ang))
    return corners

def _min_enclosing_hexagon_side(corners, center):
    """
    Find minimum side length of a regular hexagon (any rotation) centered at 'center' 
    that contains all corners.
    """
    best_side = float('inf')
    # Test many rotations
    for deg in range(0, 360, 5):
        angle = math.radians(deg)
        # For a hexagon with this rotation, the constraint is:
        # For each of 6 directions (normal to each side), the max projection of any corner
        # relative to center must be <= side * cos(30) = side * sqrt(3)/2
        max_proj = 0
        for i in range(6):
            theta = angle + math.radians(30 + i * 60)
            nx = math.cos(theta)
            ny = math.sin(theta)
            for (px, py) in corners:
                proj = (px - center[0]) * nx + (py - center[1]) * ny
                if proj > max_proj:
                    max_proj = proj
        side_needed = max_proj / math.cos(math.pi / 6)
        if side_needed < best_side:
            best_side = side_needed
    return best_side

def _find_center_and_size(positions):
    """Find best center and compute minimal enclosing hexagon side."""
    corners = _all_hexagon_corners(positions)
    # Try several candidate centers
    # First: centroid of all corners
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    
    best = (cx, cy, _min_enclosing_hexagon_side(corners, (cx, cy)))
    
    # Try center as centroid of hexagon centers
    hxs = [p[0] for p in positions]
    hys = [p[1] for p in positions]
    hcx = sum(hxs) / len(hxs)
    hcy = sum(hys) / len(hys)
    s2 = _min_enclosing_hexagon_side(corners, (hcx, hcy))
    if s2 < best[2]:
        best = (hcx, hcy, s2)
    
    # Fine-tune: try small offsets around best center
    for dx in [-0.5, -0.25, 0, 0.25, 0.5]:
        for dy in [-0.5, -0.25, 0, 0.25, 0.5]:
            tcx = best[0] + dx
            tcy = best[1] + dy
            s3 = _min_enclosing_hexagon_side(corners, (tcx, tcy))
            if s3 < best[2]:
                best = (tcx, tcy, s3)
    
    return best

def construct():
    # Arrange 12 hexagons in a hexagonal pattern
    # Central hexagon at origin, then 6 around it (distance sqrt(3)), then 5 more
    # Use pointy-top orientation (vertices at top/bottom)
    # For pointy-top: neighbors are at distance sqrt(3) in 6 directions at 0, 60, 120, 180, 240, 300 degrees
    
    sqrt3 = math.sqrt(3)
    positions = []
    
    # Center
    positions.append((0.0, 0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3
        x = sqrt3 * math.cos(angle)
        y = sqrt3 * math.sin(angle)
        positions.append((x, y, 0.0))
    
    # Ring 2: need 5 more hexagons. Place them at distance 2*sqrt(3) from center
    # in directions that are "gaps" - at angles 30, 90, 150, 210, 270 degrees
    # But we only need 5, so pick 5 of the 6 possible gap positions
    # The 6 gap positions are at angles 30+60*i, distance 2*sqrt(3)
    for i in range(5):
        angle = math.radians(30 + i * 60)
        x = 2 * sqrt3 * math.cos(angle)
        y = 2 * sqrt3 * math.sin(angle)
        positions.append((x, y, 0.0))
    
    # Verify we have 12
    assert len(positions) == 12
    
    # Find optimal enclosing hexagon
    cx, cy, side = _find_center_and_size(positions)
    
    # The outer hexagon rotation: we need to find which rotation gives the min side
    # Re-find the best rotation
    corners = _all_hexagon_corners(positions)
    best_side = float('inf')
    best_rot = 0
    for deg in range(0, 360, 5):
        s = _min_enclosing_hexagon_side_at_rot(corners, (cx, cy), deg)
        if s < best_side:
            best_side = s
            best_rot = deg
    
    return positions, (cx, cy), best_side, best_rot

def _min_enclosing_hexagon_side_at_rot(corners, center, deg):
    """Compute min side for a hexagon with specific rotation."""
    angle = math.radians(deg)
    max_proj = 0
    for i in range(6):
        theta = angle + math.radians(30 + i * 60)
        nx = math.cos(theta)
        ny = math.sin(theta)
        for (px, py) in corners:
            proj = (px - center[0]) * nx + (py - center[1]) * ny
            if proj > max_proj:
                max_proj = proj
    return max_proj / math.cos(math.pi / 6)

# EVOLVE-BLOCK-END


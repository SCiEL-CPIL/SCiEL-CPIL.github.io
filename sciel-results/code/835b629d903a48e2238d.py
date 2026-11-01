"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial.distance import cdist
import math

N = 11

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side length, and rotation in degrees."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside or on the boundary of the hexagon."""
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle_rad = math.radians(-angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side s centered at origin, 
    # the distance from center to each side (apothem) is s * sqrt(3)/2
    # The hexagon can be defined by 6 half-planes
    apothem = side * math.sqrt(3) / 2
    for i in range(6):
        theta = i * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def _hexagon_distance_sq(h1_center, h1_angle, h2_center, h2_angle, side=1.0):
    """
    Approximate the minimum distance between two unit hexagons.
    Returns squared distance if they are separated, 0 if they overlap or touch.
    This is a conservative check: we use the distance between centers minus 
    the maximum extent of each hexagon.
    """
    cx1, cy1 = h1_center
    cx2, cy2 = h2_center
    dx = cx2 - cx1
    dy = cy2 - cy1
    dist_sq = dx * dx + dy * dy
    
    # The maximum distance from center to any point in a unit hexagon is the side length (1.0)
    # Actually, for a regular hexagon with side s, the circumradius is s, and inradius is s*sqrt(3)/2
    # The maximum distance between two points in the same hexagon is 2*s (diameter)
    # For two hexagons to be disjoint, the distance between centers must be at least 
    # the sum of their "radii" in the direction connecting them.
    # A safe lower bound for the required center distance is 2*side (if aligned along a vertex direction)
    # But this is very conservative. A better bound: the minimum distance between two unit hexagons
    # when optimally oriented is sqrt(3) (side-by-side along flat sides).
    # Let's use a conservative check: if center distance < sqrt(3) - epsilon, they likely overlap.
    # Actually, let's just check if the distance is less than 2*side which guarantees overlap,
    # and treat distances between sqrt(3) and 2 as uncertain (potential overlap).
    
    # For simplicity in the graph construction, let's use a threshold.
    # Two unit hexagons with arbitrary rotation: the minimum possible center distance for non-overlap
    # is sqrt(3) ≈ 1.732 (when flat sides face each other).
    # The maximum possible center distance where they could still overlap is 2.0 (vertex to vertex).
    # So if dist < sqrt(3), they definitely overlap.
    # If dist > 2.0, they definitely don't overlap.
    # In between, it depends on rotation.
    
    min_dist = math.sqrt(3)
    max_dist = 2.0
    
    if dist_sq < min_dist * min_dist:
        return 0.0  # Definitely overlapping
    elif dist_sq > max_dist * max_dist:
        return dist_sq  # Definitely separated
    else:
        # Uncertain region - be conservative and assume potential overlap
        # Return a small value to indicate they might conflict
        return 0.0  # Treat as conflicting to be safe

def construct():
    n = N  # 11 unit hexagons to place
    
    # Strategy: Use a hexagonal grid layout for the inner hexagons.
    # Unit hexagons can be packed in a honeycomb pattern.
    # In a honeycomb, adjacent hexagons have center distance sqrt(3).
    # 
    # For N hexagons, we want the smallest enclosing hexagon.
    # 
    # Let's try different packings:
    # 1. Linear chain: N hexagons in a row. Length ~ (N-1)*sqrt(3) + 2
    #    Enclosing hexagon side ~ (N-1)*sqrt(3)/2 + 1
    #
    # 2. 2D hexagonal cluster: arrange in a hexagonal shape.
    #    For a hexagonal cluster with "radius" r (number of hexagons from center to edge),
    #    total hexagons = 1 + 6*(1+2+...+r) = 1 + 3*r*(r+1)
    #    For N=11: 1+3r(r+1) >= 11 => 3r(r+1) >= 10 => r(r+1)>=3.33 => r>=2 (since 2*3=6>=3.33)
    #    With r=2: 1+3*2*3=19 hexagons. Too many.
    #    With r=1: 1+3*1*2=7 hexagons. Not enough.
    #    So we need a partial hexagonal cluster.
    
    # Let's place hexagons in a compact arrangement.
    # Positions for a honeycomb lattice:
    # Hex centers at:
    #   (0, 0)
    #   (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), etc.
    
    sqrt3 = math.sqrt(3)
    
    # Generate candidate positions on a honeycomb lattice
    # For a hexagonal lattice with side 1 hexagons:
    # Center positions: (i*sqrt3 + j*sqrt3/2, j*1.5) for integers i, j
    
    positions = []
    # Generate a grid of candidate positions
    for j in range(-3, 4):
        for i in range(-3, 4):
            x = i * sqrt3 + j * sqrt3 / 2
            y = j * 1.5
            positions.append((x, y))
    
    # Sort by distance from origin to prefer central positions
    positions.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Select N positions that are mutually non-overlapping
    # Two hexagons at positions p1 and p2 are non-overlapping if |p1-p2| >= sqrt(3)
    min_center_dist = sqrt3  # minimum distance between centers for non-overlap
    
    selected = []
    for pos in positions:
        # Check if this position conflicts with already selected ones
        conflict = False
        for sel_pos in selected:
            dx = pos[0] - sel_pos[0]
            dy = pos[1] - sel_pos[1]
            if dx*dx + dy*dy < min_center_dist * min_center_dist - 1e-9:
                conflict = True
                break
        if not conflict:
            selected.append(pos)
            if len(selected) == n:
                break
    
    # If we didn't get enough positions, expand the search
    while len(selected) < n:
        for j in range(-6, 7):
            for i in range(-6, 7):
                x = i * sqrt3 + j * sqrt3 / 2
                y = j * 1.5
                pos = (x, y)
                if pos in selected:
                    continue
                conflict = False
                for sel_pos in selected:
                    dx = pos[0] - sel_pos[0]
                    dy = pos[1] - sel_pos[1]
                    if dx*dx + dy*dy < min_center_dist * min_center_dist - 1e-9:
                        conflict = True
                        break
                if not conflict:
                    selected.append(pos)
                    if len(selected) == n:
                        break
            if len(selected) == n:
                break
    
    # Now find the minimum enclosing regular hexagon for these positions
    # The inner hexagons have side 1, so their centers must be at least 
    # (1 * sqrt(3)/2) = apothem away from the boundary of the outer hexagon.
    # So the outer hexagon must contain all inner hexagons, meaning the outer hexagon's
    # inradius must be at least the maximum distance from center to any point of any inner hexagon.
    
    # For a regular hexagon with side S centered at (cx, cy), a point (px, py) is inside
    # if for all 6 directions, the projection is <= S*sqrt(3)/2.
    
    def hexagon_side_for_points(points, outer_angle_deg=0):
        """Find minimum side length of a regular hexagon (with given rotation) enclosing all points."""
        # The outer hexagon must have inradius >= max over all points of (distance in 6 directions)
        # For a hexagon with inradius r, the constraint is: for each of 6 directions theta_k,
        # |dot(point - center, dir_k)| <= r
        # where dir_k are unit vectors in the 6 normal directions.
        
        # First, find the center that minimizes the required inradius.
        # The optimal center is the centroid or somewhere near it.
        # Let's try the centroid first.
        
        pts = np.array(points)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        
        angle_rad = math.radians(outer_angle_deg)
        max_inradius = 0
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            for px, py in points:
                proj = abs((px - cx) * nx + (py - cy) * ny)
                if proj > max_inradius:
                    max_inradius = proj
        
        inradius = max_inradius + 1.0 * math.sqrt(3) / 2  # Add apothem of inner hexagon
        side = inradius * 2 / math.sqrt(3)
        
        return side, (cx, cy)
    
    # Try a few outer hexagon rotations to find the best
    best_side = float('inf')
    best_center = (0, 0)
    best_angle = 0
    
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        side, center = hexagon_side_for_points(selected, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    # The inner hexagons can be rotated arbitrarily. Let's set them all to 0 degrees.
    inner = [[x, y, 0.0] for x, y in selected]
    
    return inner, best_center, best_side, best_angle

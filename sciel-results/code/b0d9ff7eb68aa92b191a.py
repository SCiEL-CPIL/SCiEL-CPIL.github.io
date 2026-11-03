"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    # Rotate point by -angle to align hexagon with standard orientation
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # Standard hexagon (flat-top) with side s has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon can be described by 3 pairs of parallel lines
    # For a hexagon with circumradius = side (since it's regular), the constraints are:
    # |lx| <= side
    # |ly| <= side * sqrt(3)/2  -- no, this is for the bounding box
    
    # Better: use the support function approach
    # A regular hexagon centered at origin with circumradius R=side has the property that
    # a point (lx, ly) is inside if and only if for all 6 directions (at angles 0,60,...,300),
    # the projection is <= side
    # Actually, for a regular hexagon with side length s, the apothem is s*sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes
    
    # Let me use a different approach: the hexagon with vertices at angles 0,60,...,300 and radius s
    # A point is inside if its distance to each edge is non-negative
    
    # The 6 edges have outward normals at angles 30, 90, 150, 210, 270, 330 (for flat-top)
    # For pointy-top (vertices at 0,60,...), normals are at 0, 60, 120, 180, 240, 300
    
    # Let's just check against all 6 vertices using cross products or use the fact that
    # a regular hexagon with circumradius R has the property that max(|lx|, |ly|, |(lx+ly)/sqrt(3)|) <= R
    
    # Actually, simplest: use the half-plane representation
    # For a regular hexagon centered at origin with circumradius R, the 6 supporting lines are at distance R*sqrt(3)/2 from center
    # with normals at angles 0, 60, 120, 180, 240, 300 degrees
    
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        normal_angle = np.radians(60 * i)  # normals at 0, 60, 120, 180, 240, 300
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def compute_min_outer_side(inner_positions, outer_center, outer_angle_deg):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    cx, cy = outer_center
    # For each inner hexagon, find the maximum "extent" in each of the 6 normal directions
    # of the outer hexagon
    max_extent = 0.0
    for (ix, iy, iangle) in inner_positions:
        # Get vertices of inner hexagon
        verts = hexagon_vertices(ix, iy, 1.0, iangle)
        for vx, vy in verts:
            # Transform to outer hexagon's local frame
            dx = vx - cx
            dy = vy - cy
            angle_rad = np.radians(outer_angle_deg)
            cos_a = np.cos(-angle_rad)
            sin_a = np.sin(-angle_rad)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            # Check extent in each of the 6 normal directions
            for i in range(6):
                normal_angle = np.radians(60 * i)
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                extent = lx * nx + ly * ny
                if extent > max_extent:
                    max_extent = extent
    
    # The apothem of the outer hexagon must be at least max_extent
    # apothem = side * sqrt(3) / 2, so side = 2 * max_extent / sqrt(3)
    side = 2.0 * max_extent / np.sqrt(3)
    return side

def check_no_overlap(inner_positions):
    """Check that no two inner hexagons overlap."""
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ai = inner_positions[i]
            xj, yj, aj = inner_positions[j]
            # Check if hexagons overlap using SAT (Separating Axis Theorem)
            vi = hexagon_vertices(xi, yi, 1.0, ai)
            vj = hexagon_vertices(xj, yj, 1.0, aj)
            if polygons_overlap_sat(vi, vj):
                return False
    return True

def polygons_overlap_sat(poly1, poly2):
    """Check if two convex polygons overlap using SAT."""
    polys = [poly1, poly2]
    for poly in polys:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            # Edge from poly[i] to poly[j]
            edge = poly[j] - poly[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            
            # Project both polygons onto normal
            proj1 = np.dot(poly1, normal)
            proj2 = np.dot(poly2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 or max2 < min1:
                return False  # Found separating axis, no overlap
    return True  # No separating axis found, polygons overlap

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try several candidate arrangements
    best_side = float('inf')
    best_result = None
    
    # Candidate 1: Honeycomb-like arrangement
    # Central hex + 6 around + 5 more
    # For 12 hexagons, try a 2-3-3-3-1 or similar pattern
    
    # Let me try a simple grid-like arrangement first
    # Hexagons in a honeycomb: centers at distance 2*sqrt(3) apart horizontally, sqrt(3) vertically with offset
    
    # Actually, for non-overlapping unit hexagons (side=1), the minimum center-to-center distance
    # depends on relative rotation. For same orientation, it's 2*sqrt(3) ≈ 3.464 (edge-to-edge)
    # Wait, no. Two unit hexagons with same orientation: the distance between centers must be at least
    # 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732? No, that's for touching edges.
    
    # Let me reconsider. A unit regular hexagon has:
    # - Side length: 1
    # - Circumradius (center to vertex): 1
    # - Inradius (center to edge midpoint): sqrt(3)/2 ≈ 0.866
    # 
    # Two identical hexagons with same orientation touch when center distance = 2 * inradius = sqrt(3)
    # But they can also touch vertex-to-vertex at distance 2.
    # The minimum non-overlapping distance is actually sqrt(3) for edge-to-edge contact.
    
    # Let me try a compact arrangement. For 12 hexagons, a good pattern might be:
    # Row 1: 2 hexagons
    # Row 2: 3 hexagons (offset)
    # Row 3: 3 hexagons
    # Row 4: 3 hexagons (offset)
    # Row 5: 1 hexagon
    # Total: 2+3+3+3+1 = 12
    
    # Or more symmetrically:
    # A hexagonal arrangement with 12 hexagons
    
    # Let me try: center at origin, with hexagons placed in rings
    # Ring 0: 1 hexagon at center
    # Ring 1: 6 hexagons at distance sqrt(3) from center (touching)
    # Ring 2: need 5 more to make 12
    
    # For ring 1, hexagons at angles 0, 60, 120, 180, 240, 300, distance sqrt(3)
    # For ring 2, place 5 hexagons at distance 2*sqrt(3) at certain angles
    
    candidates = []
    
    # Candidate A: 1 + 6 + 5 arrangement
    pos_a = [(0.0, 0.0, 0.0)]
    d1 = np.sqrt(3)
    for i in range(6):
        angle = np.radians(60 * i)
        pos_a.append((d1 * np.cos(angle), d1 * np.sin(angle), 0.0))
    d2 = 2 * np.sqrt(3)
    # Place 5 more at specific angles to minimize bounding hexagon
    for i in range(5):
        angle = np.radians(60 * i + 30)  # Offset by 30 degrees
        pos_a.append((d2 * np.cos(angle), d2 * np.sin(angle), 0.0))
    candidates.append(pos_a)
    
    # Candidate B: More compact, try different angles for outer ring
    pos_b = [(0.0, 0.0, 0.0)]
    for i in range(6):
        angle = np.radians(60 * i)
        pos_b.append((d1 * np.cos(angle), d1 * np.sin(angle), 0.0))
    for i in range(5):
        angle = np.radians(60 * i)  # No offset
        pos_b.append((d2 * np.cos(angle), d2 * np.sin(angle), 0.0))
    candidates.append(pos_b)
    
    # Candidate C: Try a 2-3-3-3-1 row arrangement
    pos_c = []
    # Rows at y = 0, sqrt(3)/2, sqrt(3), 3*sqrt(3)/2, 2*sqrt(3)
    # Row 0 (y=0): 2 hexagons at x = -1, 1
    pos_c.append((-1.0, 0.0, 0.0))
    pos_c.append((1.0, 0.0, 0.0))
    # Row 1 (y=sqrt(3)/2): 3 hexagons at x = -1.5, 0, 1.5 -- but need to check spacing
    # Actually for honeycomb, horizontal spacing is sqrt(3) for same row, and rows are sqrt(3)/2 apart
    # with offset of sqrt(3)/2
    h_spacing = np.sqrt(3)  # horizontal distance between adjacent hex centers in same row
    v_spacing = np.sqrt(3) / 2  # vertical distance between rows
    
    pos_c = []
    # Row 0: 2 hexagons
    pos_c.append((-h_spacing/2, 0.0, 0.0))
    pos_c.append((h_spacing/2, 0.0, 0.0))
    # Row 1: 3 hexagons, offset by h_spacing/2
    pos_c.append((-h_spacing, v_spacing, 0.0))
    pos_c.append((0.0, v_spacing, 0.0))
    pos_c.append((h_spacing, v_spacing, 0.0))
    # Row 2: 3 hexagons, same x as row 0
    pos_c.append((-h_spacing/2, 2*v_spacing, 0.0))
    pos_c.append((h_spacing/2, 2*v_spacing, 0.0))
    pos_c.append((0.0, 2*v_spacing, 0.0))
    # Row 3: 3 hexagons, offset
    pos_c.append((-h_spacing, 3*v_spacing, 0.0))
    pos_c.append((0.0, 3*v_spacing, 0.0))
    pos_c.append((h_spacing, 3*v_spacing, 0.0))
    # Row 4: 1 hexagon
    pos_c.append((0.0, 4*v_spacing, 0.0))
    candidates.append(pos_c)
    
    # Evaluate each candidate
    for idx, pos in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Check for overlaps
        if not check_no_overlap(pos):
            continue
        
        # Compute minimum outer hexagon side
        # Try outer hexagon centered at centroid of inner hexagons
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        # Try several outer rotations
        for outer_angle in [0.0, 30.0, 60.0]:
            side = compute_min_outer_side(pos, (cx, cy), outer_angle)
            if side < best_side:
                best_side = side
                best_result = (pos, (cx, cy), side, outer_angle)
    
    # If no valid candidate found, return a default
    if best_result is None:
        # Fallback: spread out hexagons in a circle
        pos = []
        for i in range(N):
            angle = 2 * np.pi * i / N
            pos.append((5.0 * np.cos(angle), 5.0 * np.sin(angle), 0.0))
        return pos, (0.0, 0.0), 15.0, 0.0
    
    pos, center, side, angle = best_result
    return pos, center, side, angle

# EVOLVE-BLOCK-START
# (The construct function above is the main logic)
# EVOLVE-BLOCK-END

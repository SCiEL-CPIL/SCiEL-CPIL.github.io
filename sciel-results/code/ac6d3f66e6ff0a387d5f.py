import math
import time
import numpy as np
from itertools import combinations

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 150.0
START_TIME = time.time()

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        angle = math.radians(angle_deg + 60 * i)
        verts.append((cx + side * math.cos(angle), cy + side * math.sin(angle)))
    return verts

def _hexagon_contains_point(hex_cx, hex_cy, hex_side, hex_angle, px, py):
    """Check if point (px,py) is inside hexagon centered at (hex_cx,hex_cy)."""
    # Transform point to hexagon's local frame
    dx = px - hex_cx
    dy = py - hex_cy
    angle = math.radians(hex_angle)
    ca, sa = math.cos(angle), math.sin(angle)
    lx = dx * ca + dy * sa
    ly = -dx * sa + dy * ca
    # For a unit hexagon, check distance in each of 6 directions
    for i in range(6):
        a = math.radians(60 * i)
        nx, ny = math.cos(a), math.sin(a)
        # Distance from center to side i
        d = lx * nx + ly * ny
        if d > hex_side * SQRT3 / 2 + 1e-9:
            return False
    return True

def _hexagons_overlap(h1, h2, tol=1e-9):
    """Check if two unit hexagons overlap (including touching)."""
    x1, y1, a1 = h1
    x2, y2, a2 = h2
    # Quick bounding box check
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if dx > 2.0 + tol or dy > 2.0 + tol:
        return False
    # Check if any vertex of one is inside the other
    v1 = _hex_vertices(x1, y1, 1.0, a1)
    v2 = _hex_vertices(x2, y2, 1.0, a2)
    for vx, vy in v1:
        if _hexagon_contains_point(x2, y2, 1.0, a2, vx, vy):
            return True
    for vx, vy in v2:
        if _hexagon_contains_point(x1, y1, 1.0, a1, vx, vy):
            return True
    # Check edge intersections
    def seg_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
           ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
            return True
        return False
    for i in range(6):
        for j in range(6):
            if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def _bounding_hexagon_side(hex_positions, outer_angle=0):
    """Find minimum side length of a regular hexagon (with given rotation) containing all hexagons."""
    if not hex_positions:
        return 0.0, (0.0, 0.0)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for (x, y, ang) in hex_positions:
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.extend(verts)
    
    all_verts = np.array(all_verts)
    
    # The bounding hexagon must contain all these points
    # For a hexagon with center (cx, cy), side s, rotation theta:
    # A point (px, py) is inside if for all 6 directions:
    #   (px-cx)*cos(a) + (py-cy)*sin(a) <= s * SQRT3 / 2
    # where a = theta + 60*i for i in 0..5
    
    theta = math.radians(outer_angle)
    
    # We need to find cx, cy, s such that all constraints are satisfied
    # For each direction i, the constraint is:
    #   max over all points of [(px-cx)*cos(a_i) + (py-cy)*sin(a_i)] <= s * SQRT3 / 2
    #   min over all points of [(px-cx)*cos(a_i) + (py-cy)*sin(a_i)] >= -s * SQRT3 / 2
    # So s * SQRT3 / 2 >= max(|max_proj_i|, |min_proj_i|) for each i
    # And s * SQRT3 / 2 >= max over all i of max(|max_proj_i|, |min_proj_i|)
    
    # This is a convex optimization problem. Let's use a simple approach:
    # The optimal center is the centroid of the projected points for each pair of opposite faces.
    
    # Actually, let's just search over a grid of centers.
    xs = all_verts[:, 0]
    ys = all_verts[:, 1]
    cx_min, cx_max = xs.min() - 2, xs.max() + 2
    cy_min, cy_max = ys.min() - 2, ys.max() + 2
    
    # Grid search for center
    for cx in np.linspace(cx_min, cx_max, 50):
        for cy in np.linspace(cy_min, cy_max, 50):
            s_needed = 0.0
            for i in range(6):
                a = theta + math.radians(60 * i)
                nx, ny = math.cos(a), math.sin(a)
                proj = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
                max_proj = proj.max()
                min_proj = proj.min()
                s_i = max(abs(max_proj), abs(min_proj)) * 2 / SQRT3
                s_needed = max(s_needed, s_i)
            if s_needed < best_side:
                best_side = s_needed
                best_center = (cx, cy)
    
    return best_side, best_center

def _try_configuration(positions, angles, outer_angles=[0, 30, 60, 90, 120, 150]):
    """Try a configuration and return (side, center, outer_angle) or None."""
    best = None
    for oa in outer_angles:
        side, center = _bounding_hexagon_side([(x, y, a) for (x, y), a in zip(positions, angles)], oa)
        if best is None or side < best[0]:
            best = (side, center, oa)
    return best

def construct():
    global START_TIME, DEADLINE
    start = time.time()
    deadline = start + DEADLINE - 10  # safety margin
    
    best_result = None
    best_side = float('inf')
    
    # Honeycomb lattice positions for 11 hexagons
    # Hexagonal numbers: 1, 7, 19, ... 
    # 11 is not a hexagonal number, so use a modified arrangement
    # Arrangement: rows of 3,3,3,2
    # Flat-top hexagons, angle = 0
    
    # Position 1: central
    # Positions 2-7: surrounding
    # Positions 8-11: additional
    
    # For flat-top hexagons (angle=0):
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    # Neighbor offsets (flat-top):
    #   (0, ±sqrt(3)), (±3/2, ±sqrt(3)/2)
    
    def honeycomb_positions(n):
        """Generate honeycomb positions for n unit hexagons (flat-top)."""
        positions = [(0.0, 0.0)]
        # Ring 1: 6 hexagons
        offsets = [
            (0, SQRT3), (SQRT3/2, SQRT3/2), (SQRT3/2, -SQRT3/2),
            (0, -SQRT3), (-SQRT3/2, -SQRT3/2), (-SQRT3/2, SQRT3/2)
        ]
        for ox, oy in offsets:
            positions.append((ox, oy))
        # Ring 2: 12 hexagons, we need 4 more
        ring2 = []
        for i in range(12):
            angle = math.radians(30 + 30 * i)  # Starting at 30 degrees
            # Actually, ring 2 positions are at distance 2*sqrt(3) from center
            # in directions 0, 30, 60, ... 330
            r = 2 * SQRT3
            ring2.append((r * math.cos(angle), r * math.sin(angle)))
        for pos in ring2[:4]:
            positions.append(pos)
        return positions[:n]
    
    positions_11 = honeycomb_positions(11)
    
    # Try different inner angles and outer angles
    inner_angle_options = [0.0, 30.0]
    
    for ia in inner_angle_options:
        if time.time() > deadline:
            break
        angles = [ia] * N
        result = _try_configuration(positions_11, angles)
        if result and result[0] < best_side:
            best_side = result[0]
            best_result = ([(x, y, a) for (x, y), a in zip(positions_11, angles)],
                           result[1], result[0], result[2])
    
    # Try pointy-top arrangement (rotate positions by 90 degrees)
    positions_pt = [(y, x) for (x, y) in positions_11]
    for ia in [90.0]:
        if time.time() > deadline:
            break
        angles = [ia] * N
        result = _try_configuration(positions_pt, angles)
        if result and result[0] < best_side:
            best_side = result[0]
            best_result = ([(x, y, a) for (x, y), a in zip(positions_pt, angles)],
                           result[1], result[0], result[2])
    
    # Try a more compact arrangement: 4-3-4 or other patterns
    # Let me try a different arrangement: 
    # Row 1 (top): 2 hexagons
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # Row 4 (bottom): 3 hexagons
    # Total: 11
    
    # For flat-top hexagons, rows are offset
    # Row spacing: sqrt(3)
    # Horizontal spacing: 1.5 (for adjacent in same row), 3.0 (for non-adjacent)
    
    alt_positions = []
    # Row 0 (top): 2 hexagons at x = -0.75, 0.75, y = 1.5*SQRT3
    alt_positions.append((-0.75, 1.5 * SQRT3))
    alt_positions.append((0.75, 1.5 * SQRT3))
    # Row 1: 3 hexagons at x = -1.5, 0, 1.5, y = 0.75*SQRT3
    alt_positions.append((-1.5, 0.75 * SQRT3))
    alt_positions.append((0.0, 0.75 * SQRT3))
    alt_positions.append((1.5, 0.75 * SQRT3))
    # Row 2: 3 hexagons at x = -1.5, 0, 1.5, y = -0.75*SQRT3
    alt_positions.append((-1.5, -0.75 * SQRT3))
    alt_positions.append((0.0, -0.75 * SQRT3))
    alt_positions.append((1.5, -0.75 * SQRT3))
    # Row 3 (bottom): 3 hexagons at x = -1.5, 0, 1.5, y = -1.5*SQRT3
    alt_positions.append((-1.5, -1.5 * SQRT3))
    alt_positions.append((0.0, -1.5 * SQRT3))
    alt_positions.append((1.5, -1.5 * SQRT3))
    
    for ia in inner_angle_options:
        if time.time() > deadline:
            break
        angles = [ia] * N
        result = _try_configuration(alt_positions, angles)
        if result and result[0] < best_side:
            best_side = result[0]
            best_result = ([(x, y, a) for (x, y), a in zip(alt_positions, angles)],
                           result[1], result[0], result[2])
    
    # Try yet another arrangement: 5-3-3
    arr3 = []
    # Row 0: 5 hexagons
    for i in range(5):
        arr3.append((float(i - 2) * 1.5, 0.0))
    # Row 1: 3 hexagons
    for i in range(3):
        arr3.append((float(i - 1) * 1.5 + 0.75, SQRT3 / 2))
    # Row 2: 3 hexagons
    for i in range(3):
        arr3.append((float(i - 1) * 1.5 + 0.75, -SQRT3 / 2))
    
    for ia in inner_angle_options:
        if time.time() > deadline:
            break
        angles = [ia] * N
        result = _try_configuration(arr3, angles)
        if result and result[0] < best_side:
            best_side = result[0]
            best_result = ([(x, y, a) for (x, y), a in zip(arr3, angles)],
                           result[1], result[0], result[2])
    
    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, center, side, outer_angle = best_result
    return [[x, y, a] for x, y, a in inner], center, side, outer_angle


"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import math
import time

N = 11

def _hex_to_cartesian(q, r, scale=1.0):
    """Convert axial coordinates (q, r) to Cartesian for pointy-top hexagons."""
    x = scale * math.sqrt(3) * (q + r / 2.0)
    y = scale * 1.5 * r
    return x, y

def _hex_corners(cx, cy, angle_deg=0.0, side=1.0):
    """Get the 6 corners of a regular hexagon centered at (cx,cy) with given rotation and side length."""
    angle = math.radians(angle_deg)
    corners = []
    for i in range(6):
        a = angle + i * math.pi / 3.0
        corners.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return corners

def _min_enclosing_hexagon_side(points):
    """
    Find the minimum side length of a regular hexagon (any rotation, any center) 
    that encloses all given points. Returns (min_side, center_x, center_y, angle_deg).
    
    For a regular hexagon with inradius R, the constraint for a point to be inside is:
    For each of 6 directions (separated by 60 deg), the projection onto that direction 
    must be within [-R, R] where R = side * sqrt(3)/2.
    
    So we need: max over all points and all 6 directions of |projection| <= R
    R_min = max over directions d of (max_p p·d - min_p p·d) / 2
    Then side = R_min * 2 / sqrt(3)
    """
    if len(points) == 0:
        return 0.0, 0.0, 0.0, 0.0
    if len(points) == 1:
        px, py = points[0]
        return 2.0 / math.sqrt(3), px, py, 0.0
    
    points = np.array(points)
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    centered = points - np.array([cx, cy])
    
    min_side = float('inf')
    best_angle = 0.0
    best_cx, best_cy = cx, cy
    
    # Try various rotations
    for angle_deg in np.linspace(0, 60, 25):  # 60 deg symmetry
        angle = math.radians(angle_deg)
        # 6 directions
        max_range = 0.0
        for i in range(3):  # Only need 3 opposite pairs
            d = angle + i * math.pi / 3.0
            projections = centered @ np.array([math.cos(d), math.sin(d)])
            rng = projections.max() - projections.min()
            if rng > max_range:
                max_range = rng
        R = max_range / 2.0
        side = R * 2.0 / math.sqrt(3)
        if side < min_side:
            min_side = side
            best_angle = angle_deg
            best_cx, best_cy = cx, cy
    
    return min_side, best_cx, best_cy, best_angle

def _generate_hexagonal_positions(r_max):
    """Generate all hex positions in a hexagonal grid of radius r_max."""
    positions = []
    for q in range(-r_max, r_max + 1):
        for r in range(-r_max, r_max + 1):
            if abs(q + r) <= r_max:
                positions.append((q, r))
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_result = None
    best_side = float('inf')
    
    # Strategy: try hexagonal arrangements of increasing radius
    # For each radius, select N positions that minimize the enclosing hexagon
    
    # First, try full hexagonal clusters
    for r in range(0, 10):
        if time.time() > deadline:
            break
        positions = _generate_hexagonal_positions(r)
        n_pos = len(positions)
        if n_pos < N:
            continue
        if n_pos > 100:  # Don't go too large
            break
        
        # Convert to Cartesian
        cart_positions = [_hex_to_cartesian(q, rr) for q, rr in positions]
        
        if n_pos == N:
            # Use all
            all_corners = []
            for cx, cy in cart_positions:
                corners = _hex_corners(cx, cy, 0.0, 1.0)
                all_corners.extend(corners)
            side, ocx, ocy, oangle = _min_enclosing_hexagon_side(all_corners)
            if side < best_side:
                best_side = side
                inner = [[cx, cy, 0.0] for cx, cy in cart_positions]
                best_result = (inner, (ocx, ocy), side, oangle)
        else:
            # Need to select N positions from n_pos
            # Try center-biased selection: sort by distance from center, take closest N
            dists = [math.sqrt(cx**2 + cy**2) for cx, cy in cart_positions]
            sorted_idx = np.argsort(dists)
            selected = [cart_positions[i] for i in sorted_idx[:N]]
            
            all_corners = []
            for cx, cy in selected:
                corners = _hex_corners(cx, cy, 0.0, 1.0)
                all_corners.extend(corners)
            side, ocx, ocy, oangle = _min_enclosing_hexagon_side(all_corners)
            if side < best_side:
                best_side = side
                inner = [[cx, cy, 0.0] for cx, cy in selected]
                best_result = (inner, (ocx, ocy), side, oangle)
    
    # Also try linear and other arrangements
    if time.time() < deadline - 5:
        # Try a few more configurations
        configs = []
        
        # Horizontal line
        for spacing in [math.sqrt(3), 2.0, 1.9, 1.8]:
            xs = [spacing * (i - (N-1)/2.0) for i in range(N)]
            cart = [(x, 0.0) for x in xs]
            all_corners = []
            for cx, cy in cart:
                corners = _hex_corners(cx, cy, 0.0, 1.0)
                all_corners.extend(corners)
            side, ocx, ocy, oangle = _min_enclosing_hexagon_side(all_corners)
            if side < best_side:
                best_side = side
                inner = [[cx, cy, 0.0] for cx, cy in cart]
                best_result = (inner, (ocx, ocy), side, oangle)
        
        # Staggered rows
        for row_offset in [0.5, 0.8, 1.0]:
            rows = []
            per_row = [N // 2, N - N // 2]
            y_positions = [-row_offset * 1.5, row_offset * 1.5]
            for ri, (y, count) in enumerate(zip(y_positions, per_row)):
                for i in range(count):
                    x = (i - (count-1)/2.0) * math.sqrt(3)
                    if ri % 2 == 1:
                        x += math.sqrt(3) / 2.0
                    rows.append((x, y))
            all_corners = []
            for cx, cy in rows:
                corners = _hex_corners(cx, cy, 0.0, 1.0)
                all_corners.extend(corners)
            side, ocx, ocy, oangle = _min_enclosing_hexagon_side(all_corners)
            if side < best_side:
                best_side = side
                inner = [[cx, cy, 0.0] for cx, cy in rows]
                best_result = (inner, (ocx, ocy), side, oangle)
    
    if best_result is None:
        # Fallback: just return something valid
        inner = [[math.sqrt(3) * i, 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 50.0, 0.0
    
    return best_result

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

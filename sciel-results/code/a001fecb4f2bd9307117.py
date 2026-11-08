import numpy as np
import time

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    # Rotate point by -angle
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local frame, hexagon has vertices at angles 0,60,120,180,240,300
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with "radius" (center to vertex) = side:
    # |lx| <= side (for the two vertical-ish sides)
    # Actually, let's use the fact that a regular hexagon with circumradius R=side
    # has apothem = side * sqrt(3)/2
    # The 6 sides are at distances apothem from center, at angles 30, 90, 150, 210, 270, 330
    
    apothem = side * np.sqrt(3) / 2
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        # Distance from center to this side along normal direction
        dist = lx * np.cos(theta) + ly * np.sin(theta)
        if dist > apothem + 1e-9:
            return False
    return True

def check_all_inside(inner, outer_center, outer_side, outer_angle):
    cx, cy = outer_center
    for x, y, ang in inner:
        verts = hexagon_vertices(x, y, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, cx, cy, outer_side, outer_angle):
                return False
    return True

def min_outer_side(inner, outer_angle):
    """Find minimum outer side for given inner arrangement and outer rotation."""
    # Try different centers
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    
    # Bounding box center as initial guess
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    
    # For a given center and outer rotation, find minimum side
    # such that all inner vertices are inside
    def required_side(cx, cy, outer_angle):
        max_dist = 0
        for x, y, ang in inner:
            verts = hexagon_vertices(x, y, 1.0, ang)
            for vx, vy in verts:
                # Transform to outer hexagon local frame
                dx = vx - cx
                dy = vy - cy
                angle_rad = np.radians(outer_angle)
                cos_a = np.cos(-angle_rad)
                sin_a = np.sin(-angle_rad)
                lx = dx * cos_a - dy * sin_a
                ly = dx * sin_a + dy * cos_a
                
                # For each of 6 sides, compute required side length
                for k in range(6):
                    theta = np.radians(30 + 60 * k)
                    dist = lx * np.cos(theta) + ly * np.sin(theta)
                    if dist > 0:
                        # Need apothem >= dist, so side >= dist / (sqrt(3)/2) = 2*dist/sqrt(3)
                        required = 2 * dist / np.sqrt(3)
                        if required > max_dist:
                            max_dist = required
        return max_dist
    
    # Optimize center
    best_side = float('inf')
    best_cx, best_cy = cx0, cy0
    
    # Grid search for center
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    for i in range(11):
        for j in range(11):
            cx_try = min(xs) + (i / 10) * x_range
            cy_try = min(ys) + (j / 10) * y_range
            s = required_side(cx_try, cy_try, outer_angle)
            if s < best_side:
                best_side = s
                best_cx, best_cy = cx_try, cy_try
    
    return best_side, (best_cx, best_cy)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    
    # Honeycomb arrangement: 4 cols x 3 rows
    inner = []
    cols = 4
    rows = 3
    for c in range(cols):
        for r in range(rows):
            x = c * 1.5
            y = r * sqrt3 + (sqrt3 / 2) * (c % 2)
            inner.append([x, y, 0.0])
    
    # Try different outer rotations
    best_result = None
    best_side = float('inf')
    
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline:
            break
        side, center = min_outer_side(inner, outer_angle)
        if side < best_side:
            best_side = side
            best_result = (inner, center, side + 1e-6, outer_angle)
    
    if best_result is None:
        # Fallback
        inner_fb = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner_fb, (0.0, 0.0), 40.0, 0.0
    
    return best_result

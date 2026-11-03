"""Improved program for unit hexagons in a hexagon packing."""
import numpy as np
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside hexagon centered at (cx, cy)."""
    # Transform to local coordinates
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate back
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # In local coords, hexagon has vertices at (side*cos(i*pi/3), side*sin(i*pi/3))
    # The hexagon is defined by |lx| <= side, |ly| <= side*sqrt(3)/2, and |lx*sqrt(3) + ly| <= side*sqrt(3)
    # Actually for a regular hexagon with flat top (angle=0), vertices are at angles 0, 60, 120, 180, 240, 300
    # Let's use the standard: for angle=0, vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    # The constraints are:
    # |lx| <= s
    # |ly| <= s*sqrt(3)/2
    # |sqrt(3)*lx + ly| <= s*sqrt(3)  -- wait, let me think again
    
    # For a regular hexagon centered at origin with one vertex at (s, 0):
    # The six sides are at distances s*cos(30°) = s*sqrt(3)/2 from center for the "flat" sides
    # and the vertices are at distance s.
    # The half-width in x is s, half-height in y is s*sqrt(3)/2
    # But it's not a rectangle. The constraints are:
    # |lx| <= s
    # |ly| <= s*sqrt(3)/2
    # |lx*sqrt(3) + ly| <= s*sqrt(3)  -> this gives the slanted sides
    
    # Let me verify: at vertex (s, 0): lx=s, ly=0. |lx|=s OK. |ly|=0 OK. |s*sqrt(3)+0|=s*sqrt(3) OK (boundary).
    # At vertex (s/2, s*sqrt(3)/2): lx=s/2, ly=s*sqrt(3)/2. |lx|=s/2<s OK. |ly|=s*sqrt(3)/2 OK. |s/2*sqrt(3)+s*sqrt(3)/2|=|s*sqrt(3)|=s*sqrt(3) OK.
    # At point (0, s*sqrt(3)/2): lx=0, ly=s*sqrt(3)/2. |ly|=s*sqrt(3)/2 OK. |0+s*sqrt(3)/2|=s*sqrt(3)/2 < s*sqrt(3) OK.
    
    eps = 1e-9
    if abs(lx) > side + eps:
        return False
    if abs(ly) > side * np.sqrt(3) / 2 + eps:
        return False
    if abs(np.sqrt(3) * lx + ly) > side * np.sqrt(3) + eps:
        return False
    if abs(np.sqrt(3) * lx - ly) > side * np.sqrt(3) + eps:
        return False
    return True

def hexagons_overlap(h1, h2):
    """Check if two unit hexagons overlap. h1, h2 are (cx, cy, angle_deg)."""
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    # Check all vertices of each hexagon against the other
    verts1 = hex_vertices(cx1, cy1, 1.0, a1)
    verts2 = hex_vertices(cx2, cy2, 1.0, a2)
    for v in verts1:
        if point_in_hexagon(v[0], v[1], cx2, cy2, 1.0, a2):
            return True
    for v in verts2:
        if point_in_hexagon(v[0], v[1], cx1, cy1, 1.0, a1):
            return True
    # Also check if centers are too close (simple distance check)
    dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    if dist < 1.0:  # minimum distance between centers of non-overlapping unit hexagons
        # This is a necessary but not sufficient condition, but if dist < 1 they definitely overlap
        return True
    return False

def all_inside_outer(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    for cx, cy, angle in inner:
        # Check all vertices of inner hexagon
        verts = hex_vertices(cx, cy, 1.0, angle)
        for v in verts:
            if not point_in_hexagon(v[0], v[1], ox, oy, outer_side, outer_angle):
                return False
    return True

def no_overlaps(inner):
    """Check if no two inner hexagons overlap."""
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            if hexagons_overlap(inner[i], inner[j]):
                return False
    return True

def construct():
    """Construct N unit hexagons inside a minimal outer hexagon."""
    # Strategy: Use a hexagonal lattice arrangement
    # For N hexagons, arrange them in a hexagonal pattern
    # The optimal arrangement for small N is a hexagonal cluster
    
    # For N=11, a good arrangement is a central hexagon surrounded by 6, then 4 more
    # Actually, let's try a linear arrangement first and optimize
    
    # Let's try different arrangements and pick the best
    
    best_score = float('inf')
    best_config = None
    
    # Arrangement 1: Linear row
    # Place hexagons in a row, side by side
    # Unit hexagons side by side: center distance = 2 * side * sin(60°) = sqrt(3) for flat-top
    # Wait, for regular hexagons with side 1, the distance between centers of adjacent hexagons
    # in a hexagonal lattice is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    
    # Let's try a hexagonal packing pattern
    # Row-based hexagonal packing
    
    # Try multiple initial configurations and optimize
    
    configs = []
    
    # Config 1: Single row
    x_positions = []
    for i in range(N):
        x_positions.append(i * np.sqrt(3))
    config1 = [[x - (N-1)*np.sqrt(3)/2, 0.0, 0.0] for x in x_positions]
    configs.append(config1)
    
    # Config 2: Two rows offset
    # Row 1: 6 hexagons, Row 2: 5 hexagons
    config2 = []
    row1_y = -np.sqrt(3)/2
    row2_y = np.sqrt(3)/2
    for i in range(6):
        x = i * np.sqrt(3) - 2.5 * np.sqrt(3)
        config2.append([x, row1_y, 0.0])
    for i in range(5):
        x = i * np.sqrt(3) - 2.0 * np.sqrt(3)
        config2.append([x, row2_y, 0.0])
    configs.append(config2)
    
    # Config 3: Three rows
    # 4, 4, 3
    config3 = []
    ys = [-np.sqrt(3), 0.0, np.sqrt(3)]
    counts = [4, 4, 3]
    for r, (y, count) in enumerate(zip(ys, counts)):
        for i in range(count):
            x = i * np.sqrt(3) - (count-1) * np.sqrt(3) / 2
            if r % 2 == 1:
                x += np.sqrt(3)/2
            config3.append([x, y, 0.0])
    configs.append(config3)
    
    # Config 4: Hexagonal cluster
    # Center + 6 around + 4 more
    config4 = [[0.0, 0.0, 0.0]]
    for i in range(6):
        angle = i * np.pi / 3
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        config4.append([x, y, 0.0])
    # Add 4 more in the next ring
    for i in range(4):
        angle = i * np.pi / 3 + np.pi/6
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        config4.append([x, y, 0.0])
    configs.append(config4)
    
    # For each config, find the minimum enclosing hexagon
    for inner in configs:
        # Find bounding box
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # The outer hexagon needs to contain all vertices of all inner hexagons
        # For a hexagon with angle 0, the constraints are based on the three axis directions
        
        # For each inner hexagon, compute the maximum extent in each of the 3 hexagon axes
        # For a hexagon centered at (cx, cy) with angle a, the support function in direction theta is:
        # h(theta) = cx*cos(theta) + cy*sin(theta) + side * max_i cos(theta - angle - i*pi/3)
        
        # For the outer hexagon (angle 0), the 3 directions are 0, 60, 120 degrees
        # The outer hexagon contains a point if:
        # |x| <= S
        # |y| <= S*sqrt(3)/2
        # |sqrt(3)*x + y| <= S*sqrt(3)
        # |sqrt(3)*x - y| <= S*sqrt(3)
        
        # So we need S >= max over all points of: |x|, |y|/(sqrt(3)/2), |sqrt(3)*x+y|/sqrt(3), |sqrt(3)*x-y|/sqrt(3)
        
        max_s = 0
        for cx, cy, angle in inner:
            verts = hex_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                s1 = abs(vx)
                s2 = abs(vy) / (np.sqrt(3)/2)
                s3 = abs(np.sqrt(3)*vx + vy) / np.sqrt(3)
                s4 = abs(np.sqrt(3)*vx - vy) / np.sqrt(3)
                max_s = max(max_s, s1, s2, s3, s4)
        
        # Check validity
        if all_inside_outer(inner, (0.0, 0.0), max_s, 0.0) and no_overlaps(inner):
            if max_s < best_score:
                best_score = max_s
                best_config = (inner, (0.0, 0.0), max_s, 0.0)
    
    if best_config is not None:
        return best_config
    
    # Fallback: use the first config
    return configs[0], (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-START
def construct():
    """Construct N unit hexagons inside a minimal outer hexagon."""
    N_local = 11
    
    # Try multiple arrangements and pick the best
    best_score = float('inf')
    best_result = None
    
    # Generate several candidate configurations
    candidates = []
    
    # 1. Single row
    row1 = [[i * np.sqrt(3) - (N_local-1)*np.sqrt(3)/2, 0.0, 0.0] for i in range(N_local)]
    candidates.append(row1)
    
    # 2. Two rows: 6 and 5
    two_rows = []
    for i in range(6):
        two_rows.append([i * np.sqrt(3) - 2.5*np.sqrt(3), -np.sqrt(3)/2, 0.0])
    for i in range(5):
        two_rows.append([i * np.sqrt(3) - 2.0*np.sqrt(3), np.sqrt(3)/2, 0.0])
    candidates.append(two_rows)
    
    # 3. Three rows: 4, 4, 3
    three_rows = []
    for i in range(4):
        three_rows.append([i * np.sqrt(3) - 1.5*np.sqrt(3), -np.sqrt(3), 0.0])
    for i in range(4):
        three_rows.append([i * np.sqrt(3) - 1.5*np.sqrt(3) + np.sqrt(3)/2, 0.0, 0.0])
    for i in range(3):
        three_rows.append([i * np.sqrt(3) - 1.0*np.sqrt(3), np.sqrt(3), 0.0])
    candidates.append(three_rows)
    
    # 4. Hexagonal cluster: center + 6 + 4
    cluster = [[0.0, 0.0, 0.0]]
    for i in range(6):
        a = i * np.pi / 3
        cluster.append([np.sqrt(3)*np.cos(a), np.sqrt(3)*np.sin(a), 0.0])
    for i in range(4):
        a = i * np.pi / 3 + np.pi/6
        cluster.append([2*np.sqrt(3)*np.cos(a), 2*np.sqrt(3)*np.sin(a), 0.0])
    candidates.append(cluster)
    
    # 5. Compact triangular-ish
    compact = []
    positions = [
        (0, 0), (np.sqrt(3), 0), (2*np.sqrt(3), 0),
        (np.sqrt(3)/2, np.sqrt(3)/2), (3*np.sqrt(3)/2, np.sqrt(3)/2), (5*np.sqrt(3)/2, np.sqrt(3)/2),
        (0, np.sqrt(3)), (np.sqrt(3), np.sqrt(3)), (2*np.sqrt(3), np.sqrt(3)),
        (np.sqrt(3)/2, 3*np.sqrt(3)/2), (3*np.sqrt(3)/2, 3*np.sqrt(3)/2)
    ]
    # Center them
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    for x, y in positions:
        compact.append([x - cx0, y - cy0, 0.0])
    candidates.append(compact)
    
    for inner in candidates:
        # Compute minimum enclosing hexagon side length
        max_s = 0.0
        for cx, cy, angle in inner:
            verts = hex_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                s1 = abs(vx)
                s2 = abs(vy) / (np.sqrt(3)/2)
                s3 = abs(np.sqrt(3)*vx + vy) / np.sqrt(3)
                s4 = abs(np.sqrt(3)*vx - vy) / np.sqrt(3)
                max_s = max(max_s, s1, s2, s3, s4)
        
        # Validate
        if all_inside_outer(inner, (0.0, 0.0), max_s, 0.0) and no_overlaps(inner):
            if max_s < best_score:
                best_score = max_s
                best_result = (inner, (0.0, 0.0), max_s, 0.0)
    
    if best_result is not None:
        return best_result
    
    # Fallback
    return candidates[0], (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END

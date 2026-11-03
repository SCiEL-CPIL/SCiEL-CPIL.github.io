import math
import time
import random

N = 11

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Returns list of 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        a = math.radians(angle_deg + i * 60)
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        verts.append((x, y))
    return verts

def min_outer_side(vertices):
    """
    Estimates the minimum side length of a regular hexagon enclosing the given vertices.
    Assumes the enclosing hexagon is centered at the centroid of the vertices.
    """
    if not vertices:
        return 0.0, 0.0
    
    # Centroid
    cx = sum(p[0] for p in vertices) / len(vertices)
    cy = sum(p[1] for p in vertices) / len(vertices)
    
    # Precompute distance and angle from centroid
    items = []
    for x, y in vertices:
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)
        ang = math.degrees(math.atan2(dy, dx))
        items.append((dist, ang))
    
    best_s = float('inf')
    best_a = 0.0
    
    # Sample outer hexagon angles
    # The function is periodic with 60 degrees.
    for alpha_deg in range(0, 60, 1):
        s = 0.0
        for dist, ang in items:
            rel = (ang - alpha_deg) % 60.0
            rad_rel = math.radians(rel)
            cos_val = math.cos(rad_rel - math.radians(30))
            # cos_val is positive in [0, 60]
            r_unit = (math.sqrt(3) / 2.0) / cos_val
            req_s = dist / r_unit
            if req_s > s:
                s = req_s
        if s < best_s:
            best_s = s
            best_a = float(alpha_deg)
            
    # Fine tune around best integer angle
    for alpha_deg in [best_a - 0.5, best_a, best_a + 0.5]:
        s = 0.0
        for dist, ang in items:
            rel = (ang - alpha_deg) % 60.0
            rad_rel = math.radians(rel)
            cos_val = math.cos(rad_rel - math.radians(30))
            r_unit = (math.sqrt(3) / 2.0) / cos_val
            req_s = dist / r_unit
            if req_s > s:
                s = req_s
        if s < best_s:
            best_s = s
            best_a = alpha_deg

    return best_s, best_a

def check_collision(hex1, hex2):
    """
    hex1, hex2: lists of 6 vertices.
    Returns True if they intersect.
    Uses SAT.
    """
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            dx = x2 - x1
            dy = y2 - y1
            norm = (-dy, dx)
            length = math.hypot(norm[0], norm[1])
            if length > 1e-9:
                normals.append((norm[0] / length, norm[1] / length))
        return normals

    normals1 = get_normals(hex1)
    normals2 = get_normals(hex2)
    
    for nx, ny in normals1 + normals2:
        min1, max1 = float('inf'), -float('inf')
        for x, y in hex1:
            p = x * nx + y * ny
            if p < min1: min1 = p
            if p > max1: max1 = p
            
        min2, max2 = float('inf'), -float('inf')
        for x, y in hex2:
            p = x * nx + y * ny
            if p < min2: min2 = p
            if p > max2: max2 = p
            
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = 140.0
    
    d = math.sqrt(3)
    init_positions = [
        (0.0, 0.0),
        (d, 0.0),
        (-d, 0.0),
        (d/2, 1.5),
        (-d/2, 1.5),
        (d/2, -1.5),
        (-d/2, -1.5),
        (2*d, 0.0),
        (-2*d, 0.0),
        (1.5*d, 1.5),
        (-1.5*d, 1.5)
    ]
    
    positions = [[p[0], p[1], 0.0] for p in init_positions]
    
    def get_all_vertices(positions):
        verts = []
        for p in positions:
            verts.extend(hex_vertices(p[0], p[1], p[2]))
        return verts

    def is_valid(positions):
        hexes = [hex_vertices(p[0], p[1], p[2]) for p in positions]
        for i in range(len(hexes)):
            for j in range(i + 1, len(hexes)):
                if check_collision(hexes[i], hexes[j]):
                    return False
        return True

    verts = get_all_vertices(positions)
    best_s, best_angle = min_outer_side(verts)
    best_positions = [p[:] for p in positions]
    
    step = 0.5
    no_improve = 0
    
    while time.time() - start_time < deadline:
        i = random.randint(0, N - 1)
        
        new_x = positions[i][0] + random.uniform(-step, step)
        new_y = positions[i][1] + random.uniform(-step, step)
        new_angle = positions[i][2] + random.uniform(-5, 5)
        
        old_p = positions[i][:]
        positions[i] = [new_x, new_y, new_angle]
        
        if is_valid(positions):
            verts = get_all_vertices(positions)
            s, ang = min_outer_side(verts)
            if s < best_s:
                best_s = s
                best_positions = [p[:] for p in positions]
                best_angle = ang
                no_improve = 0
                step *= 0.99
            else:
                positions[i] = old_p
                no_improve += 1
        else:
            positions[i] = old_p
            no_improve += 1
            
        if no_improve > 100:
            step = 0.5
            no_improve = 0
            
        if step < 0.01:
            step = 0.01

    verts = get_all_vertices(best_positions)
    final_s, final_angle = min_outer_side(verts)
    
    v_cx = sum(p[0] for p in verts) / len(verts)
    v_cy = sum(p[1] for p in verts) / len(verts)
    
    inner = [[p[0], p[1], p[2]] for p in best_positions]
    center = (v_cx, v_cy)
    side = final_s
    angle = final_angle
    
    return inner, center, side, angle

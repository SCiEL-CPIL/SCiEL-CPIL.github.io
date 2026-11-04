"""Unit regular hexagons inside a regular hexagon n=12."""
import time
import math
import numpy as np

N = 12
DEADLINE = 150.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - 15.0 - _elapsed()  # safety margin

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon (side length 'side', center (cx,cy), rotated by angle_deg)."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def support_function_point(verts, dx, dy):
    """Support function of polygon in direction (dx,dy)."""
    return max(vx * dx + vy * dy for vx, vy in verts)

def polygon_area_signed(verts):
    """Signed area via shoelace."""
    n = len(verts)
    a = 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0

def point_in_hexagon(px, py, verts):
    """Ray-casting point in polygon."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def hexagon_overlap_area(verts1, verts2):
    """Approximate overlap area of two convex polygons using Sutherland-Hodgman clipping."""
    # Clip polygon1 by polygon2
    def clip_polygon(subject, clip):
        def inside(p, edge_start, edge_end):
            ax, ay = edge_start
            bx, by = edge_end
            px, py = p
            return (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= -1e-12

        def intersect(p1, p2, q1, q2):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = q1
            x4, y4 = q2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-12:
                return p1
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        output = list(subject)
        for i in range(len(clip)):
            if not output:
                break
            input_list = output
            output = []
            s = input_list[-1]
            a = clip[i]
            b = clip[(i + 1) % len(clip)]
            for e in input_list:
                if inside(e, a, b):
                    if not inside(s, a, b):
                        output.append(intersect(s, e, a, b))
                    output.append(e)
                elif inside(s, a, b):
                    output.append(intersect(s, e, a, b))
                s = e
        return output

    clipped = clip_polygon(verts1, verts2)
    if len(clipped) < 3:
        return 0.0
    return abs(polygon_area_signed(clipped))

def compute_overlap_penalty(inner, outer_side, outer_center, outer_angle, weight=1000.0):
    """Penalty for overlaps: sum of overlap areas * weight + max distance violation."""
    ox, oy = outer_center
    outer_verts = hex_vertices(ox, oy, outer_side, outer_angle)
    
    # Check all inner hexagons are inside outer
    penalty = 0.0
    for (ix, iy, iang) in inner:
        iv = hex_vertices(ix, iy, 1.0, iang)
        # Check each vertex of inner is in outer (approximate)
        for vx, vy in iv:
            if not point_in_hexagon(vx, vy, outer_verts):
                # Distance to outer boundary
                # Find closest point on outer boundary
                min_dist = float('inf')
                for i in range(6):
                    ax, ay = outer_verts[i]
                    bx, by = outer_verts[(i + 1) % 6]
                    # Point to line segment distance
                    dx, dy = bx - ax, by - ay
                    len2 = dx * dx + dy * dy
                    if len2 < 1e-12:
                        continue
                    t = ((vx - ax) * dx + (vy - ay) * dy) / len2
                    t = max(0, min(1, t))
                    cx = ax + t * dx
                    cy = ay + t * dy
                    d = math.sqrt((vx - cx) ** 2 + (vy - cy) ** 2)
                    min_dist = min(min_dist, d)
                penalty += weight * min_dist
        # Check overlap with other inner hexagons
        for j, (jx, jy, jang) in enumerate(inner):
            if inner.index((ix, iy, iang)) >= j:
                continue
            jv = hex_vertices(jx, jy, 1.0, jang)
            ov = hexagon_overlap_area(iv, jv)
            if ov > 1e-6:
                penalty += weight * ov
    return penalty

def evaluate_config(inner, outer_side, outer_center, outer_angle):
    """Full evaluation: check validity and return outer_side if valid, else penalty."""
    ox, oy = outer_center
    outer_verts = hex_vertices(ox, oy, outer_side, outer_angle)
    
    # Check all vertices of inner hexagons are inside outer
    for (ix, iy, iang) in inner:
        iv = hex_vertices(ix, iy, 1.0, iang)
        for vx, vy in iv:
            if not point_in_hexagon(vx, vy, outer_verts):
                return None  # invalid
    
    # Check no overlap between inner hexagons
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            ix, iy, iang = inner[i]
            jx, jy, jang = inner[j]
            iv = hex_vertices(ix, iy, 1.0, iang)
            jv = hex_vertices(jx, jy, 1.0, jang)
            ov = hexagon_overlap_area(iv, jv)
            if ov > 1e-6:
                return None  # invalid
    
    return outer_side

def initial_layout():
    """Create an initial layout: 12 hexagons in a ring around center."""
    inner = []
    # 6 hexagons in a ring at distance 2, and 6 more at distance 3 or similar
    # For now, place them in two rings
    for k in range(6):
        angle = k * 60.0
        r = 2.5
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        inner.append([x, y, 0.0])
    for k in range(6):
        angle = k * 60.0 + 30.0
        r = 4.0
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        inner.append([x, y, 0.0])
    return inner

def compute_min_side(inner, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    ox, oy = outer_center
    # For a regular hexagon with center (ox,oy), side s, angle a, a point (px,py) is inside
    # if for all 6 edges, the point is on the correct side.
    # We can compute the required side length by finding the max "radial extent" in each of the 6 directions.
    
    a = math.radians(outer_angle)
    # The 6 outward normals of the outer hexagon
    normals = []
    for k in range(6):
        theta = a + k * math.pi / 3.0 + math.pi / 6  # normals are offset by 30 degrees
        normals.append((math.cos(theta), math.sin(theta)))
    
    # For each inner hexagon, find the max projection along each normal
    max_projs = [0.0] * 6
    for (ix, iy, iang) in inner:
        iv = hex_vertices(ix, iy, 1.0, iang)
        for nx, ny in normals:
            proj = max(vx * nx + vy * ny for vx, vy in iv)
            # The outer hexagon center is at (ox, oy), so the half-width in this direction is:
            # proj - (ox*nx + oy*ny) must be <= s * (apothem in this direction)
            # For a regular hexagon of side s, the apothem is s * sqrt(3)/2
            # The support function in the normal direction is s * sqrt(3)/2
            required = proj - (ox * nx + oy * ny)
            if required > max_projs[normals.index((nx, ny))]:
                max_projs[normals.index((nx, ny))] = required
    
    # The side length s must satisfy: s * sqrt(3)/2 >= max_projs[k] for all k
    # So s >= 2 / sqrt(3) * max(max_projs)
    max_p = max(max_projs) if max_projs else 0
    min_side = (2.0 / math.sqrt(3.0)) * max_p
    return max(min_side, 0.0)

def optimize(inner, max_time=60.0):
    """Local optimization: minimize outer side length by adjusting inner positions/rotations and outer center/angle."""
    start = time.time()
    best_inner = [list(x) for x in inner]
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # First, compute initial side
    current_center = (0.0, 0.0)
    current_angle = 0.0
    current_side = compute_min_side(inner, current_center, current_angle)
    
    # Verify validity
    if evaluate_config(inner, current_side, current_center, current_angle) is not None:
        best_inner = [list(x) for x in inner]
        best_side = current_side
        best_center = current_center
        best_angle = current_angle
    
    # Gradient-free local search
    np.random.seed(42)
    lr = 0.5
    step = 0.2
    
    for iteration in range(2000):
        if time.time() - start > max_time or _remaining() < 30:
            break
        
        # Random perturbation
        cand = []
        for (x, y, ang) in best_inner:
            nx = x + np.random.normal(0, step)
            ny = y + np.random.normal(0, step)
            nang = ang + np.random.normal(0, step * 10)
            cand.append([nx, ny, nang])
        
        # Try adjusting center and angle
        cand_center = (best_center[0] + np.random.normal(0, step), best_center[1] + np.random.normal(0, step))
        cand_angle = best_angle + np.random.normal(0, step * 10)
        
        cand_side = compute_min_side(cand, cand_center, cand_angle)
        
        # Check validity
        valid = evaluate_config(cand, cand_side, cand_center, cand_angle)
        if valid is not None and cand_side < best_side:
            best_inner = [list(x) for x in cand]
            best_side = cand_side
            best_center = cand_center
            best_angle = cand_angle
            step = max(0.05, step * 0.95)
        else:
            step = min(1.0, step * 1.05)
    
    return best_inner, best_center, best_side, best_angle

def construct():
    start_time = time.time()
    
    # Initial layout
    inner = initial_layout()
    
    # Optimize
    best_inner, best_center, best_side, best_angle = optimize(inner, max_time=min(100.0, _remaining() - 20))
    
    # Final polish: try to reduce side further by fine-tuning
    if _remaining() > 30:
        inner2, center2, side2, angle2 = optimize(best_inner, max_time=min(60.0, _remaining() - 20))
        if side2 < best_side:
            best_inner, best_center, best_side, best_angle = inner2, center2, side2, angle2
    
    # Ensure we have a valid result
    final_side = best_side
    if not math.isfinite(final_side) or final_side <= 0:
        # Fallback: place hexagons in a line
        final_inner = [[3.0 * k, 0.0, 0.0] for k in range(-N // 2, N - N // 2)]
        final_center = (0.0, 0.0)
        final_angle = 0.0
        final_side = compute_min_side(final_inner, final_center, final_angle)
    
    return final_inner, list(best_center), float(best_side), float(best_angle)

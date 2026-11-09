import math
import time
import random
import itertools
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    sqrt3 = math.sqrt(3.0)

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = -math.radians(angle_deg)
        rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        for k in range(6):
            normal_angle = math.pi / 6 + k * math.pi / 3.0
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            dot = rx * nx + ry * ny
            if dot > side * sqrt3 / 2.0 + 1e-10:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        if d1 == 0 and on_segment(p3, p4, p1):
            return True
        if d2 == 0 and on_segment(p3, p4, p2):
            return True
        if d3 == 0 and on_segment(p1, p2, p3):
            return True
        if d4 == 0 and on_segment(p1, p2, p4):
            return True
        return False

    def on_segment(a, b, c):
        return min(a[0], b[0]) - 1e-10 <= c[0] <= max(a[0], b[0]) + 1e-10 and \
               min(a[1], b[1]) - 1e-10 <= c[1] <= max(a[1], b[1]) + 1e-10

    def check_disjoint(hex1, hex2):
        cx1, cy1, a1 = hex1
        cx2, cy2, a2 = hex2
        v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
        v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, cx2, cy2, 1.0, a2):
                return False
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, cx1, cy1, 1.0, a1):
                return False
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j + 1) % 6]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def compute_required_side(inner, outer_center, outer_angle):
        ox, oy = outer_center
        max_s = 0.0
        angle_rad = -math.radians(outer_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        normals = []
        for k in range(6):
            normal_angle = math.pi / 6 + k * math.pi / 3.0
            normals.append((math.cos(normal_angle), math.sin(normal_angle)))
        
        for cx, cy, a_deg in inner:
            verts = hexagon_vertices(cx, cy, 1.0, a_deg)
            for vx, vy in verts:
                dx = vx - ox
                dy = vy - oy
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                for nx, ny in normals:
                    dot = rx * nx + ry * ny
                    if dot > 0:
                        s_req = dot * 2.0 / sqrt3
                        if s_req > max_s:
                            max_s = s_req
        return max_s

    def is_valid(inner):
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                if not check_disjoint(inner[i], inner[j]):
                    return False
        return True

    # Generate candidate grids
    # Grid 1: Standard hex grid (flat top)
    def gen_hex_grid(radius):
        pts = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                if abs(q + r) <= radius:
                    x = 1.5 * q
                    y = sqrt3 * (r + q / 2.0)
                    pts.append((x, y))
        return pts

    # Grid 2: Rotated hex grid (pointy top) - same points, just different interpretation
    # But since we can rotate inner hexagons, the grid shape is what matters.
    # Let's also try a "loose" grid where we allow slight offsets? No, stick to discrete for speed.
    
    # Try radius 2 (19 points) and radius 3 (37 points)
    grids = [
        gen_hex_grid(2),
        gen_hex_grid(3),
    ]
    
    # Also try a "compressed" grid? No, unit hexagons on hex grid touch.
    # If we compress, they overlap. So grid points must be at least sqrt(3) apart for same orientation.
    # The hex grid generated above has distance sqrt(3) between neighbors.
    # So placing unit hexagons (side 1) on these points with 0 rotation works perfectly.
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Search over subsets of points and orientations
    # For radius 2, we have 19 points. We need to pick 12.
    # For radius 3, we have 37 points. We need to pick 12.
    
    # To make it faster, we can fix the orientation of all inner hexagons to 0 first.
    # Then refine.
    
    for grid in grids:
        if time.time() > deadline - 5:
            break
            
        n_pts = len(grid)
        if n_pts < N:
            continue
            
        # If grid is small enough, iterate all combinations
        # C(19, 12) = 50388. Very fast.
        # C(37, 12) is huge (~50 billion). We need a heuristic for larger grids.
        
        if n_pts <= 20:
            combos = itertools.combinations(range(n_pts), N)
            for combo in combos:
                if time.time() > deadline - 2:
                    break
                inner = []
                for idx in combo:
                    x, y = grid[idx]
                    inner.append([x, y, 0.0])
                
                if not is_valid(inner):
                    continue
                
                # Compute bounding hexagon
                # Try a few outer angles and centers
                # Optimal center is likely the centroid of the points or close to it
                cx_avg = sum(p[0] for p in inner) / N
                cy_avg = sum(p[1] for p in inner) / N
                
                # Try outer angles: 0, 30, 60, 90...
                # And centers: centroid, and slight offsets?
                # For now, just centroid and angle 0, 30, 60
                for out_angle in [0, 30, 60, 90, 120, 150]:
                    # Also try shifting center slightly to minimize max projection?
                    # For now, just centroid
                    s = compute_required_side(inner, (cx_avg, cy_avg), out_angle)
                    if s < best_score:
                        best_score = s
                        best_inner = [list(h) for h in inner]
                        best_center = (cx_avg, cy_avg)
                        best_angle = out_angle
        
        else:
            # Heuristic for larger grids:
            # Pick the N points that are "most central"
            # Sort by distance to origin
            sorted_pts = sorted(enumerate(grid), key=lambda x: x[1][0]**2 + x[1][1]**2)
            # Take the first N
            combo = [idx for idx, _ in sorted_pts[:N]]
            inner = []
            for idx in combo:
                x, y = grid[idx]
                inner.append([x, y, 0.0])
            
            if is_valid(inner):
                cx_avg = sum(p[0] for p in inner) / N
                cy_avg = sum(p[1] for p in inner) / N
                for out_angle in [0, 30, 60, 90, 120, 150]:
                    s = compute_required_side(inner, (cx_avg, cy_avg), out_angle)
                    if s < best_score:
                        best_score = s
                        best_inner = [list(h) for h in inner]
                        best_center = (cx_avg, cy_avg)
                        best_angle = out_angle
            
            # Also try random subsets from the larger grid
            for _ in range(500):
                if time.time() > deadline - 2:
                    break
                combo = random.sample(range(n_pts), N)
                inner = []
                for idx in combo:
                    x, y = grid[idx]
                    inner.append([x, y, 0.0])
                
                if not is_valid(inner):
                    continue
                
                cx_avg = sum(p[0] for p in inner) / N
                cy_avg = sum(p[1] for p in inner) / N
                for out_angle in [0, 30, 60]:
                    s = compute_required_side(inner, (cx_avg, cy_avg), out_angle)
                    if s < best_score:
                        best_score = s
                        best_inner = [list(h) for h in inner]
                        best_center = (cx_avg, cy_avg)
                        best_angle = out_angle

    if best_inner is None:
        # Fallback: simple arrangement
        inner = []
        axial_positions = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
            (2, 0), (2, -1), (1, 1), (-1, 2), (-2, 1)
        ]
        for q, r in axial_positions:
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            inner.append([x, y, 0.0])
        best_score = compute_required_side(inner, (0,0), 0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # Refine: Optimize outer center and angle for the best inner configuration
    # Use a local search on outer params
    if best_inner is not None and time.time() < deadline - 5:
        current_score = best_score
        current_center = list(best_center)
        current_angle = best_angle
        
        # Gradient-free optimization for outer params
        # Since inner is fixed, the problem is convex in outer params? No, support function is max of linear functions, so it's convex.
        # We can use scipy.optimize.minimize or just grid search / Nelder-Mead.
        
        # Let's use a simple coordinate descent / hill climbing
        step_c = 0.1
        step_a = 1.0 # degrees
        
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            
            improved = False
            
            # Try moving center
            for dx in [-step_c, 0, step_c]:
                for dy in [-step_c, 0, step_c]:
                    new_cx = current_center[0] + dx
                    new_cy = current_center[1] + dy
                    s = compute_required_side(best_inner, (new_cx, new_cy), current_angle)
                    if s < current_score - 1e-6:
                        current_score = s
                        current_center = [new_cx, new_cy]
                        improved = True
            
            # Try rotating
            for da in [-step_a, 0, step_a]:
                new_a = current_angle + da
                s = compute_required_side(best_inner, (current_center[0], current_center[1]), new_a)
                if s < current_score - 1e-6:
                    current_score = s
                    current_angle = new_a
                    improved = True
            
            if not improved:
                # Reduce step size
                step_c *= 0.5
                step_a *= 0.5
                if step_c < 1e-4:
                    break
        
        best_score = current_score
        best_center = (current_center[0], current_center[1])
        best_angle = current_angle

    # Add safety margin
    side = best_score * 1.001

    return best_inner, best_center, side, best_angle

# EVOLVE-BLOCK-START
# The code above is the complete construct function.
# We need to wrap it in the expected structure if the evaluator expects a specific file format,
# but the prompt says "Keep the program's entry-point function name(s) and their return interface EXACTLY".
# The current program has `def construct():` at the top level.
# I will output the full code block as requested.
# EVOLVE-BLOCK-END

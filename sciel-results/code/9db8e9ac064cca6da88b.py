"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # --- Geometry helpers ---
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = a + k * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def fast_outer_side(inner_positions, outer_center, outer_angle):
        """
        Calculate minimum outer side length such that all inner hexagons fit inside.
        Uses projection of vertices onto outer hexagon normals.
        """
        cx, cy = outer_center
        a_outer = math.radians(outer_angle)
        
        # Normals of the outer hexagon sides
        # Vertices are at outer_angle + k*60.
        # Side midpoints (normals) are at outer_angle + 30 + k*60.
        normals = []
        for k in range(6):
            theta = a_outer + math.pi/6 + k * math.pi / 3.0
            normals.append((math.cos(theta), math.sin(theta)))
        
        max_proj = 0.0
        for (ix, iy, iang) in inner_positions:
            # Vertices of inner hex
            a_inner = math.radians(iang)
            for k in range(6):
                theta = a_inner + k * math.pi / 3.0
                vx = ix + math.cos(theta)
                vy = iy + math.sin(theta)
                # Project onto each normal
                for nx, ny in normals:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    if proj > max_proj:
                        max_proj = proj
                        
        # Distance from center to side A = S * sqrt(3) / 2
        # S = 2 * A / sqrt(3)
        side = 2.0 * max_proj / math.sqrt(3)
        return side

    def polygon_overlap_area(v1, v2):
        """Compute overlap area of two convex polygons using Sutherland-Hodgman clipping."""
        def point_in_convex(p, poly):
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-12:
                    return False
            return True

        def line_intersect(p1, p2, p3, p4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-15:
                return p1
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        def polygon_area_signed(verts):
            n = len(verts)
            area = 0.0
            for i in range(n):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % n]
                area += x1 * y2 - x2 * y1
            return area / 2.0

        def clip_polygon(subject, clip):
            output = list(subject)
            for i in range(len(clip)):
                if not output:
                    break
                inp = output
                output = []
                S = inp[-1]
                for E in inp:
                    if point_in_convex(E, clip):
                        if not point_in_convex(S, clip):
                            output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                        output.append(E)
                    elif point_in_convex(S, clip):
                        output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                    S = E
            return output

        clipped = clip_polygon(v1, v2)
        if len(clipped) < 3:
            return 0.0
        return abs(polygon_area_signed(clipped))

    def objective(params):
        """
        Objective function: Minimize outer side length + overlap penalty.
        params: [x1, y1, a1, ..., x11, y11, a11, cx, cy, ca]
        """
        # Unpack
        inner_positions = []
        for i in range(N):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2] * 180.0 / math.pi
            inner_positions.append((x, y, a))
        
        cx = params[3 * N]
        cy = params[3 * N + 1]
        ca = params[3 * N + 2] * 180.0 / math.pi
        
        # Fast outer side
        outer_side = fast_outer_side(inner_positions, (cx, cy), ca)
        
        # Overlap penalty
        overlap_penalty = 0.0
        # Pre-compute vertices for all inner hexes
        verts_list = []
        for i in range(N):
            x, y, a = inner_positions[i]
            verts_list.append(hex_vertices(x, y, 1.0, a))
            
        for i in range(N):
            for j in range(i + 1, N):
                # Quick distance check
                dx = inner_positions[i][0] - inner_positions[j][0]
                dy = inner_positions[i][1] - inner_positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 2.0: # Max distance for overlap is 2 (vertex to vertex)
                    continue
                ov = polygon_overlap_area(verts_list[i], verts_list[j])
                if ov > 1e-8:
                    overlap_penalty += ov ** 2

        return outer_side + 1000.0 * overlap_penalty

    def is_valid(inner_positions, outer_center, outer_side, outer_angle):
        """Check if configuration is valid."""
        cx, cy = outer_center
        # Check containment using fast method
        actual_side = fast_outer_side(inner_positions, outer_center, outer_angle)
        if actual_side > outer_side + 1e-6:
            return False
        
        # Check non-overlap
        verts_list = []
        for i in range(N):
            x, y, a = inner_positions[i]
            verts_list.append(hex_vertices(x, y, 1.0, a))
            
        for i in range(N):
            for j in range(i + 1, N):
                dx = inner_positions[i][0] - inner_positions[j][0]
                dy = inner_positions[i][1] - inner_positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 2.0:
                    continue
                ov = polygon_overlap_area(verts_list[i], verts_list[j])
                if ov > 1e-6:
                    return False
        return True

    # --- Seed Solution ---
    def analytic_solution(angle_inner=0.0):
        """
        Construct a symmetric packing.
        """
        eps = 1e-9
        centers = [(0.0, 0.0)]
        
        # 6 neighbors
        for k in range(6):
            angle = math.radians(k * 60)
            x = (math.sqrt(3) + eps) * math.cos(angle)
            y = (math.sqrt(3) + eps) * math.sin(angle)
            centers.append((x, y))
            
        # 4 more at dist 3
        # Angles 30, 90, 150, 210
        for k in range(4):
            angle = math.radians(30 + k * 60)
            x = (3.0 + 2*eps) * math.cos(angle)
            y = (3.0 + 2*eps) * math.sin(angle)
            centers.append((x, y))
        
        inner = []
        for x, y in centers:
            inner.append([x, y, angle_inner])
        
        # Calculate side
        outer_center = (0.0, 0.0)
        outer_angle = 30.0 # Align with symmetry
        side = fast_outer_side(inner, outer_center, outer_angle)
        
        return inner, outer_center, side, outer_angle

    # Get baseline
    inner_seed, outer_center_seed, side_seed, angle_seed = analytic_solution(0.0)
    best_score = side_seed
    best_params = np.zeros(3 * N + 3)
    for i, (x, y, a) in enumerate(inner_seed):
        best_params[3*i] = x
        best_params[3*i+1] = y
        best_params[3*i+2] = a * math.pi / 180.0
    best_params[3*N] = outer_center_seed[0]
    best_params[3*N+1] = outer_center_seed[1]
    best_params[3*N+2] = angle_seed * math.pi / 180.0
    
    best_result = (inner_seed, outer_center_seed, side_seed, angle_seed)

    # Optimization
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            
            # Nelder-Mead
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
            
            if result.fun < best_score:
                best_params = result.x
                # Decode and verify
                inner_positions = []
                for i in range(N):
                    x = best_params[3 * i]
                    y = best_params[3 * i + 1]
                    a = best_params[3 * i + 2] * 180.0 / math.pi
                    inner_positions.append((x, y, a))
                
                cx = best_params[3 * N]
                cy = best_params[3 * N + 1]
                ca = best_params[3 * N + 2] * 180.0 / math.pi
                
                outer_side = fast_outer_side(inner_positions, (cx, cy), ca)
                
                if is_valid(inner_positions, (cx, cy), outer_side, ca):
                    best_score = outer_side
                    best_result = (inner_positions, (cx, cy), outer_side, ca)
                    
        except Exception:
            pass

    # Final polish if time permits
    if time.time() < deadline - 5:
        # Try a few random restarts if we have time? No, just trust the local opt.
        pass

    inner_positions, outer_center, outer_side, outer_angle = best_result
    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
# EVOLVE-BLOCK-END

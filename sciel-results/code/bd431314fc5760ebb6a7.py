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

    def point_in_convex(p, poly):
        """Check if point p is inside convex polygon poly (CCW)."""
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-12:
                return False
        return True

    def polygon_area_signed(verts):
        n = len(verts)
        area = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def line_intersect(p1, p2, p3, p4):
        """Intersection of line p1-p2 and line p3-p4."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-15:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def polygon_overlap_area(v1, v2):
        """Compute overlap area of two convex polygons using Sutherland-Hodgman clipping."""
        # Clip v1 by each edge of v2
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

    def hex_in_hexagon(inner_verts, outer_verts):
        """Check if all vertices of inner hex are inside outer hex."""
        return all(point_in_convex(v, outer_verts) for v in inner_verts)

    def outer_side_required(inner_positions, outer_center, outer_angle):
        """Find minimum outer side length such that all inner hexagons fit inside."""
        cx, cy = outer_center
        # Binary search on outer side length
        lo = 1.0
        hi = 100.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            outer_verts = hex_vertices(cx, cy, mid, outer_angle)
            all_fit = True
            for (ix, iy, iang) in inner_positions:
                inner_verts = hex_vertices(ix, iy, 1.0, iang)
                if not hex_in_hexagon(inner_verts, outer_verts):
                    all_fit = False
                    break
            if all_fit:
                hi = mid
            else:
                lo = mid
        return hi

    def is_valid(inner_positions, outer_center, outer_side, outer_angle):
        """Check if configuration is valid: no overlaps, all inside outer hex."""
        cx, cy = outer_center
        outer_verts = hex_vertices(cx, cy, outer_side, outer_angle)
        # Check containment
        for (ix, iy, iang) in inner_positions:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            if not hex_in_hexagon(inner_verts, outer_verts):
                return False
        # Check non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(*inner_positions[i][:2], 1.0, inner_positions[i][2])
                vj = hex_vertices(*inner_positions[j][:2], 1.0, inner_positions[j][2])
                if polygon_overlap_area(vi, vj) > 1e-8:
                    return False
        return True

    # --- Analytic seed solution based on optimal lattice packing ---
    def analytic_solution():
        """
        Construct the optimal concentric ring packing:
        - 1 center hex
        - 6 ring-1 hexagons at distance sqrt(3) (touching)
        - 4 ring-2 hexagons at distance 3 (touching ring-1)
        All hexagons rotated by 30 degrees (flat-top orientation).
        Outer hexagon also rotated by 30 degrees.
        """
        eps = 1e-9
        centers = []
        centers.append((0.0, 0.0))
        
        # 6 neighbors at dist sqrt(3) + eps
        for k in range(6):
            angle = math.radians(k * 60)
            x = (math.sqrt(3) + eps) * math.cos(angle)
            y = (math.sqrt(3) + eps) * math.sin(angle)
            centers.append((x, y))
            
        # 4 more at dist 3 + 2*eps
        # Angles 30, 90, 150, 210
        for k in range(4):
            angle = math.radians(30 + k * 60)
            x = (3.0 + 2*eps) * math.cos(angle)
            y = (3.0 + 2*eps) * math.sin(angle)
            centers.append((x, y))
        
        inner = []
        for x, y in centers:
            inner.append([x, y, 30.0])
        
        # Calculate required outer side length analytically
        # Outer hexagon orientation: 30 degrees (flat top/bottom, vertical sides)
        # Width = sqrt(3) * S
        # Height = 2 * S
        
        # Max X extent:
        # Ring 2 hexagons at 30 and 150 degrees have centers at x = (3+2eps)*cos(30) = (3+2eps)*sqrt(3)/2
        # Rightmost point of that hex (rotated 30 deg): center_x + (sqrt(3)/2)*1
        # Max X = (3+2eps)*sqrt(3)/2 + sqrt(3)/2 = (4+2eps)*sqrt(3)/2 = (2+eps)*sqrt(3)
        
        # Max Y extent:
        # Ring 2 hexagon at 90 deg has center at y = 3+2eps
        # Topmost point: vertex at 90 deg from center, so y_offset = 1
        # Max Y = 3+2eps + 1 = 4+2eps
        
        max_x = (2 + eps) * math.sqrt(3)
        max_y = 4 + 2 * eps
        
        s_x = 2 * max_x / math.sqrt(3)
        s_y = max_y
        
        side = max(s_x, s_y)
        
        # Add a tiny margin to ensure containment
        side += 1e-9
        
        return inner, (0.0, 0.0), side, 30.0

    # --- High-precision 1D search over global rotation ---
    def min_side_for_rotation(theta_rad):
        """
        Calculate the minimum outer hexagon side length for a cluster rotated by theta_rad.
        Assumes outer hexagon is fixed at angle 0 (flat-top).
        Cluster local geometry is defined by centers and local angles.
        """
        # Pre-computed local centers and angles from analytic_solution
        # We re-use the logic from analytic_solution but extract local coords
        # For speed, we hardcode the local centers based on the seed's analytic_solution
        # Centers:
        # 1: (0,0)
        # 6: dist sqrt(3), angles 0, 60, ...
        # 4: dist 3, angles 30, 90, ...
        
        local_centers = []
        local_centers.append((0.0, 0.0))
        for k in range(6):
            angle = math.radians(k * 60)
            x = math.sqrt(3) * math.cos(angle)
            y = math.sqrt(3) * math.sin(angle)
            local_centers.append((x, y))
        for k in range(4):
            angle = math.radians(30 + k * 60)
            x = 3.0 * math.cos(angle)
            y = 3.0 * math.sin(angle)
            local_centers.append((x, y))
            
        # Local angle for all inner hexagons is 30 degrees
        local_angle_rad = math.radians(30.0)
        
        # Compute vertices of all 11 hexagons in local frame
        # There are 66 vertices total
        all_verts = []
        for cx, cy in local_centers:
            for k in range(6):
                theta_v = local_angle_rad + k * math.pi / 3.0
                vx = cx + math.cos(theta_v)
                vy = cy + math.sin(theta_v)
                all_verts.append((vx, vy))
                
        # Rotate all vertices by theta_rad
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        
        max_proj = 0.0
        # Outer hexagon normals for angle 0 (flat top)
        # Normals point to midpoints of sides: 0, 60, 120, 180, 240, 300 degrees
        # Wait, flat top means vertices at 30, 90...?
        # If vertices are at 30, 90, 150, 210, 270, 330:
        # Side midpoints (normals) are at 0, 60, 120, 180, 240, 300.
        # Apothem A = S * sqrt(3) / 2.
        # Constraint: p . n <= A.
        # So A_max = max(p . n).
        # S = 2 * A_max / sqrt(3).
        
        normals = []
        for k in range(6):
            angle_n = math.radians(k * 60)
            normals.append((math.cos(angle_n), math.sin(angle_n)))
            
        for vx, vy in all_verts:
            # Rotate
            rx = vx * cos_t - vy * sin_t
            ry = vx * sin_t + vy * cos_t
            for nx, ny in normals:
                proj = rx * nx + ry * ny
                if proj > max_proj:
                    max_proj = proj
                    
        return (2.0 * max_proj) / math.sqrt(3.0)

    # Golden Section Search
    def golden_section_search(func, a, b, tol=1e-6, max_iter=100):
        invphi = (math.sqrt(5) - 1) / 2
        invphi2 = invphi * invphi
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
        fc = func(c)
        fd = func(d)
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if b - a < tol:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - invphi * (b - a)
                fc = func(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + invphi * (b - a)
                fd = func(d)
        return (a + b) / 2

    # Get the analytic solution as baseline
    inner_seed, outer_center_seed, side_seed, angle_seed = analytic_solution()
    best_score = side_seed
    best_result = (inner_seed, outer_center_seed, side_seed, angle_seed)

    # Perform 1D search
    if time.time() < deadline - 10:
        # Search range [0, pi/3] due to 6-fold symmetry of the outer hexagon
        # Note: The cluster is not 6-fold symmetric, but the outer hexagon is.
        # So we only need to search 60 degrees.
        theta_opt = golden_section_search(min_side_for_rotation, 0, math.pi / 3.0, tol=1e-7, max_iter=200)
        
        # Construct the result
        # Rotate local centers by theta_opt
        # Local centers logic same as in min_side_for_rotation
        local_centers = []
        local_centers.append((0.0, 0.0))
        for k in range(6):
            angle = math.radians(k * 60)
            x = math.sqrt(3) * math.cos(angle)
            y = math.sqrt(3) * math.sin(angle)
            local_centers.append((x, y))
        for k in range(4):
            angle = math.radians(30 + k * 60)
            x = 3.0 * math.cos(angle)
            y = 3.0 * math.sin(angle)
            local_centers.append((x, y))
            
        cos_t = math.cos(theta_opt)
        sin_t = math.sin(theta_opt)
        
        inner_positions = []
        for cx, cy in local_centers:
            rx = cx * cos_t - cy * sin_t
            ry = cx * sin_t + cy * cos_t
            # Angle is 30 deg + theta_opt
            angle_deg = (30.0 + math.degrees(theta_opt)) % 360.0
            inner_positions.append((rx, ry, angle_deg))
            
        outer_side = min_side_for_rotation(theta_opt)
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        # Verify validity (optional but safe)
        # The lattice is touching, so overlap area should be 0.
        # We just check if it fits in the computed side.
        if is_valid(inner_positions, outer_center, outer_side, outer_angle):
            if outer_side < best_score:
                best_score = outer_side
                best_result = (inner_positions, outer_center, outer_side, outer_angle)
        else:
            # If invalid due to floating point, use seed
            pass

    # Return the best result
    inner_positions, outer_center, outer_side, outer_angle = best_result
    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
# EVOLVE-BLOCK-END

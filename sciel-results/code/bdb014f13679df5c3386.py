"""Improved program for hexagon packing in hexagon."""
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
        """Return 6 vertices of a regular hexagon (pointy-top at angle=0)."""
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
        """Compute overlap area of two convex polygons."""
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

    def min_separation(v1, v2):
        """Compute minimum distance between two convex polygons (0 if overlapping)."""
        if polygon_overlap_area(v1, v2) > 1e-10:
            return 0.0
        min_dist = float('inf')
        for p in v1:
            for q in v2:
                d = math.hypot(p[0] - q[0], p[1] - q[1])
                if d < min_dist:
                    min_dist = d
        return min_dist

    def outer_side_required_fast(inner_params, outer_center, outer_angle):
        """
        Compute minimum outer side length using support function approach.
        For a regular hexagon with side s centered at (cx,cy) with rotation angle,
        edge normals are at angles outer_angle + 30 + k*60 degrees.
        """
        cx, cy = outer_center
        apothem_factor = math.sqrt(3) / 2.0
        normal_angles = [math.radians(outer_angle + 30 + k * 60) for k in range(6)]
        normals = [(math.cos(a), math.sin(a)) for a in normal_angles]
        
        max_val = 0.0
        for (ix, iy, iang) in inner_params:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                for nx, ny in normals:
                    proj = dx * nx + dy * ny
                    if proj > max_val:
                        max_val = proj
        return max_val / apothem_factor

    def is_valid(inner_params, outer_center, outer_side, outer_angle):
        """Check if configuration is valid: no overlaps, all inside outer hex."""
        cx, cy = outer_center
        outer_verts = hex_vertices(cx, cy, outer_side, outer_angle)
        for (ix, iy, iang) in inner_params:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            if not all(point_in_convex(v, outer_verts) for v in inner_verts):
                return False
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                if polygon_overlap_area(vi, vj) > 1e-8:
                    return False
        return True

    # --- Analytic seed solution ---
    def analytic_solution():
        """
        Construct the concentric ring packing with proper spacing.
        For unit hexagons in hexagonal lattice:
        - Adjacent hex centers are 2*sqrt(3) apart (for touching hexes)
        - Actually for flat-top hexagons, distance between touching centers is 2*sqrt(3)
        - For pointy-top, same distance
        """
        eps = 1e-9
        # Distance between centers of touching unit hexagons = 2*sqrt(3)
        d = 2.0 * math.sqrt(3) + eps
        
        centers = []
        centers.append((0.0, 0.0))
        
        # Ring 1: 6 hexagons
        for k in range(6):
            angle = math.radians(k * 60)
            x = d * math.cos(angle)
            y = d * math.sin(angle)
            centers.append((x, y))
            
        # Ring 2: 12 hexagons (we need 4 more to total 11, so take 4 from ring 2)
        # Actually let's place them optimally
        for k in range(4):
            angle = math.radians(30 + k * 90)
            x = 2 * d * math.cos(angle)
            y = 2 * d * math.sin(angle)
            centers.append((x, y))
        
        inner = []
        for x, y in centers:
            inner.append((x, y, 0.0))
        
        # Calculate required outer side length
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        side = outer_side_required_fast(inner, outer_center, outer_angle)
        
        return inner, outer_center, side, outer_angle

    # Get the analytic solution as baseline
    inner_seed, outer_center_seed, side_seed, angle_seed = analytic_solution()
    best_score = side_seed
    best_result = (inner_seed, outer_center_seed, side_seed, angle_seed)

    # --- Smooth objective function for optimization ---
    def objective(params):
        """
        Smooth objective: outer side length + separation penalty.
        params: 33-dimensional vector (x, y, angle_degrees) for each of 11 hexagons.
        """
        inner_params = []
        for i in range(N):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2]
            inner_params.append((x, y, a))

        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        outer_side = outer_side_required_fast(inner_params, outer_center, outer_angle)

        # Separation penalty using smooth min
        sep_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                sep = min_separation(vi, vj)
                # Penalize when separation < 0.01 (should be >= 0)
                if sep < 0.01:
                    sep_penalty += (0.01 - sep) ** 2

        return outer_side + 100.0 * sep_penalty

    def initial_guess_from_seed():
        """Generate initial positions from the analytic seed."""
        inner, _, _, _ = analytic_solution()
        params = np.zeros(3 * N)
        for i, (x, y, a) in enumerate(inner):
            params[3 * i] = x
            params[3 * i + 1] = y
            params[3 * i + 2] = a
        return params

    # --- Optimization using multiple strategies ---
    from scipy.optimize import minimize

    # Strategy 1: L-BFGS-B with tight tolerances
    if time.time() < deadline - 60:
        params0 = initial_guess_from_seed()
        try:
            result = minimize(
                objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 50000}
            )
            if time.time() < deadline - 50:
                best_params = result.x
                inner_positions = []
                for i in range(N):
                    x = best_params[3 * i]
                    y = best_params[3 * i + 1]
                    a = best_params[3 * i + 2]
                    inner_positions.append((x, y, a))

                outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), 0.0)

                if is_valid(inner_positions, (0.0, 0.0), outer_side, 0.0):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner_positions, (0.0, 0.0), outer_side, 0.0)
        except Exception:
            pass

    # Strategy 2: Nelder-Mead with adaptive simplex
    if time.time() < deadline - 40:
        params0 = initial_guess_from_seed()
        try:
            result = minimize(
                objective,
                params0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True, 'maxfev': 20000}
            )
            if time.time() < deadline - 30:
                best_params = result.x
                inner_positions = []
                for i in range(N):
                    x = best_params[3 * i]
                    y = best_params[3 * i + 1]
                    a = best_params[3 * i + 2]
                    inner_positions.append((x, y, a))

                outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), 0.0)

                if is_valid(inner_positions, (0.0, 0.0), outer_side, 0.0):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner_positions, (0.0, 0.0), outer_side, 0.0)
        except Exception:
            pass

    # Strategy 3: Differential Evolution for global exploration
    if time.time() < deadline - 30:
        try:
            bounds = [(-10, 10), (-10, 10), (-180, 180)] * N
            result = minimize(
                objective,
                initial_guess_from_seed(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            if time.time() < deadline - 20:
                best_params = result.x
                inner_positions = []
                for i in range(N):
                    x = best_params[3 * i]
                    y = best_params[3 * i + 1]
                    a = best_params[3 * i + 2]
                    inner_positions.append((x, y, a))

                outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), 0.0)

                if is_valid(inner_positions, (0.0, 0.0), outer_side, 0.0):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner_positions, (0.0, 0.0), outer_side, 0.0)
        except Exception:
            pass

    # Strategy 4: Final high-precision polish
    if time.time() < deadline - 10:
        inner_positions, outer_center, outer_side, outer_angle = best_result
        params0 = np.zeros(3 * N)
        for i, (x, y, a) in enumerate(inner_positions):
            params0[3 * i] = x
            params0[3 * i + 1] = y
            params0[3 * i + 2] = a
        
        try:
            result = minimize(
                objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 100000}
            )
            best_params = result.x
            inner_positions = []
            for i in range(N):
                x = best_params[3 * i]
                y = best_params[3 * i + 1]
                a = best_params[3 * i + 2]
                inner_positions.append((x, y, a))

            outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), 0.0)

            if is_valid(inner_positions, (0.0, 0.0), outer_side, 0.0):
                if outer_side < best_score:
                    best_score = outer_side
                    best_result = (inner_positions, (0.0, 0.0), outer_side, 0.0)
        except Exception:
            pass

    # Return the best result
    inner_positions, outer_center, outer_side, outer_angle = best_result
    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
# EVOLVE-BLOCK-END

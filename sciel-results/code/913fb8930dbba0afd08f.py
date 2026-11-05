"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

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

    def fast_containment(inner_params, outer_center, outer_side, outer_angle):
        """
        Fast check: for each inner hex, check all 6 vertices against outer hex.
        inner_params: list of (x, y, angle_deg)
        """
        cx, cy = outer_center
        outer_verts = hex_vertices(cx, cy, outer_side, outer_angle)
        for (ix, iy, iang) in inner_params:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            if not hex_in_hexagon(inner_verts, outer_verts):
                return False
        return True

    def outer_side_required_fast(inner_params, outer_center, outer_angle):
        """
        Compute minimum outer side length using exact max projection.
        Vectorized.
        """
        cx, cy = outer_center
        apothem_factor = math.sqrt(3) / 2.0
        
        # Edge normals
        normal_angles = np.radians(outer_angle + 30 + np.arange(6) * 60)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
        
        max_val = 0.0
        for (ix, iy, iang_deg) in inner_params:
            iang = math.radians(iang_deg)
            v_angles = iang + np.arange(6) * math.pi / 3.0
            v_offsets = np.column_stack([np.cos(v_angles), np.sin(v_angles)])
            v_pos = v_offsets + np.array([ix, iy])
            v_rel = v_pos - np.array([cx, cy])
            projs = v_rel @ normals.T
            m = np.max(projs)
            if m > max_val:
                max_val = m
        
        return max_val / apothem_factor

    def is_valid(inner_params, outer_center, outer_side, outer_angle):
        """Check if configuration is valid: no overlaps, all inside outer hex."""
        # Check containment
        if not fast_containment(inner_params, outer_center, outer_side, outer_angle):
            return False
        # Check non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                if polygon_overlap_area(vi, vj) > 1e-8:
                    return False
        return True

    # --- Structured lattice-based packing with 6-fold symmetry ---
    def generate_lattice_packing(ring1_scale, ring2_scale, ring3_scale, all_angles=0.0):
        """
        Generate a symmetric packing using hexagonal lattice coordinates.
        Centers are placed at:
        - Origin (1 hex)
        - Ring 1: 6 positions at distance ring1_scale * 2, angles 0,60,120,180,240,300
        - Ring 2: 6 positions at distance ring2_scale * 2, angles 30,90,150,210,270,330
        - Ring 3: 3 positions at distance ring3_scale * 4, angles 0,120,240
        Total: 1 + 6 + 6 + 3 = 16, but we only need 11.
        
        Actually let's use: 1 center + 6 ring1 + 4 ring2 = 11
        Ring 2 positions: angles 30, 90, 210, 270 (or other symmetric subsets)
        """
        centers = [(0.0, 0.0)]
        
        # Ring 1: 6 hexagons
        for k in range(6):
            angle = math.radians(k * 60)
            x = ring1_scale * 2.0 * math.cos(angle)
            y = ring1_scale * 2.0 * math.sin(angle)
            centers.append((x, y))
        
        # Ring 2: 4 hexagons at 90-degree intervals, offset by 30 degrees
        for k in range(4):
            angle = math.radians(30 + k * 90)
            x = ring2_scale * 2.0 * math.cos(angle)
            y = ring2_scale * 2.0 * math.sin(angle)
            centers.append((x, y))
        
        inner = [(x, y, all_angles) for x, y in centers]
        return inner

    def objective_lattice(params):
        """
        Objective for lattice parameters: [ring1_scale, ring2_scale, outer_angle]
        Returns the outer side length (to minimize).
        """
        r1, r2, outer_angle = params
        inner = generate_lattice_packing(r1, r2, 0.0)
        
        # Check for overlaps
        max_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                ov = polygon_overlap_area(vi, vj)
                if ov > max_overlap:
                    max_overlap = ov
        
        if max_overlap > 1e-10:
            # Penalize overlaps heavily
            return 100.0 + 1000.0 * max_overlap
        
        outer_side = outer_side_required_fast(inner, (0.0, 0.0), outer_angle)
        return outer_side

    # --- Try multiple symmetric configurations ---
    best_score = float('inf')
    best_result = None

    # Baseline: standard concentric ring packing
    inner_seed = generate_lattice_packing(1.0, 2.0)
    side_seed = outer_side_required_fast(inner_seed, (0.0, 0.0), 0.0)
    if is_valid(inner_seed, (0.0, 0.0), side_seed, 0.0):
        best_score = side_seed
        best_result = (inner_seed, (0.0, 0.0), side_seed, 0.0)

    # --- Optimize lattice parameters using scipy ---
    if time.time() < deadline - 30:
        try:
            from scipy.optimize import minimize
            
            # Multi-start optimization over lattice parameters
            initial_guesses = [
                [1.0, 2.0, 0.0],
                [1.05, 1.9, 0.0],
                [0.95, 2.1, 0.0],
                [1.0, 2.0, 15.0],
                [1.0, 2.0, 30.0],
                [1.1, 1.8, 0.0],
                [0.9, 2.2, 0.0],
            ]
            
            for guess in initial_guesses:
                if time.time() > deadline - 15:
                    break
                
                try:
                    result = minimize(
                        objective_lattice,
                        guess,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                    )
                    
                    if result.fun < best_score and result.fun < 100.0:
                        r1, r2, outer_angle = result.x
                        inner_opt = generate_lattice_packing(r1, r2, 0.0)
                        side_opt = outer_side_required_fast(inner_opt, (0.0, 0.0), outer_angle)
                        
                        if is_valid(inner_opt, (0.0, 0.0), side_opt, outer_angle):
                            best_score = side_opt
                            best_result = (inner_opt, (0.0, 0.0), side_opt, outer_angle)
                except Exception:
                    pass
        except Exception:
            pass

    # --- Try a different symmetric pattern: offset rings ---
    if time.time() < deadline - 25:
        def generate_offset_packing(r1, r2, offset_angle):
            """
            Alternative: ring 1 at angles 0,60,...,300
            Ring 2 at angles offset_angle, offset_angle+90, offset_angle+180, offset_angle+270
            """
            centers = [(0.0, 0.0)]
            
            for k in range(6):
                angle = math.radians(k * 60)
                x = r1 * 2.0 * math.cos(angle)
                y = r1 * 2.0 * math.sin(angle)
                centers.append((x, y))
            
            for k in range(4):
                angle = math.radians(offset_angle + k * 90)
                x = r2 * 2.0 * math.cos(angle)
                y = r2 * 2.0 * math.sin(angle)
                centers.append((x, y))
            
            inner = [(x, y, 0.0) for x, y in centers]
            return inner

        def objective_offset(params):
            r1, r2, offset_angle = params
            inner = generate_offset_packing(r1, r2, offset_angle)
            
            max_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                    vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                    ov = polygon_overlap_area(vi, vj)
                    if ov > max_overlap:
                        max_overlap = ov
            
            if max_overlap > 1e-10:
                return 100.0 + 1000.0 * max_overlap
            
            outer_side = outer_side_required_fast(inner, (0.0, 0.0), 0.0)
            return outer_side

        try:
            from scipy.optimize import minimize
            
            guesses = [
                [1.0, 2.0, 30.0],
                [1.0, 2.0, 0.0],
                [1.0, 2.0, 15.0],
                [1.05, 1.95, 30.0],
            ]
            
            for guess in guesses:
                if time.time() > deadline - 10:
                    break
                
                try:
                    result = minimize(
                        objective_offset,
                        guess,
                        method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                    )
                    
                    if result.fun < best_score and result.fun < 100.0:
                        r1, r2, offset_angle = result.x
                        inner_opt = generate_offset_packing(r1, r2, offset_angle)
                        side_opt = outer_side_required_fast(inner_opt, (0.0, 0.0), 0.0)
                        
                        if is_valid(inner_opt, (0.0, 0.0), side_opt, 0.0):
                            best_score = side_opt
                            best_result = (inner_opt, (0.0, 0.0), side_opt, 0.0)
                except Exception:
                    pass
        except Exception:
            pass

    # --- Fine polish: optimize individual positions from best lattice result ---
    if time.time() < deadline - 10 and best_result is not None:
        try:
            from scipy.optimize import minimize
            
            best_inner, _, best_side, _ = best_result
            
            # Convert to parameter vector
            params0 = np.zeros(3 * N)
            for i, (x, y, a) in enumerate(best_inner):
                params0[3 * i] = x
                params0[3 * i + 1] = y
                params0[3 * i + 2] = a * math.pi / 180.0

            def objective_free(params):
                inner = []
                for i in range(N):
                    x = params[3 * i]
                    y = params[3 * i + 1]
                    a = params[3 * i + 2] * 180.0 / math.pi
                    inner.append((x, y, a))
                
                max_overlap = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                        vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                        ov = polygon_overlap_area(vi, vj)
                        if ov > max_overlap:
                            max_overlap = ov
                
                if max_overlap > 1e-10:
                    return 100.0 + 1000.0 * max_overlap
                
                outer_side = outer_side_required_fast(inner, (0.0, 0.0), 0.0)
                return outer_side

            result = minimize(
                objective_free,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score and result.fun < 100.0:
                inner_opt = []
                for i in range(N):
                    x = result.x[3 * i]
                    y = result.x[3 * i + 1]
                    a = result.x[3 * i + 2] * 180.0 / math.pi
                    inner_opt.append((x, y, a))
                
                side_opt = outer_side_required_fast(inner_opt, (0.0, 0.0), 0.0)
                
                if is_valid(inner_opt, (0.0, 0.0), side_opt, 0.0):
                    best_score = side_opt
                    best_result = (inner_opt, (0.0, 0.0), side_opt, 0.0)
        except Exception:
            pass

    # Return the best result
    if best_result is None:
        # Fallback to seed
        inner_seed = generate_lattice_packing(1.0, 2.0)
        side_seed = outer_side_required_fast(inner_seed, (0.0, 0.0), 0.0)
        best_result = (inner_seed, (0.0, 0.0), side_seed, 0.0)

    inner_positions, outer_center, outer_side, outer_angle = best_result
    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
# EVOLVE-BLOCK-END


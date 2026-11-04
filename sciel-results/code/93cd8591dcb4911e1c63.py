"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

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
        """Fast check: for each inner hex, check all 6 vertices against outer hex."""
        cx, cy = outer_center
        outer_verts = hex_vertices(cx, cy, outer_side, outer_angle)
        for (ix, iy, iang) in inner_params:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            if not hex_in_hexagon(inner_verts, outer_verts):
                return False
        return True

    def outer_side_required_fast(inner_params, outer_center, outer_angle):
        """
        Compute minimum outer side length using support function approach.
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
        if not fast_containment(inner_params, outer_center, outer_side, outer_angle):
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
        Construct the concentric ring packing:
        - 1 center hex
        - 6 ring-1 hexagons at distance sqrt(3)
        - 4 ring-2 hexagons at distance 3
        All hexagons rotated by 30 degrees.
        """
        eps = 1e-9
        centers = []
        centers.append((0.0, 0.0))
        
        for k in range(6):
            angle = math.radians(k * 60)
            x = (math.sqrt(3) + eps) * math.cos(angle)
            y = (math.sqrt(3) + eps) * math.sin(angle)
            centers.append((x, y))
            
        for k in range(4):
            angle = math.radians(30 + k * 60)
            x = (3.0 + 2*eps) * math.cos(angle)
            y = (3.0 + 2*eps) * math.sin(angle)
            centers.append((x, y))
        
        inner = []
        for x, y in centers:
            inner.append((x, y, 30.0))
        
        max_x = (2 + eps) * math.sqrt(3)
        max_y = 4 + 2 * eps
        
        s_x = 2 * max_x / math.sqrt(3)
        s_y = max_y
        
        side = max(s_x, s_y) + 1e-9
        
        return inner, (0.0, 0.0), side, 30.0

    inner_seed, outer_center_seed, side_seed, angle_seed = analytic_solution()
    best_score = side_seed
    best_result = (inner_seed, outer_center_seed, side_seed, angle_seed)

    # --- Strategy: Rigid-body cluster rotation optimization ---
    # Treat the 11-hexagon cluster as a single rigid object.
    # The internal geometry is fixed (from the analytic seed).
    # We optimize:
    #   1. The rotation of the entire cluster (cluster_angle)
    #   2. The rotation of the outer hexagon (outer_angle)
    
    def get_rotated_inner(cluster_angle_deg):
        """Get rotated inner hexagon positions."""
        base_inner, _, _, _ = analytic_solution()
        ca = math.radians(cluster_angle_deg)
        cos_a = math.cos(ca)
        sin_a = math.sin(ca)
        
        rotated_inner = []
        for (x, y, a) in base_inner:
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            new_a = a + cluster_angle_deg
            rotated_inner.append((rx, ry, new_a))
        return rotated_inner

    def cluster_side_length(cluster_angle_deg, outer_angle_deg):
        """Compute the outer side length needed."""
        rotated_inner = get_rotated_inner(cluster_angle_deg)
        side = outer_side_required_fast(rotated_inner, (0.0, 0.0), outer_angle_deg)
        return side

    def cluster_side_length_smooth(cluster_angle_deg, outer_angle_deg, beta=50.0):
        """
        Smooth surrogate using log-sum-exp for the max over projections.
        This makes the objective differentiable and helps gradient-based methods.
        """
        base_inner, _, _, _ = analytic_solution()
        ca = math.radians(cluster_angle_deg)
        cos_a = math.cos(ca)
        sin_a = math.sin(ca)
        
        apothem_factor = math.sqrt(3) / 2.0
        outer_angle_rad = math.radians(outer_angle_deg)
        normal_angles = [outer_angle_rad + math.radians(30 + k * 60) for k in range(6)]
        normals = [(math.cos(a), math.sin(a)) for a in normal_angles]
        
        # Collect all projection values
        projections = []
        for (x, y, a) in base_inner:
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            inner_verts = hex_vertices(rx, ry, 1.0, a + cluster_angle_deg)
            for vx, vy in inner_verts:
                dx = vx
                dy = vy
                for nx, ny in normals:
                    proj = dx * nx + dy * ny
                    projections.append(proj)
        
        # Log-sum-exp for smooth max
        max_proj = max(projections)
        if max_proj < -700:
            return max_proj / apothem_factor
        exp_sum = sum(math.exp(p - max_proj) for p in projections)
        smooth_max = max_proj + math.log(exp_sum) / beta
        return smooth_max / apothem_factor

    def objective_angles(params):
        """params: [cluster_angle_deg, outer_angle_deg]"""
        return cluster_side_length(params[0], params[1])

    def objective_angles_smooth(params):
        """Smooth version for gradient-based optimization."""
        return cluster_side_length_smooth(params[0], params[1])

    # Multi-start grid search
    best_angle_pair = None
    best_side_val = best_score
    
    n_grid = 40
    for i in range(n_grid):
        for j in range(n_grid):
            ca = i * 60.0 / n_grid
            oa = j * 60.0 / n_grid
            s = cluster_side_length(ca, oa)
            if s < best_side_val:
                best_side_val = s
                best_angle_pair = (ca, oa)
    
    # Multi-start Nelder-Mead from diverse initial points
    if time.time() < deadline - 30:
        try:
            from scipy.optimize import minimize
            
            # Diverse initial points
            initial_points = []
            if best_angle_pair is not None:
                initial_points.append(np.array(best_angle_pair))
            initial_points.append(np.array([0.0, 0.0]))
            initial_points.append(np.array([15.0, 15.0]))
            initial_points.append(np.array([30.0, 0.0]))
            initial_points.append(np.array([0.0, 30.0]))
            initial_points.append(np.array([7.5, 22.5]))
            initial_points.append(np.array([22.5, 7.5]))
            
            for x0 in initial_points:
                if time.time() > deadline - 25:
                    break
                try:
                    res = minimize(
                        objective_angles,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                    )
                    s_opt = res.fun
                    if s_opt < best_side_val:
                        best_side_val = s_opt
                        best_angle_pair = (res.x[0], res.x[1])
                except Exception:
                    pass
        except Exception:
            pass
    
    # Coordinate descent polish with golden-section refinement
    if time.time() < deadline - 15:
        if best_angle_pair is not None:
            ca_cur, oa_cur = best_angle_pair
            golden = (1.0 + math.sqrt(5.0)) / 2.0
            inv_golden = 1.0 / golden
            inv_golden2 = 1.0 - inv_golden
            
            # Coordinate descent iterations
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Optimize cluster_angle with oa_cur fixed
                # Use golden section search in a window around current best
                window = max(0.5, 1.0 * math.exp(-0.02 * iteration))
                a = ca_cur - window
                b = ca_cur + window
                
                # Golden section search
                c = b - inv_golden * (b - a)
                d = a + inv_golden * (b - a)
                fc = cluster_side_length(c, oa_cur)
                fd = cluster_side_length(d, oa_cur)
                
                for _ in range(50):
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - inv_golden * (b - a)
                        fc = cluster_side_length(c, oa_cur)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + inv_golden * (b - a)
                        fd = cluster_side_length(d, oa_cur)
                
                ca_new = (a + b) / 2.0
                s_ca = cluster_side_length(ca_new, oa_cur)
                if s_ca < best_side_val:
                    ca_cur = ca_new
                    best_side_val = s_ca
                
                # Optimize outer_angle with ca_cur fixed
                window = max(0.5, 1.0 * math.exp(-0.02 * iteration))
                a = oa_cur - window
                b = oa_cur + window
                
                c = b - inv_golden * (b - a)
                d = a + inv_golden * (b - a)
                fc = cluster_side_length(ca_cur, c)
                fd = cluster_side_length(ca_cur, d)
                
                for _ in range(50):
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - inv_golden * (b - a)
                        fc = cluster_side_length(ca_cur, c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + inv_golden * (b - a)
                        fd = cluster_side_length(ca_cur, d)
                
                oa_new = (a + b) / 2.0
                s_oa = cluster_side_length(ca_cur, oa_new)
                if s_oa < best_side_val:
                    oa_cur = oa_new
                    best_side_val = s_oa
            
            # Final check
            if best_side_val < best_score:
                rotated_inner = get_rotated_inner(ca_cur)
                if is_valid(rotated_inner, (0.0, 0.0), best_side_val, oa_cur):
                    best_score = best_side_val
                    best_result = (rotated_inner, (0.0, 0.0), best_side_val, oa_cur)

    # --- Local relaxation: allow small perturbations of individual hexagons ---
    # Use smooth surrogate for the outer side length to help gradient methods.
    
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            
            inner_current, oc, side_current, oa_current = best_result
            
            def objective_local(params):
                inner_params = []
                for i in range(N):
                    x = params[3 * i]
                    y = params[3 * i + 1]
                    a = params[3 * i + 2] * 180.0 / math.pi
                    inner_params.append((x, y, a))
                
                # Use smooth surrogate for outer side
                base_inner, _, _, _ = analytic_solution()
                apothem_factor = math.sqrt(3) / 2.0
                outer_angle_rad = math.radians(oa_current)
                normal_angles = [outer_angle_rad + math.radians(30 + k * 60) for k in range(6)]
                normals = [(math.cos(a), math.sin(a)) for a in normal_angles]
                
                projections = []
                for (ix, iy, iang) in inner_params:
                    inner_verts = hex_vertices(ix, iy, 1.0, iang)
                    for vx, vy in inner_verts:
                        dx = vx
                        dy = vy
                        for nx, ny in normals:
                            proj = dx * nx + dy * ny
                            projections.append(proj)
                
                max_proj = max(projections)
                if max_proj < -700:
                    outer_side = max_proj / apothem_factor
                else:
                    beta = 30.0
                    exp_sum = sum(math.exp(p - max_proj) for p in projections)
                    outer_side = (max_proj + math.log(exp_sum) / beta) / apothem_factor
                
                # Overlap penalty (smooth)
                overlap_penalty = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                        vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                        ov = polygon_overlap_area(vi, vj)
                        if ov > 1e-12:
                            overlap_penalty += ov ** 2
                
                return outer_side + 500.0 * overlap_penalty

            params0 = np.zeros(3 * N)
            for i, (x, y, a) in enumerate(inner_current):
                params0[3 * i] = x
                params0[3 * i + 1] = y
                params0[3 * i + 2] = a * math.pi / 180.0

            result = minimize(
                objective_local,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if time.time() < deadline - 3:
                best_params = result.x
                inner_positions = []
                for i in range(N):
                    x = best_params[3 * i]
                    y = best_params[3 * i + 1]
                    a = best_params[3 * i + 2] * 180.0 / math.pi
                    inner_positions.append((x, y, a))

                outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), oa_current)

                if is_valid(inner_positions, (0.0, 0.0), outer_side, oa_current):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner_positions, (0.0, 0.0), outer_side, oa_current)
        except Exception:
            pass

    # Final high-precision polish: coordinate descent on the full 2D angle space
    # with very tight tolerances
    if time.time() < deadline - 2:
        try:
            inner_current, oc, side_current, oa_current = best_result
            # Extract cluster angle from the current best
            # Try small perturbations around current best
            base_inner, _, _, _ = analytic_solution()
            
            # Determine the cluster angle that produced best_result
            # We'll just do a fine grid around the current best angles
            if best_angle_pair is not None:
                ca_base, oa_base = best_angle_pair
                for dca in np.linspace(-0.1, 0.1, 5):
                    for doa in np.linspace(-0.1, 0.1, 5):
                        ca_try = ca_base + dca
                        oa_try = oa_base + doa
                        s_try = cluster_side_length(ca_try, oa_try)
                        if s_try < best_score - 1e-10:
                            rotated_inner = get_rotated_inner(ca_try)
                            if is_valid(rotated_inner, (0.0, 0.0), s_try, oa_try):
                                best_score = s_try
                                best_result = (rotated_inner, (0.0, 0.0), s_try, oa_try)
        except Exception:
            pass
    
    # Return the best result
    inner_positions, outer_center, outer_side, outer_angle = best_result
    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
# EVOLVE-BLOCK-END

import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    def get_hex_vertices(cx, cy, angle_deg):
        angles = [math.radians(angle_deg + 60*k) for k in range(6)]
        return [(cx + math.cos(a), cy + math.sin(a)) for a in angles]

    def point_in_polygon(pt, poly):
        x, y = pt
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if (y > p1y) != (y > p2y):
                xint = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
                if x < xint:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def polygons_disjoint(v1, v2):
        for v in v1:
            if point_in_polygon(v, v2):
                return False
        for v in v2:
            if point_in_polygon(v, v1):
                return False
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1)%n2]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def point_polygon_distance(pt, poly):
        min_dist = float('inf')
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                d = math.hypot(pt[0]-p1[0], pt[1]-p1[1])
            else:
                t = ((pt[0]-p1[0])*dx + (pt[1]-p1[1])*dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                proj_x = p1[0] + t*dx
                proj_y = p1[1] + t*dy
                d = math.hypot(pt[0]-proj_x, pt[1]-proj_y)
            if d < min_dist:
                min_dist = d
        return min_dist

    def get_min_L(vertices_list):
        max_proj = 0.0
        normals = [math.radians(30 + 60*k) for k in range(6)]
        for v in vertices_list:
            for ang in normals:
                proj = v[0]*math.cos(ang) + v[1]*math.sin(ang)
                if proj > max_proj:
                    max_proj = proj
        return max_proj / (sqrt3 / 2.0)

    # Minkowski-sum based smooth separation
    # For two convex polygons P (at pos_i, angle_i) and Q (at pos_j, angle_j),
    # the separation is: min over directions u of (h_P(u) + h_Q(-u) - d·u)
    # where h is support function and d is relative position.
    # For unit regular hexagons, the Minkowski difference is a 12-gon.
    # We use a soft-min over the 12 edges of the Minkowski difference.

    # Precompute the 12 edge normals of the Minkowski difference of two unit hexagons
    # A unit hexagon has support function h(u) = max_k cos(u - k*pi/3) for u in [0, 2pi)
    # The Minkowski sum H+H has support 2*h, and H+(-H) has support h(u)+h(-u)
    # For regular hexagons, h(u) = cos(u mod pi/3 - pi/6) ... let's compute properly

    # Support function of unit regular hexagon (flat-top, vertices at angles 0,60,...,300):
    # h(u) = max_k cos(u - k*pi/3), k=0..5
    # For the Minkowski difference, we need h_H(u) + h_H(-u) for each direction u
    # The critical directions are the 12 normals of the Minkowski difference polygon

    # The Minkowski difference of two unit hexagons (with relative rotation) 
    # is a 12-sided polygon. Its support function in direction u is:
    # s(u) = h_H1(u) + h_H2(-u)
    # where H2 is the second hexagon rotated by (theta_j - theta_i)

    def support_hex(u, angle_deg):
        """Support function of unit regular hexagon at angle u (radians), rotated by angle_deg."""
        a = math.radians(angle_deg)
        vals = [math.cos(u - a - k * math.pi / 3.0) for k in range(6)]
        return max(vals)

    def minkowski_separation_soft(xi, yi, thi, xj, yj, thj, beta=8.0):
        """
        Compute soft separation between two unit hexagons using Minkowski sum approach.
        Returns a smooth approximation of max(0, overlap_depth).
        The separation in direction u is: d·u - s(u) where s(u)=h1(u)+h2(-u)
        Overlap exists if min_u (d·u - s(u)) < 0
        We use soft-min: -log-sum-exp(-beta*(d·u_k - s(u_k)))
        """
        dx = xj - xi
        dy = yj - yi
        dvec = (dx, dy)

        # 12 critical directions for Minkowski difference
        # These are the normals of the 12 edges of the Minkowski difference polygon
        # For two hexagons with relative angle delta, the 12 directions are:
        # 6 from hex1 normals + 6 from hex2 normals (rotated)
        # Hex1 normals (outward): angles 0, 60, 120, 180, 240, 300 degrees + 30 offset
        # Actually for flat-top hexagon with vertices at 0,60,...,300:
        # Edge normals are at 30, 90, 150, 210, 270, 330 degrees

        th_diff = thj - thi

        # 12 directions: 6 from each hexagon's edge normals
        dirs = []
        # Hex1 edge normals (relative to hex1 frame): 30, 90, 150, 210, 270, 330 deg
        for k in range(6):
            base_angle = math.radians(30 + 60 * k)
            # Rotate by hex1's angle
            dirs.append(base_angle + math.radians(thi))
        # Hex2 edge normals (relative to hex2 frame), but we need -normal for Minkowski diff
        for k in range(6):
            base_angle = math.radians(30 + 60 * k)
            # The -normal of hex2 edge, rotated by hex2's angle
            dirs.append(base_angle + math.pi + math.radians(thj))

        # Compute d·u - s(u) for each direction
        vals = []
        for u in dirs:
            cos_u = math.cos(u)
            sin_u = math.sin(u)
            d_dot_u = dx * cos_u + dy * sin_u
            # Support of hex1 in direction u
            s1 = support_hex(u, thi)
            # Support of hex2 in direction -u
            s2 = support_hex(-u, thj)
            s = s1 + s2
            vals.append(d_dot_u - s)

        # Soft min: approximates min(vals)
        # We want penalty = max(0, -soft_min)
        # soft_min = -1/beta * log(sum(exp(-beta*vals)))
        max_val = max(vals)
        # For numerical stability
        exp_vals = [math.exp(-beta * (v - max_val)) for v in vals]
        sum_exp = sum(exp_vals)
        soft_min = max_val - math.log(sum_exp) / beta

        # Penalty: smooth max(0, -soft_min)
        if soft_min < 0:
            # Overlap: quadratic penalty
            return 10.0 * soft_min * soft_min
        else:
            # Small positive: mild repulsion to maintain separation
            if soft_min < 0.05:
                return 0.5 * (0.05 - soft_min) ** 2
            return 0.0

    def objective(params, penalty_weight=1.0):
        if time.time() > deadline:
            return 1e10
        all_vertices = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            th = params[3*i+2]
            verts = get_hex_vertices(x, y, th)
            all_vertices.extend(verts)
        L_req = get_min_L(all_vertices)
        penalty = 0.0
        for i in range(N):
            xi, yi, thi = params[3*i], params[3*i+1], params[3*i+2]
            for j in range(i+1, N):
                xj, yj, thj = params[3*j], params[3*j+1], params[3*j+2]
                penalty += minkowski_separation_soft(xi, yi, thi, xj, yj, thj)
        return L_req + penalty_weight * penalty

    def objective_strict(params):
        """Objective with stricter penalty for use in final polish."""
        return objective(params, penalty_weight=5.0)

    # Generate diverse initial configurations
    initializations = []
    
    # 1. Hexagonal grid pattern (topology-first: known non-overlapping layout)
    grid_centers = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) <= 3:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                grid_centers.append((x, y))
    
    dists = [(math.hypot(x, y), i) for i, (x, y) in enumerate(grid_centers)]
    dists.sort()
    init_grid_idx = [d[1] for d in dists[:12]]
    
    params_grid = []
    for i in init_grid_idx:
        x, y = grid_centers[i]
        params_grid.extend([x, y, 0.0])
    initializations.append(params_grid)
    
    # 2. Slightly perturbed grid (different topology seeds)
    np.random.seed(42)
    params_grid2 = []
    for i in init_grid_idx:
        x, y = grid_centers[i]
        x += np.random.uniform(-0.2, 0.2)
        y += np.random.uniform(-0.2, 0.2)
        params_grid2.extend([x, y, 0.0])
    initializations.append(params_grid2)

    # 3. Ring configuration
    params_ring = []
    for i in range(12):
        angle = 2 * math.pi * i / 12
        r = 2.5
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        th = angle * 180 / math.pi
        params_ring.extend([x, y, th])
    initializations.append(params_ring)
    
    # 4. Two rings (inner 6 + outer 6)
    params_two_rings = []
    for i in range(6):
        angle = 2 * math.pi * i / 6
        r = 1.5
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        params_two_rings.extend([x, y, 0.0])
    for i in range(6):
        angle = 2 * math.pi * i / 6 + math.pi/6
        r = 3.5
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        params_two_rings.extend([x, y, 0.0])
    initializations.append(params_two_rings)

    # 5. Staggered rows
    params_staggered = []
    row_y = [-2.0, 0.0, 2.0]
    for row_idx, y in enumerate(row_y):
        for col in range(4):
            x = -2.25 + col * 1.5
            if row_idx % 2 == 1:
                x += 0.75
            params_staggered.extend([x, y, 0.0])
    initializations.append(params_staggered)
    
    best_params = None
    best_score = float('inf')

    # Phase 1: Global exploration with Nelder-Mead on multiple starts
    for init_params in initializations:
        if time.time() > deadline - 50.0:
            break
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Gradient-based polish with L-BFGS-B (smooth objective enables this)
    if best_params is not None and time.time() < deadline - 25.0:
        try:
            res = minimize(objective_strict, best_params, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Fine-tuning with Nelder-Mead on tighter tolerances
    if best_params is not None and time.time() < deadline - 10.0:
        try:
            res = minimize(objective_strict, best_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 4: Final L-BFGS-B polish with very tight tolerance
    if best_params is not None and time.time() < deadline - 3.0:
        try:
            res = minimize(objective_strict, best_params, method='L-BFGS-B', 
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 20000})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # If no valid solution found, fallback to grid
    if best_params is None:
        best_params = params_grid
        
    # Evaluate best_params
    positions = []
    all_vertices = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        th = best_params[3*i+2]
        positions.append([x, y, th])
        verts = get_hex_vertices(x, y, th)
        all_vertices.extend(verts)
        
    L_final = get_min_L(all_vertices)
    
    # Verify disjointness with small tolerance
    is_valid = True
    verts_list = [get_hex_vertices(pos[0], pos[1], pos[2]) for pos in positions]
    for i in range(N):
        for j in range(i+1, N):
            if not polygons_disjoint(verts_list[i], verts_list[j]):
                is_valid = False
                break
        if not is_valid:
            break
        
    if not is_valid:
        # Try to fix by scaling up slightly
        for scale_factor in [1.0001, 1.001, 1.01, 1.05, 1.1]:
            test_positions = []
            test_vertices = []
            for i in range(N):
                x = best_params[3*i] * scale_factor
                y = best_params[3*i+1] * scale_factor
                th = best_params[3*i+2]
                test_positions.append([x, y, th])
                verts = get_hex_vertices(x, y, th)
                test_vertices.extend(verts)
            
            tv_list = [get_hex_vertices(p[0], p[1], p[2]) for p in test_positions]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not polygons_disjoint(tv_list[i], tv_list[j]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                L_final = get_min_L(test_vertices)
                return test_positions, (0.0, 0.0), L_final, 0.0
        
        # Fallback to known valid grid solution
        inner = []
        SCALE = 1.000001
        for i in init_grid_idx:
            x, y = grid_centers[i]
            inner.append([x * SCALE, y * SCALE, 0.0])
            
        fb_vertices = []
        for p in inner:
            v = get_hex_vertices(p[0], p[1], p[2])
            fb_vertices.extend(v)
        L_final = get_min_L(fb_vertices)
        
        return inner, (0.0, 0.0), L_final, 0.0

    return positions, (0.0, 0.0), L_final, 0.0

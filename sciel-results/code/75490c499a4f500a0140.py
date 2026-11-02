"""Packing 12 unit hexagons in a regular hexagon using QCQP-style linear separation.
Exploits the fact that regular hexagons are zonotopes: the Minkowski difference
of two unit regular hexagons (with relative rotation θ) is a centrally-symmetric
polygon whose support function gives exact linear separation constraints.
Optimization: interior-point style with penalty, plus local refinement.
"""
import numpy as np
import time
import math

N = 12
SQRT3 = math.sqrt(3.0)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hex_verts(cx, cy, side, angle_deg):
        """6 vertices of regular hexagon."""
        a = math.radians(angle_deg)
        angs = a + np.radians(np.arange(0, 360, 60))
        return np.column_stack([cx + side * np.cos(angs), cy + side * np.sin(angs)])

    def minkowski_diff_vertices(theta_deg):
        """
        Vertices of the Minkowski difference H(0) - H(theta) where H is unit regular hexagon.
        This is a centrally symmetric polygon. Two hexagons are disjoint iff
        the vector between centers is outside this polygon (strictly).
        Actually: H1 and H2 disjoint iff (c1-c2) is NOT in H1 - H2 (interior).
        The Minkowski difference of two convex polygons is a convex polygon.
        For two unit regular hexagons with relative angle theta, the Minkowski
        difference has at most 12 vertices.
        """
        # H1 vertices
        v1 = hex_verts(0, 0, 1.0, 0.0)
        # H2 vertices rotated by theta
        v2 = hex_verts(0, 0, 1.0, theta_deg)
        # Minkowski difference: {a - b : a in H1, b in H2} = H1 + (-H2)
        # -H2 is H2 rotated 180 degrees
        neg_v2 = -v2
        # Minkowski sum of two convex polygons: compute all edge sums
        # For two convex polygons, the Minkowski sum can be computed by
        # merging edge vectors sorted by angle.
        def minkowski_sum(p1, p2):
            """Minkowski sum of two convex polygons (arrays of vertices in CCW order)."""
            n1, n2 = len(p1), len(p2)
            # Compute edge vectors
            e1 = np.roll(p1, -1, axis=0) - p1
            e2 = np.roll(p2, -1, axis=0) - p2
            # Sort by angle
            a1 = np.arctan2(e1[:, 1], e1[:, 0])
            a2 = np.arctan2(e2[:, 1], e2[:, 0])
            idx1 = np.argsort(a1)
            idx2 = np.argsort(a2)
            e1s = e1[idx1]
            e2s = e2[idx2]
            a1s = a1[idx1]
            a2s = a2[idx2]
            # Merge
            merged = []
            i = j = 0
            while i < n1 or j < n2:
                if i >= n1:
                    merged.append(e2s[j])
                    j += 1
                elif j >= n2:
                    merged.append(e1s[i])
                    i += 1
                else:
                    if a1s[i] <= a2s[j]:
                        merged.append(e1s[i])
                        i += 1
                    else:
                        merged.append(e2s[j])
                        j += 1
            # Start from p1[0] + p2[0]
            result = [p1[0] + p2[0]]
            for e in merged:
                result.append(result[-1] + e)
            return np.array(result[:-1])  # remove last (closes the polygon)

        md = minkowski_sum(v1, neg_v2)
        return md

    def support_function_constraints(md_verts):
        """
        Given Minkowski difference vertices, return the half-plane constraints.
        Two hexagons with centers c1, c2 are disjoint iff
        (c1 - c2) is outside the Minkowski difference polygon.
        Equivalently, for each edge of the MD polygon, the dot product of
        (c1-c2) with the outward normal must be >= the support value.
        We want: for all edges, (c1-c2)·n_i >= h_i (outward)
        i.e., (c2-c1)·n_i <= -h_i
        """
        n = len(md_verts)
        constraints = []  # (normal, offset) such that d·normal <= offset means separation
        for i in range(n):
            p1 = md_verts[i]
            p2 = md_verts[(i + 1) % n]
            edge = p2 - p1
            # Outward normal (for CCW polygon, right perpendicular is outward)
            normal = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            # Support value: dot of vertex with normal
            offset = np.dot(p1, normal)
            # For separation: (c1 - c2)·normal > offset  =>  (c2 - c1)·normal < -offset
            # In our formulation: d = c2 - c1, we need d·normal <= -offset for all edges
            # But we want a single constraint that captures "d is outside MD"
            # Actually: d is inside MD iff d·n_i <= h_i for all i (where h_i = max dot)
            # d is outside iff exists i: d·n_i > h_i
            # For a smooth penalty: sum over i of max(0, d·n_i - h_i)^2
            constraints.append((normal, offset))
        return constraints

    # Precompute Minkowski differences for various relative angles
    # Since angles can vary, we compute on the fly but cache
    md_cache = {}

    def get_md_constraints(theta_deg):
        """Get separation constraints for relative angle theta_deg."""
        # Normalize angle to [0, 60)
        theta = theta_deg % 60.0
        # Round to 0.5 degrees for caching
        key = round(theta * 2) / 2
        if key not in md_cache:
            md = minkowski_diff_vertices(key)
            md_cache[key] = support_function_constraints(md)
        return md_cache[key]

    def separation_penalty(c1, a1, c2, a2):
        """
        Compute smooth separation penalty between two hexagons.
        c1, c2: center positions (2-vectors)
        a1, a2: angles in degrees
        Returns 0 if separated, positive if overlapping.
        """
        d = c2 - c1  # vector from hex1 to hex2
        theta = a2 - a1
        constraints = get_md_constraints(theta)
        penalty = 0.0
        for normal, offset in constraints:
            val = np.dot(d, normal) - offset
            if val > 0:
                penalty += val * val
        return penalty

    def containment_penalty(verts, outer_cx, outer_cy, outer_side, outer_angle):
        """
        Penalty for vertices being outside the outer hexagon.
        Uses support function of outer hexagon.
        """
        # Outer hexagon: center (outer_cx, outer_cy), side length outer_side, rotation outer_angle
        # A point p is inside iff for each of 6 face normals n_i:
        # (p - center)·n_i <= apothem
        # Penalty: sum of max(0, (p-center)·n_i - apothem)^2
        apothem = outer_side * SQRT3 / 2
        base_angle = math.radians(outer_angle)
        # Face normals are at angles: base_angle + 30 + k*60
        penalty = 0.0
        for k in range(6):
            na = base_angle + math.radians(30 + k * 60)
            nx, ny = math.cos(na), math.sin(na)
            for vx, vy in verts:
                dist = (vx - outer_cx) * nx + (vy - outer_cy) * ny
                if dist > apothem:
                    penalty += (dist - apothem) ** 2
        return penalty

    def total_penalty(params):
        """
        Full penalty function.
        params: [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12, ox,oy,os,oa]
        """
        n = N
        penalty = 0.0
        outer_cx = params[3*n]
        outer_cy = params[3*n+1]
        outer_side = max(params[3*n+2], 1.0)
        outer_angle = params[3*n+3]

        # Containment
        for i in range(n):
            cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
            verts = hex_verts(cx, cy, 1.0, ang)
            penalty += containment_penalty(verts, outer_cx, outer_cy, outer_side, outer_angle)

        # Separation
        for i in range(n):
            for j in range(i+1, n):
                c1 = np.array([params[3*i], params[3*i+1]])
                c2 = np.array([params[3*j], params[3*j+1]])
                a1 = params[3*i+2]
                a2 = params[3*j+2]
                # Quick reject
                dist = np.linalg.norm(c2 - c1)
                if dist > 4.0:
                    continue
                penalty += separation_penalty(c1, a1, c2, a2)

        return outer_side + penalty * 50.0

    def hex_verts_fast(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        # Unit hexagon vertices at angles 0,60,...,300
        ux = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
        uy = np.array([0.0, SQRT3/2, SQRT3/2, 0.0, -SQRT3/2, -SQRT3/2])
        # Rotate
        rx = side * (ux * cos_a - uy * sin_a)
        ry = side * (ux * sin_a + uy * cos_a)
        return np.column_stack([cx + rx, cy + ry])

    def separation_penalty_fast(c1x, c1y, c1a, c2x, c2y, c2a):
        """Faster separation using precomputed MD for common angles."""
        dx = c2x - c1x
        dy = c2y - c1y
        theta = (c2a - c1a) % 60.0
        # Quantize theta
        theta_q = round(theta * 4) / 4.0
        if theta_q not in md_cache:
            md = minkowski_diff_vertices(theta_q)
            md_cache[theta_q] = support_function_constraints(md)
        constraints = md_cache[theta_q]
        penalty = 0.0
        for normal, offset in constraints:
            val = dx * normal[0] + dy * normal[1] - offset
            if val > 0:
                penalty += val * val
        return penalty

    def total_penalty_fast(params):
        n = N
        penalty = 0.0
        outer_cx = params[3*n]
        outer_cy = params[3*n+1]
        outer_side = max(params[3*n+2], 1.0)
        outer_angle = params[3*n+3]
        apothem = outer_side * SQRT3 / 2
        base_angle = math.radians(outer_angle)
        nx_arr = np.array([math.cos(base_angle + math.radians(30 + k*60)) for k in range(6)])
        ny_arr = np.array([math.sin(base_angle + math.radians(30 + k*60)) for k in range(6)])

        for i in range(n):
            cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
            a_rad = math.radians(ang)
            cos_a, sin_a = math.cos(a_rad), math.sin(a_rad)
            ux = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
            uy = np.array([0.0, SQRT3/2, SQRT3/2, 0.0, -SQRT3/2, -SQRT3/2])
            rx = ux * cos_a - uy * sin_a
            ry = ux * sin_a + uy * cos_a
            vx = cx + rx
            vy = cy + ry
            for k in range(6):
                dist = (vx - outer_cx) * nx_arr[k] + (vy - outer_cy) * ny_arr[k]
                if dist > apothem:
                    penalty += (dist - apothem) ** 2

        for i in range(n):
            for j in range(i+1, n):
                dx = params[3*j] - params[3*i]
                dy = params[3*j+1] - params[3*i+1]
                if dx*dx + dy*dy > 16.0:
                    continue
                penalty += separation_penalty_fast(params[3*i], params[3*i+1], params[3*i+2],
                                                    params[3*j], params[3*j+1], params[3*j+2])

        return outer_side + penalty * 50.0

    def is_valid(inner, oc, os, oa):
        """Check validity."""
        # Containment
        for i in range(N):
            cx, cy, ang = inner[i]
            verts = hex_verts(cx, cy, 1.0, ang)
            for vx, vy in verts:
                if not point_in_hex(vx, vy, oc[0], oc[1], os, oa):
                    return False
        # Separation
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(inner[i], inner[j]):
                    return False
        return True

    def point_in_hex(px, py, cx, cy, side, angle_deg):
        dx, dy = px - cx, py - cy
        a = math.radians(angle_deg)
        ca, sa = math.cos(-a), math.sin(-a)
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        apothem = side * SQRT3 / 2
        for k in range(6):
            na = math.radians(30 + k * 60)
            d = lx * math.cos(na) + ly * math.sin(na)
            if d > apothem + 1e-9:
                return False
        return True

    def hexagons_disjoint(h1, h2):
        v1 = hex_verts(h1[0], h1[1], 1.0, h1[2])
        v2 = hex_verts(h2[0], h2[1], 1.0, h2[2])
        # SAT
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                    return True
        return False

    def initial_config_a():
        """Hexagonal grid 3x4."""
        positions = []
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * 3.0
                y = (row - 1) * 2.598
                if row % 2 == 1:
                    x += 1.5
                positions.append([x, y, 0.0])
        return positions

    def initial_config_b():
        """Central + ring + outer."""
        positions = [[0.0, 0.0, 0.0]]
        for k in range(6):
            a = k * math.pi / 3
            positions.append([2.0*math.cos(a), 2.0*math.sin(a), 0.0])
        for k in range(5):
            a = k * math.pi / 3 + math.pi / 6
            positions.append([3.0*math.cos(a), 3.0*math.sin(a), 0.0])
        return positions[:12]

    def make_params(positions, oc, os, oa):
        params = []
        for p in positions:
            params.extend([p[0], p[1], p[2]])
        params.extend([oc[0], oc[1], os, oa])
        return np.array(params, dtype=float)

    def parse_params(params):
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        oc = (params[3*N], params[3*N+1])
        os = params[3*N+2]
        oa = params[3*N+3]
        return inner, oc, os, oa

    def optimize(params, max_time):
        """
        Optimize using a combination of:
        1. Gradient-free local search (Nelder-Mead style)
        2. Random restarts with perturbation
        """
        from scipy.optimize import minimize

        current = params.copy()
        current_val = total_penalty_fast(current)
        best = current.copy()
        best_val = current_val

        # Nelder-Mead refinement
        try:
            res = minimize(total_penalty_fast, current, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 0.01, 'fatol': 0.01, 'adaptive': True})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass

        # Local perturbation search
        rng = np.random.default_rng(12345)
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            # Perturb best
            perturbed = best + rng.normal(0, 0.3, size=len(best))
            perturbed[3*N+2] = max(perturbed[3*N+2], 1.0)
            val = total_penalty_fast(perturbed)
            if val < best_val:
                best = perturbed
                best_val = val
                # Further refine
                try:
                    res = minimize(total_penalty_fast, best, method='Nelder-Mead',
                                  options={'maxiter': 100, 'xatol': 0.01, 'fatol': 0.01, 'adaptive': True})
                    if res.fun < best_val:
                        best = res.x.copy()
                        best_val = res.fun
                except Exception:
                    pass

        return best, best_val

    # Main
    best_params = None
    best_score = float('inf')

    # Try multiple initial configurations
    inits = [initial_config_a(), initial_config_b()]

    # Also try with different outer angles
    for oa in [0.0, 30.0, 15.0]:
        for pos_list in inits:
            if time.time() > deadline - 25:
                break
            positions = [p[:] for p in pos_list]
            for p in positions:
                p[2] = oa  # align inner hexagons with outer
            # Estimate outer side
            # Compute bounding of all vertices
            all_vx, all_vy = [], []
            for cx, cy, ang in positions:
                verts = hex_verts(cx, cy, 1.0, ang)
                all_vx.extend(verts[:, 0])
                all_vy.extend(verts[:, 1])
            cx_mean = (max(all_vx) + min(all_vx)) / 2
            cy_mean = (max(all_vy) + min(all_vy)) / 2
            # Estimate side from extent
            extent = max(max(all_vx) - min(all_vx), max(all_vy) - min(all_vy))
            est_side = extent / (SQRT3) + 0.5
            est_side = max(est_side, 3.0)

            params = make_params(positions, [cx_mean, cy_mean], est_side, oa)
            result_params, result_score = optimize(params, 20.0)
            if result_score < best_score:
                best_score = result_score
                best_params = result_params.copy()

    # Final optimization pass on best
    if best_params is not None and time.time() < deadline - 15:
        result_params, result_score = optimize(best_params, 30.0)
        if result_score < best_score:
            best_score = result_score
            best_params = result_params.copy()

    # Fallback
    if best_params is None:
        positions = initial_config_a()
        best_params = make_params(positions, [0.0, 0.0], 8.0, 0.0)

    inner, oc, os, oa = parse_params(best_params)

    # Validate and fix
    if not is_valid(inner, oc, os, oa):
        # Expand outer
        for _ in range(100):
            os += 0.1
            if is_valid(inner, oc, os, oa):
                break
        # If still invalid due to overlap, spread positions
        if not is_valid(inner, oc, os, oa):
            for i in range(N):
                inner[i] = (inner[i][0] * 1.3, inner[i][1] * 1.3, inner[i][2])
            for _ in range(100):
                os += 0.2
                if is_valid(inner, oc, os, oa):
                    break

    return inner, oc, os, oa

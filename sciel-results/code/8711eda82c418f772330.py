"""Hexagon packing: 11 unit hexagons inside a regular hexagon.
Structural change: use LP (scipy.linprog) for exact outer hexagon sizing/centering,
and differential_evolution + L-BFGS-B for inner position optimization with a
smooth surrogate. This replaces the manual Nelder-Mead and combinatorial scan."""
import math
import time
import itertools
import random
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution

N = 11


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True


def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True


def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle
    that contains all hexagons in `hexes`."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / math.sqrt(3)


def _lp_outer_side_and_center(hexes, oangle=0.0):
    """Use LP to find optimal outer hexagon center and side length for given inner hexes."""
    # Variables: [cx, cy, t] where t is the apothem of outer hex
    # Minimize t subject to: for each inner vertex v, for each of 6 outer normals n_k:
    #   (vx - cx)*nx_k + (vy - cy)*ny_k <= t
    c = [0.0, 0.0, 1.0]
    
    A_ub = []
    b_ub = []
    
    for cx_h, cy_h, s_h, ang_h in hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                theta = math.radians(oangle + 30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                # (vx - cx)*nx + (vy - cy)*ny <= t
                # => -nx*cx - ny*cy - t <= -vx*nx - vy*ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-vx * nx - vy * ny)
    
    # t >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    bounds = [(None, None), (None, None), (0.0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            center = (res.x[0], res.x[1])
            side = res.x[2] * 2.0 / math.sqrt(3)
            return center, side
    except Exception:
        pass
    
    # Fallback: centroid
    cx_mean = sum(h[0] for h in hexes) / len(hexes)
    cy_mean = sum(h[1] for h in hexes) / len(hexes)
    side = _min_outer_side(hexes, cx_mean, cy_mean, oangle)
    return (cx_mean, cy_mean), side


def _smooth_outer_side_lp(hexes, oangle, beta=50.0):
    """Smooth approximation using log-sum-exp over projections, with LP-optimized center."""
    # First get approximate center from LP
    center, side = _lp_outer_side_and_center(hexes, oangle)
    ox, oy = center
    
    # Now compute smooth max projection
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    projs = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                projs.append(proj)
    
    if not projs:
        return side
    
    m = max(projs)
    exp_sum = sum(math.exp(p - m) for p in projs)
    soft_max = m + math.log(exp_sum) / beta
    
    return soft_max * 2.0 / math.sqrt(3)


def _separation_violation(hexes):
    """Compute total pairwise separation violation (positive means overlap)."""
    n = len(hexes)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, s1, a1 = hexes[i]
            cx2, cy2, s2, a2 = hexes[j]
            dx = cx2 - cx1
            dy = cy2 - cy1
            dist = math.sqrt(dx * dx + dy * dy)
            min_dist = math.sqrt(3)
            if dist < min_dist:
                total += (min_dist - dist) ** 2
    return total


def _objective(params, beta, oangle=0.0):
    """Combined objective: smooth outer side + separation penalty."""
    positions = []
    for i in range(11):
        x = params[2 * i]
        y = params[2 * i + 1]
        positions.append((x, y, 1.0, 0.0))
    
    outer_side = _smooth_outer_side_lp(positions, oangle, beta)
    sep_viol = _separation_violation(positions)
    
    return outer_side + 10.0 * sep_viol


def _objective_lp(params, oangle=0.0):
    """Exact objective using LP for outer side."""
    positions = []
    for i in range(11):
        x = params[2 * i]
        y = params[2 * i + 1]
        positions.append((x, y, 1.0, 0.0))
    
    center, side = _lp_outer_side_and_center(positions, oangle)
    sep_viol = _separation_violation(positions)
    
    return side + 10.0 * sep_viol


def _exact_side_for_positions(params, oangle=0.0):
    """Get exact side length for given position params."""
    positions = []
    for i in range(11):
        x = params[2 * i]
        y = params[2 * i + 1]
        positions.append((x, y, 1.0, 0.0))
    center, side = _lp_outer_side_and_center(positions, oangle)
    return center, side


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Axial coordinate to Cartesian for flat-top hexagons
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Generate initial hexagonal lattice configuration
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)

    # Ring 2 positions
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]

    best_side = float('inf')
    best_params = None
    best_center = None

    # Phase 1: Coarse scan over ring2 combinations and outer orientations
    all_combos = list(itertools.combinations(range(len(ring2)), 4))
    oangles = [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0,
               10.0, -10.0, 20.0, -20.0, 50.0, -50.0, 80.0, -80.0, 5.0, -5.0]

    for combo in all_combos:
        if time.time() > deadline - 60.0:
            break

        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]

        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]

        for oang in oangles:
            center, side = _lp_outer_side_and_center(inner_hexes, oang)
            if side < best_side:
                best_side = side
                best_center = center
                best_params = []
                for x, y in full_xy:
                    best_params.extend([x, y])

    if best_params is None:
        # Fallback
        full_xy = [axial_to_xy(q, r) for q, r in pos_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        best_center, best_side = _lp_outer_side_and_center(inner_hexes, 0.0)
        best_params = []
        for x, y in full_xy:
            best_params.extend([x, y])

    # Phase 2: Differential evolution for global optimization of positions
    if time.time() < deadline - 40.0:
        # Define bounds for each position
        # Use a reasonable search box around the initial configuration
        cx_init = sum(best_params[2*i] for i in range(11)) / 11.0
        cy_init = sum(best_params[2*i+1] for i in range(11)) / 11.0
        search_radius = 5.0
        
        bounds = []
        for i in range(11):
            bounds.append((cx_init - search_radius, cx_init + search_radius))
            bounds.append((cy_init - search_radius, cy_init + search_radius))
        
        # Objective function for DE
        def de_objective(params):
            return _objective_lp(params, oangle=0.0)
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            
            if de_result.fun < best_side:
                # Verify disjointness
                positions = []
                for i in range(11):
                    positions.append((de_result.x[2*i], de_result.x[2*i+1], 1.0, 0.0))
                if _all_disjoint(positions):
                    center, side = _lp_outer_side_and_center(positions, 0.0)
                    if side < best_side:
                        best_side = side
                        best_center = center
                        best_params = list(de_result.x)
        except Exception:
            pass

    # Phase 3: L-BFGS-B local refinement with smooth surrogate
    if time.time() < deadline - 20.0:
        for oang in [0.0, 15.0, -15.0, 30.0]:
            if time.time() > deadline - 10.0:
                break
            
            def lbfgs_objective(params):
                return _objective(params, beta=50.0, oangle=oang)
            
            try:
                result = minimize(
                    lbfgs_objective,
                    x0=best_params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_side:
                    positions = []
                    for i in range(11):
                        positions.append((result.x[2*i], result.x[2*i+1], 1.0, 0.0))
                    if _all_disjoint(positions):
                        center, side = _lp_outer_side_and_center(positions, oang)
                        if side < best_side:
                            best_side = side
                            best_center = center
                            best_params = list(result.x)
            except Exception:
                pass

    # Phase 4: Final exact polish using LP
    if time.time() < deadline - 5.0:
        # Use the best parameters found
        cur_params = list(best_params)
        
        # Fine-tune with small perturbations
        for step_size in [0.01, 0.001]:
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for idx in range(len(cur_params)):
                    for delta in [-step_size, step_size]:
                        new_params = list(cur_params)
                        new_params[idx] += delta
                        positions = []
                        for i in range(11):
                            positions.append((new_params[2*i], new_params[2*i+1], 1.0, 0.0))
                        if _all_disjoint(positions):
                            center, side = _lp_outer_side_and_center(positions, 0.0)
                            if side < best_side - 1e-12:
                                best_side = side
                                best_center = center
                                cur_params = new_params
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    cur_params = list(cur_params)

        best_params = cur_params

    # Build final inner hexagons
    final_positions = []
    for i in range(11):
        x = best_params[2 * i]
        y = best_params[2 * i + 1]
        final_positions.append((x, y, 1.0, 0.0))

    # Verify disjointness
    if not _all_disjoint(final_positions):
        # Scale up slightly to ensure disjointness
        for scale in [1.001, 1.01, 1.05]:
            scaled = [(x * scale, y * scale, 1.0, 0.0) for x, y, s, a in final_positions]
            if _all_disjoint(scaled):
                final_positions = scaled
                break

    # Compute final outer hexagon using LP
    ox, oy = best_center
    oangle = 0.0
    final_side = _min_outer_side(final_positions, ox, oy, oangle)

    # Verify all inner hexagons are inside outer
    outer_hex = (ox, oy, final_side, oangle)
    valid = True
    for h in final_positions:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Binary search for minimum valid side length
        lo, hi = final_side, final_side * 1.1
        for _ in range(50):
            mid = (lo + hi) / 2
            test_outer = (ox, oy, mid, oangle)
            ok = all(_inner_in_outer(h, test_outer) for h in final_positions)
            if ok:
                hi = mid
            else:
                lo = mid
        final_side = hi

    # Format output
    inner = [[p[0], p[1], p[3]] for p in final_positions]

    return inner, (ox, oy), final_side, oangle

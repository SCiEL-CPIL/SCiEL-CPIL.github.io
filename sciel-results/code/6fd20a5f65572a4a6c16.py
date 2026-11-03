"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses convex optimization (LP for outer hexagon, nonlinear for inner positions).
"""
import math
import time
import itertools
import random
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution

N = 11
DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_GLOBAL_DEADLINE = None  # Set in construct()


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon."""
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


def _min_outer_side_lp(inner_hexes, oangle=0.0):
    """Find minimum side length of outer hexagon using LP."""
    c = [0, 0, 1]  # minimize t (apothem)
    
    A_ub = []
    b_ub = []
    
    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                nx = math.cos(math.radians(oangle + 30 + 60 * k))
                ny = math.sin(math.radians(oangle + 30 + 60 * k))
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        t = res.x[2]
        return t * 2.0 / math.sqrt(3)
    else:
        return float('inf')


def _compute_outer_for_positions(positions, angles, oangle=0.0):
    """Compute optimal outer hexagon for given inner positions and angles."""
    inner_hexes = [(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(len(positions))]
    side = _min_outer_side_lp(inner_hexes, oangle)
    
    # Get center from LP
    c = [0, 0, 1]
    A_ub = []
    b_ub = []
    
    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                nx = math.cos(math.radians(oangle + 30 + 60 * k))
                ny = math.sin(math.radians(oangle + 30 + 60 * k))
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        center = (res.x[0], res.x[1])
    else:
        cx_mean = sum(p[0] for p in positions) / len(positions)
        cy_mean = sum(p[1] for p in positions) / len(positions)
        center = (cx_mean, cy_mean)
    
    return center, side


def _compute_outer_fast(positions, angles, oangle=0.0, cx_hint=None, cy_hint=None):
    """Fast computation using smooth support function, then refine with LP if needed."""
    n = len(positions)
    if cx_hint is None:
        cx_hint = sum(p[0] for p in positions) / n
    if cy_hint is None:
        cy_hint = sum(p[1] for p in positions) / n
    
    # Use smooth estimate
    side_est = _outer_side_smooth(positions, angles, oangle, cx_hint, cy_hint)
    
    # Try a few center offsets to find better center
    best_side = side_est
    best_cx, best_cy = cx_hint, cy_hint
    
    # Small grid search around hint
    for dx in [-0.1, 0.0, 0.1]:
        for dy in [-0.1, 0.0, 0.1]:
            s = _outer_side_smooth(positions, angles, oangle, cx_hint + dx, cy_hint + dy)
            if s < best_side:
                best_side = s
                best_cx = cx_hint + dx
                best_cy = cy_hint + dy
    
    return (best_cx, best_cy), best_side





def _support_function(s, ang, nx, ny):
    """Support function of a unit hexagon with side s, rotation ang, in direction (nx, ny)."""
    max_val = 0.0
    for m in range(6):
        a = math.radians(ang + 60 * m)
        vx = s * math.cos(a)
        vy = s * math.sin(a)
        val = vx * nx + vy * ny
        if val > max_val:
            max_val = val
    return max_val


def _outer_side_smooth(positions, angles, oangle, cx, cy):
    """Compute outer hexagon side length via smooth support function (no LP)."""
    sqrt3 = math.sqrt(3)
    max_apothem = 0.0
    for i in range(len(positions)):
        px, py = positions[i]
        ang = angles[i]
        for k in range(6):
            theta = math.radians(oangle + 30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            # Support of inner hex in direction (nx, ny) from center
            s_inner = _support_function(1.0, ang, nx, ny)
            # Project center
            proj = (px - cx) * nx + (py - cy) * ny
            # Required apothem contribution
            apothem_needed = proj + s_inner
            if apothem_needed > max_apothem:
                max_apothem = apothem_needed
    # Also need negative directions
    for i in range(len(positions)):
        px, py = positions[i]
        ang = angles[i]
        for k in range(6):
            theta = math.radians(oangle + 30 + 60 * k + 180)
            nx = math.cos(theta)
            ny = math.sin(theta)
            s_inner = _support_function(1.0, ang, nx, ny)
            proj = (px - cx) * nx + (py - cy) * ny
            apothem_needed = proj + s_inner
            if apothem_needed > max_apothem:
                max_apothem = apothem_needed
    return max_apothem * 2.0 / sqrt3


def _pairwise_violation(pos, angles, i, j):
    """Check if hexagons i and j are disjoint. Return max violation (positive = overlap)."""
    max_violation = 0.0
    dx = pos[2*j] - pos[2*i]
    dy = pos[2*j+1] - pos[2*i+1]
    dist_sq = dx * dx + dy * dy
    # Quick reject: if distance > 2 (sum of circumradii for unit hexes), they're disjoint
    if dist_sq > 4.0 + 1e-12:
        return 0.0
    
    for k in range(12):
        theta = math.radians(15 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        s_i = _support_function(1.0, angles[i], nx, ny)
        s_j = _support_function(1.0, angles[j], -nx, -ny)
        
        dist_proj = dx * nx + dy * ny
        violation = s_i + s_j - dist_proj
        if violation > max_violation:
            max_violation = violation
    
    return max_violation


def construct():
    global _GLOBAL_DEADLINE
    start_time = time.time()
    _GLOBAL_DEADLINE = start_time + 150.0 - DEADLINE_MARGIN
    deadline = _GLOBAL_DEADLINE
    sqrt3 = math.sqrt(3)
    
    def time_ok():
        return time.time() < deadline

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Generate lattice positions: center + ring1 + ring2
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)  # 7 positions

    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # Phase 1: Lattice enumeration with multiple outer orientations
    # Use only a few orientations to save time
    oangles_phase1 = [0.0, 30.0, -30.0]
    # Limit combinations to keep time bounded
    max_combos = 50
    combo_count = 0
    for combo in itertools.combinations(range(len(ring2)), 4):
        if not time_ok() or combo_count >= max_combos:
            break
        combo_count += 1
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]

        for oangle in oangles_phase1:
            if not time_ok():
                break
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            center, side = _compute_outer_for_positions(full_xy, [0.0]*11, oangle)
            if side < best_side:
                best_side = side
                best_positions = full_xy[:]
                best_angles = [0.0] * 11
                best_center = center
                best_oangle = oangle

    if best_positions is None:
        full_xy = [axial_to_xy(q, r) for q, r in pos_axial[:7]]
        best_positions = full_xy
        best_angles = [0.0] * 7
        best_center, best_side = _compute_outer_for_positions(best_positions, best_angles, 0.0)
        best_oangle = 0.0

    # Phase 1b: Rigid-body kinematic chain search
    # Hexagon 0 (center) is fixed at origin. Each hexagon i>0 is linked to
    # hexagon (i-1) by a fixed-distance joint. The joint angle relative to
    # the parent's orientation determines the child's position.
    # This reduces 33 DOF to ~22 DOF (11 angles + 11 relative orientations).
    
    def rigid_chain_positions(joint_angles, rel_orientations):
        """Compute positions from joint angles and relative orientations.
        joint_angles[i] = angle of vector from hex i-1 to hex i (absolute)
        rel_orientations[i] = rotation of hex i relative to hex i-1
        Returns list of (x, y, angle) for all 11 hexagons.
        """
        n = len(joint_angles) + 1  # 11 hexagons
        positions = [(0.0, 0.0, 0.0)]  # hex 0 at origin, angle 0
        for i in range(1, n):
            parent_x, parent_y, parent_angle = positions[i-1]
            ja = math.radians(joint_angles[i-1])
            # Distance between adjacent hex centers in tight packing
            dist = sqrt3  # distance between centers of touching unit hexagons
            child_x = parent_x + dist * math.cos(ja)
            child_y = parent_y + dist * math.sin(ja)
            child_angle = parent_angle + rel_orientations[i-1]
            positions.append((child_x, child_y, child_angle))
        return positions
    
    def rigid_chain_objective(state):
        """Objective for rigid chain: outer side + overlap penalty."""
        n_joints = 10  # 10 joints for 11 hexagons
        joint_angles = list(state[:n_joints])
        rel_orientations = list(state[n_joints:])
        
        positions = rigid_chain_positions(joint_angles, rel_orientations)
        pos_list = [(p[0], p[1]) for p in positions]
        ang_list = [p[2] for p in positions]
        
        # Compute outer side using smooth estimate
        cx_mean = sum(p[0] for p in pos_list) / len(pos_list)
        cy_mean = sum(p[1] for p in pos_list) / len(pos_list)
        side_est = _outer_side_smooth(pos_list, ang_list, best_oangle, cx_mean, cy_mean)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(len(pos_list)):
            for j in range(i + 1, len(pos_list)):
                dx = pos_list[j][0] - pos_list[i][0]
                dy = pos_list[j][1] - pos_list[i][1]
                dist_sq = dx * dx + dy * dy
                if dist_sq < 4.0:  # potential overlap
                    dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.001
                    nx = dx / dist
                    ny = dy / dist
                    s_i = _support_function(1.0, ang_list[i], nx, ny)
                    s_j = _support_function(1.0, ang_list[j], -nx, -ny)
                    violation = s_i + s_j - dist
                    if violation > 0:
                        penalty += violation ** 2
        return side_est + 300.0 * penalty
    
    # Try several topologies via differential evolution on the rigid chain
    if time_ok() and (deadline - time.time()) > 30:
        # Initial guess: roughly circular arrangement
        base_angles = [i * 60.0 for i in range(10)]  # hexagonal ring
        base_rels = [0.0] * 10
        
        for trial_seed in [42]:
            if not time_ok() or (deadline - time.time()) < 15:
                break
            
            bounds = []
            for i in range(10):
                bounds.append((base_angles[i] - 40, base_angles[i] + 40))
            for i in range(10):
                bounds.append((-60, 60))
            
            try:
                de_res = differential_evolution(
                    rigid_chain_objective, bounds,
                    maxiter=20, popsize=12, tol=1e-5,
                    seed=trial_seed, polish=False, init='sobol'
                )
                if de_res.fun < best_side + 20:
                    # Verify and extract
                    joint_angles = list(de_res.x[:10])
                    rel_orientations = list(de_res.x[10:])
                    positions = rigid_chain_positions(joint_angles, rel_orientations)
                    hexes = [(p[0], p[1], 1.0, p[2]) for p in positions]
                    
                    if _all_disjoint(hexes):
                        pos_list = [(p[0], p[1]) for p in positions]
                        ang_list = [p[2] for p in positions]
                        center, side = _compute_outer_for_positions(pos_list, ang_list, best_oangle)
                        if side < best_side:
                            best_side = side
                            best_positions = pos_list
                            best_angles = ang_list
                            best_center = center
            except Exception:
                pass

    # Phase 2: Nonlinear optimization of inner positions and angles
    if time_ok() and (deadline - time.time()) > 20:
        n_hex = len(best_positions)

        def objective_fast(state):
            """Fast objective using smooth outer side estimate."""
            if not time_ok():
                return 1e6
            positions = [(state[3*i], state[3*i+1]) for i in range(n_hex)]
            angles_list = [state[3*i+2] for i in range(n_hex)]
            
            # Use smooth estimate for speed
            cx_mean = sum(p[0] for p in positions) / n_hex
            cy_mean = sum(p[1] for p in positions) / n_hex
            side_est = _outer_side_smooth(positions, angles_list, best_oangle, cx_mean, cy_mean)

            # Penalty for overlaps
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_violation(state, angles_list, i, j)
                    if v > 0:
                        penalty += v ** 2
            return side_est + 200.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        # First pass: fast optimization with reduced iterations
        try:
            result = minimize(objective_fast, init_state, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_side + 50:
                final_state = result.x
                hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

        # Phase 2b: Rigid-chain local refinement on best solution
        if time_ok() and (deadline - time.time()) > 15 and len(best_positions) == 11:
            # Try to fit a rigid chain to the current best solution
            # and optimize joint angles locally
            def rigid_local_objective(state):
                if not time_ok():
                    return 1e6
                n_joints = 10
                joint_angles = list(state[:n_joints])
                rel_orientations = list(state[n_joints:])
                positions = rigid_chain_positions(joint_angles, rel_orientations)
                pos_list = [(p[0], p[1]) for p in positions]
                ang_list = [p[2] for p in positions]
                cx_mean = sum(p[0] for p in pos_list) / len(pos_list)
                cy_mean = sum(p[1] for p in pos_list) / len(pos_list)
                side_est = _outer_side_smooth(pos_list, ang_list, best_oangle, cx_mean, cy_mean)
                penalty = 0.0
                for i in range(len(pos_list)):
                    for j in range(i + 1, len(pos_list)):
                        dx = pos_list[j][0] - pos_list[i][0]
                        dy = pos_list[j][1] - pos_list[i][1]
                        dist_sq = dx * dx + dy * dy
                        if dist_sq < 4.0:
                            dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.001
                            nx = dx / dist
                            ny = dy / dist
                            s_i = _support_function(1.0, ang_list[i], nx, ny)
                            s_j = _support_function(1.0, ang_list[j], -nx, -ny)
                            violation = s_i + s_j - dist
                            if violation > 0:
                                penalty += violation ** 2
                return side_est + 300.0 * penalty
            
            # Initialize joint angles from current best positions
            init_ja = [0.0] * 10
            init_ro = [0.0] * 10
            for i in range(1, 11):
                dx = best_positions[i][0] - best_positions[i-1][0]
                dy = best_positions[i][1] - best_positions[i-1][1]
                init_ja[i-1] = math.degrees(math.atan2(dy, dx))
                init_ro[i-1] = best_angles[i] - best_angles[i-1]
            
            init_state_r = init_ja + init_ro
            try:
                res_r = minimize(rigid_local_objective, init_state_r, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
                if res_r.fun < best_side + 5:
                    ja = list(res_r.x[:10])
                    ro = list(res_r.x[10:])
                    positions = rigid_chain_positions(ja, ro)
                    hexes = [(p[0], p[1], 1.0, p[2]) for p in positions]
                    if _all_disjoint(hexes):
                        pos_list = [(p[0], p[1]) for p in positions]
                        ang_list = [p[2] for p in positions]
                        center, side = _compute_outer_for_positions(pos_list, ang_list, best_oangle)
                        if side < best_side:
                            best_side = side
                            best_positions = pos_list
                            best_angles = ang_list
                            best_center = center
            except Exception:
                pass

    # Phase 3: Optimize outer angle with current best positions (finer grid)
    if time_ok() and (deadline - time.time()) > 10:
        # Coarse pass
        coarse_angles = [i * 5.0 for i in range(-6, 7)]
        for oangle in coarse_angles:
            if not time_ok():
                break
            center, side = _compute_outer_for_positions(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle
        # Fine pass around best
        fine_angles = [best_oangle + d for d in
                       [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]]
        for oangle in fine_angles:
            if not time_ok():
                break
            center, side = _compute_outer_for_positions(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle

    # Phase 4: Differential evolution for global search (use fast objective)
    if time_ok() and (deadline - time.time()) > 25:
        n_hex = len(best_positions)
        
        def de_objective_fast(state):
            if not time_ok():
                return 1e6
            positions = [(state[3*i], state[3*i+1]) for i in range(n_hex)]
            angles_list = [state[3*i+2] for i in range(n_hex)]
            cx_mean = sum(p[0] for p in positions) / n_hex
            cy_mean = sum(p[1] for p in positions) / n_hex
            side_est = _outer_side_smooth(positions, angles_list, best_oangle, cx_mean, cy_mean)

            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_violation(state, angles_list, i, j)
                    if v > 0:
                        penalty += v ** 2
            return side_est + 500.0 * penalty

        # Bounds for differential evolution
        bounds = []
        for i in range(n_hex):
            bounds.append((-5, 5))  # x
            bounds.append((-5, 5))  # y
            bounds.append((-30, 30))  # angle

        try:
            de_result = differential_evolution(de_objective_fast, bounds, maxiter=30,
                                                popsize=15, tol=1e-7, seed=42,
                                                polish=False, init='sobol')
            if de_result.fun < best_side + 10:
                final_state = de_result.x
                hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # Phase 5: High-precision polish - local refinement around best solution
    if time_ok() and (deadline - time.time()) > 10:
        n_hex = len(best_positions)

        def polish_objective(state):
            if not time_ok():
                return 1e6
            inner_hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2])
                           for i in range(n_hex)]
            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for cx_h, cy_h, s_h, ang_h in inner_hexes:
                verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(best_oangle + 30 + 60 * k))
                        ny = math.sin(math.radians(best_oangle + 30 + 60 * k))
                        A_ub.append([-nx, -ny, -1])
                        b_ub.append(-vx * nx - vy * ny)
            A_ub.append([0, 0, -1])
            b_ub.append(0)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)], method='highs')
            if not res.success:
                return 1e6
            t = res.x[2]
            side = t * 2.0 / sqrt3
            penalty = 0.0
            angles_list = [state[3*m+2] for m in range(n_hex)]
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_violation(state, angles_list, i, j)
                    if v > 0:
                        penalty += v ** 2
            return side + 500.0 * penalty

        # Multi-start polish: try several perturbations
        best_polish_side = best_side
        best_polish_state = None
        best_polish_center = best_center

        for attempt in range(3):
            if not time_ok() or (deadline - time.time()) < 3:
                break

            init_state = []
            for i in range(n_hex):
                x, y = best_positions[i]
                a = best_angles[i]
                if attempt > 0:
                    scale = 0.03 * attempt
                    x += (random.random() - 0.5) * scale
                    y += (random.random() - 0.5) * scale
                    a += (random.random() - 0.5) * 1.0 * attempt
                init_state.extend([x, y, a])

            try:
                result = minimize(polish_objective, init_state, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13})
                if result.fun < best_polish_side + 5:
                    final_state = result.x
                    hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                             for i in range(n_hex)]
                    if _all_disjoint(hexes):
                        positions = [(h[0], h[1]) for h in hexes]
                        angles = [h[3] for h in hexes]
                        center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                        if side < best_polish_side:
                            best_polish_side = side
                            best_polish_state = final_state
                            best_polish_center = center
            except Exception:
                pass

        if best_polish_state is not None and best_polish_side < best_side:
            final_state = best_polish_state
            hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                     for i in range(n_hex)]
            best_positions = [(h[0], h[1]) for h in hexes]
            best_angles = [h[3] for h in hexes]
            best_center = best_polish_center
            best_side = best_polish_side

    # Final verification and tight LP-based outer optimization
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                   for i in range(len(best_positions))]

    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.005, 1.01, 1.02, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                best_center, best_side = _compute_outer_for_positions(
                    best_positions, best_angles, best_oangle)
                break

    # Final tight LP to get the absolute minimum outer side
    if time_ok():
        c = [0, 0, 1]
        A_ub = []
        b_ub = []
        for cx_h, cy_h, s_h, ang_h in inner_hexes:
            verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
            for vx, vy in verts:
                for k in range(6):
                    nx = math.cos(math.radians(best_oangle + 30 + 60 * k))
                    ny = math.sin(math.radians(best_oangle + 30 + 60 * k))
                    A_ub.append([-nx, -ny, -1])
                    b_ub.append(-vx * nx - vy * ny)
        A_ub.append([0, 0, -1])
        b_ub.append(0)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            final_side = res.x[2] * 2.0 / sqrt3
            final_center = (res.x[0], res.x[1])
            # Verify all inner hexes are inside
            outer_hex = (final_center[0], final_center[1], final_side, best_oangle)
            all_inside = all(_inner_in_outer(h, outer_hex) for h in inner_hexes)
            if all_inside:
                best_side = final_side
                best_center = final_center
            else:
                # Fallback: use slightly larger side
                for margin in [1.001, 1.005, 1.01, 1.02, 1.05]:
                    outer_hex = (final_center[0], final_center[1], final_side * margin, best_oangle)
                    if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                        best_side = final_side * margin
                        best_center = final_center
                        break
        else:
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            valid = True
            for h in inner_hexes:
                if not _inner_in_outer(h, outer_hex):
                    valid = False
                    break
            if not valid:
                for _ in range(200):
                    best_side *= 1.001
                    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
                    if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                        break

    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]]
             for i in range(len(inner_hexes))]

    return inner, (best_center[0], best_center[1]), best_side, best_oangle

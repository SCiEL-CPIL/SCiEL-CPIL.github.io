"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses convex optimization (LP for outer hexagon, nonlinear for inner positions).
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution

N = 11
DEADLINE_MARGIN = 15.0  # seconds before the 150s limit


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
    # Variables: [cx, cy, t] where t is the apothem of outer hex
    # Objective: minimize t
    # For each inner vertex, for each of 6 directions of outer hex:
    #   (vx - cx) * nx_k + (vy - cy) * ny_k <= t
    c = [0, 0, 1]  # minimize t
    
    A_ub = []
    b_ub = []
    
    a_rad = math.radians(oangle)
    cos_a = math.cos(a_rad)
    sin_a = math.sin(a_rad)
    
    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                # In outer hex local frame:
                # (vx - cx) * cos(theta) + (vy - cy) * sin(theta) <= t
                # => -vx*cos(theta) + cx*cos(theta) - vy*sin(theta) + cy*sin(theta) <= t
                # => cx*cos(theta) + cy*sin(theta) - t <= vx*cos(theta) + vy*sin(theta)
                # But we need to account for rotation of outer hex...
                # Actually the outer hex is rotated by oangle, so its normals are at angles oangle + 30 + 60k
                nx = math.cos(math.radians(oangle + 30 + 60 * k))
                ny = math.sin(math.radians(oangle + 30 + 60 * k))
                # (vx - cx)*nx + (vy - cy)*ny <= t
                # -nx*cx - ny*cy - t <= -vx*nx - vy*ny
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    
    # Also need t >= 0
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        t = res.x[2]
        # side = 2 * apothem / sqrt(3)
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
        # Fallback to centroid
        cx_mean = sum(p[0] for p in positions) / len(positions)
        cy_mean = sum(p[1] for p in positions) / len(positions)
        center = (cx_mean, cy_mean)
    
    return center, side


def _separation_distance(hex1, hex2, direction):
    """Minimum distance along direction needed for two hexagons to be disjoint."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    # Support function of hexagon in direction (nx, ny):
    # max over vertices of (v - c) . (nx, ny)
    def support(cx, cy, s, ang, nx, ny):
        max_val = 0
        for k in range(6):
            a = math.radians(ang + 60 * k)
            vx = s * math.cos(a)
            vy = s * math.sin(a)
            val = vx * nx + vy * ny
            if val > max_val:
                max_val = val
        return max_val
    
    # For disjointness in direction d: (c2 - c1) . d >= support1(d) + support2(-d)
    # = support1(d) + support2(d) [by symmetry of hexagon]
    s1_dir = support(cx1, cy1, s1, a1, direction[0], direction[1])
    s2_dir = support(cx2, cy2, s2, a2, -direction[0], -direction[1])
    return s1_dir + s2_dir


def _pairwise_separation_constraint(pos, angles, i, j, oangle=0.0):
    """Check if hexagons i and j are disjoint. Return violation (negative = OK)."""
    hex_i = (pos[2*i], pos[2*i+1], 1.0, angles[i])
    hex_j = (pos[2*j], pos[2*j+1], 1.0, angles[j])
    
    # Check in multiple directions
    max_violation = 0
    for k in range(12):
        theta = math.radians(15 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        def support(cx, cy, s, ang, nx, ny):
            max_val = 0
            for m in range(6):
                a = math.radians(ang + 60 * m)
                vx = s * math.cos(a)
                vy = s * math.sin(a)
                val = vx * nx + vy * ny
                if val > max_val:
                    max_val = val
            return max_val
        
        s_i = support(pos[2*i], pos[2*i+1], 1.0, angles[i], nx, ny)
        s_j = support(pos[2*j], pos[2*j+1], 1.0, angles[j], -nx, -ny)
        
        dx = pos[2*j] - pos[2*i]
        dy = pos[2*j+1] - pos[2*i+1]
        dist_proj = dx * nx + dy * ny
        
        violation = s_i + s_j - dist_proj
        if violation > max_violation:
            max_violation = violation
    
    return max_violation


def _outer_side_smooth(state, oangle, n_hex, beta=50.0):
    """Smooth log-sum-exp surrogate for the outer hexagon side length.
    state: flat array [x1,y1,a1, x2,y2,a2, ...]
    Returns (side, gradients not computed here; used inside objective)."""
    a_rad = math.radians(oangle)
    # Precompute normals for the 6 faces of the outer hexagon
    normals = []
    for k in range(6):
        theta = math.radians(oangle + 30 + 60 * k)
        normals.append((math.cos(theta), math.sin(theta)))
    
    # For each inner hexagon vertex, compute projection onto each normal
    # We need max over all (vertex, normal) of: v_x*nx + v_y*ny - (cx*nx + cy*ny)
    # But cx,cy are also variables. For fixed positions, the LP gives us the answer.
    # For smooth surrogate: treat outer center as also optimized, or use a different approach.
    # 
    # Alternative: for fixed inner positions, the minimum enclosing hexagon with fixed orientation
    # can be found by LP. The side length is a convex function of the positions.
    # We can compute it via LP and use subgradient, but for L-BFGS-B we need smoothness.
    #
    # Better approach: parameterize by the 6 face offsets t_k (all equal for regular hexagon)
    # and outer center (cx, cy). The constraint is:
    #   For each inner vertex v_i: v_i . n_k - c . n_k <= t  for all k
    #   Minimize t subject to these constraints.
    # This is an LP with 3 variables (cx, cy, t). The optimal t is piecewise linear in the positions.
    #
    # For smoothness, we can use the log-sum-exp over the "violations" if we fix the center,
    # but the center is also optimized. Let's just use the LP value and rely on Nelder-Mead
    # or use a different smooth approximation.
    #
    # Actually, let's compute the LP value directly and use it. For the polish step, we'll
    # use a finer local search.
    
    # Compute all vertex projections
    projections = []  # list of (vx*nx + vy*ny, cx_coeff, cy_coeff)
    for i in range(n_hex):
        cx_h = state[3*i]
        cy_h = state[3*i+1]
        ang_h = state[3*i+2]
        verts = _hex_vertices(cx_h, cy_h, 1.0, ang_h)
        for vx, vy in verts:
            for nx, ny in normals:
                # Constraint: (vx - cx)*nx + (vy - cy)*ny <= t
                # => -cx*nx - cy*ny - t <= -vx*nx - vy*ny
                # The "RHS" for this constraint is -vx*nx - vy*ny
                projections.append((vx, vy, nx, ny))
    
    # We need to solve: minimize t s.t. for all (vx,vy,nx,ny):
    #   cx*nx + cy*ny - t <= vx*nx + vy*ny
    # This is the same LP as before. Let's just call the LP.
    return None  # placeholder, will use LP directly


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    sqrt3 = math.sqrt(3)
    
    # Initial lattice configuration
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)
    
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0
    
    # Try all combinations of 4 from ring2
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Try several outer hexagon orientations
        for oangle in [0.0, 15.0, 30.0, 45.0, -15.0, -30.0, -45.0]:
            if time.time() > deadline:
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
    
    # Now optimize inner positions and angles using nonlinear optimization
    if time.time() < deadline:
        n_hex = len(best_positions)
        
        # State vector: [x1, y1, a1, x2, y2, a2, ..., xn, yn, an]
        def state_to_hexes(state):
            hexes = []
            for i in range(n_hex):
                hexes.append((state[3*i], state[3*i+1], 1.0, state[3*i+2]))
            return hexes
        
        def objective(state):
            hexes = state_to_hexes(state)
            # Compute outer hexagon side
            inner_hexes = [(h[0], h[1], h[2], h[3]) for h in hexes]
            # Use LP for outer
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
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
            if not res.success:
                return 1e6
            t = res.x[2]
            side = t * 2.0 / sqrt3
            
            # Add penalty for overlaps
            penalty = 0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    hex_i = (state[3*i], state[3*i+1], 1.0, state[3*i+2])
                    hex_j = (state[3*j], state[3*j+1], 1.0, state[3*j+2])
                    max_violation = 0
                    for k in range(12):
                        theta = math.radians(15 * k)
                        nx = math.cos(theta)
                        ny = math.sin(theta)
                        
                        def support(cx, cy, s, ang, nx, ny):
                            max_val = 0
                            for m in range(6):
                                a = math.radians(ang + 60 * m)
                                vx = s * math.cos(a)
                                vy = s * math.sin(a)
                                val = vx * nx + vy * ny
                                if val > max_val:
                                    max_val = val
                            return max_val
                        
                        s_i = support(state[3*i], state[3*i+1], 1.0, state[3*i+2], nx, ny)
                        s_j = support(state[3*j], state[3*j+1], 1.0, state[3*j+2], -nx, -ny)
                        
                        dx = state[3*j] - state[3*i]
                        dy = state[3*j+1] - state[3*i+1]
                        dist_proj = dx * nx + dy * ny
                        
                        violation = s_i + s_j - dist_proj
                        if violation > max_violation:
                            max_violation = violation
                    
                    if max_violation > 0:
                        penalty += max_violation ** 2
            
            return side + 100.0 * penalty
        
        # Initial state
        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        
        # Run optimization
        try:
            result = minimize(objective, init_state, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success or result.fun < best_side + 100:
                final_state = result.x
                hexes = state_to_hexes(final_state)
                
                # Verify disjointness
                if _all_disjoint(hexes):
                    inner_hexes = [(h[0], h[1], h[2], h[3]) for h in hexes]
                    center, side = _compute_outer_for_positions(
                        [(h[0], h[1]) for h in hexes], 
                        [h[3] for h in hexes], 
                        best_oangle
                    )
                    if side < best_side:
                        best_side = side
                        best_positions = [(h[0], h[1]) for h in hexes]
                        best_angles = [h[3] for h in hexes]
                        best_center = center
        except Exception:
            pass
    
    # Try optimizing outer angle too
    if time.time() < deadline - 20:
        for oangle in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]:
            if time.time() > deadline:
                break
            inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i]) for i in range(len(best_positions))]
            center, side = _compute_outer_for_positions(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle
    
    # Final high-precision polish pass using L-BFGS-B with smooth surrogate
    if time.time() < deadline - 5:
        n_hex = len(best_positions)
        sqrt3 = math.sqrt(3)
        
        def _lp_outer_side(state, oangle, n_hex):
            """Compute outer side via LP for given state."""
            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for i in range(n_hex):
                cx_h = state[3*i]
                cy_h = state[3*i+1]
                ang_h = state[3*i+2]
                verts = _hex_vertices(cx_h, cy_h, 1.0, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(oangle + 30 + 60 * k))
                        ny = math.sin(math.radians(oangle + 30 + 60 * k))
                        A_ub.append([-nx, -ny, -1])
                        b_ub.append(-vx * nx - vy * ny)
            A_ub.append([0, 0, -1])
            b_ub.append(0)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                         bounds=[(None, None), (None, None), (0, None)], method='highs')
            if res.success:
                t = res.x[2]
                side = t * 2.0 / sqrt3
                center = (res.x[0], res.x[1])
                return side, center
            return float('inf'), (0, 0)
        
        def _overlap_penalty(state, n_hex):
            """Compute overlap penalty between all pairs of inner hexagons."""
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    hex_i = (state[3*i], state[3*i+1], 1.0, state[3*i+2])
                    hex_j = (state[3*j], state[3*j+1], 1.0, state[3*j+2])
                    # Check disjointness
                    if not _hexagons_disjoint(hex_i, hex_j, eps=1e-7):
                        # Compute penetration depth
                        max_violation = 0
                        for k in range(24):
                            theta = math.radians(15 * k)
                            nx = math.cos(theta)
                            ny = math.sin(theta)
                            
                            def support(cx, cy, s, ang, nx, ny):
                                max_val = 0
                                for m in range(6):
                                    a = math.radians(ang + 60 * m)
                                    vx = s * math.cos(a)
                                    vy = s * math.sin(a)
                                    val = vx * nx + vy * ny
                                    if val > max_val:
                                        max_val = val
                                return max_val
                            
                            s_i = support(state[3*i], state[3*i+1], 1.0, state[3*i+2], nx, ny)
                            s_j = support(state[3*j], state[3*j+1], 1.0, state[3*j+2], -nx, -ny)
                            
                            dx = state[3*j] - state[3*i]
                            dy = state[3*j+1] - state[3*i+1]
                            dist_proj = dx * nx + dy * ny
                            
                            violation = s_i + s_j - dist_proj
                            if violation > max_violation:
                                max_violation = violation
                        
                        if max_violation > 0:
                            penalty += max_violation ** 2
            return penalty
        
        def polish_objective(state):
            side, center = _lp_outer_side(state, best_oangle, n_hex)
            if side == float('inf'):
                return 1e6
            pen = _overlap_penalty(state, n_hex)
            return side + 500.0 * pen
        
        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        
        try:
            # First try L-BFGS-B
            result = minimize(polish_objective, init_state, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < best_side:
                final_state = result.x
                hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2]) for i in range(n_hex)]
                if _all_disjoint(hexes):
                    center, side = _lp_outer_side(final_state, best_oangle, n_hex)
                    if side < best_side:
                        best_side = side
                        best_positions = [(h[0], h[1]) for h in hexes]
                        best_angles = [h[3] for h in hexes]
                        best_center = center
        except Exception:
            pass
        
        # Second polish with tighter tolerance
        if time.time() < deadline - 2:
            try:
                init_state2 = []
                for i in range(n_hex):
                    init_state2.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
                
                result2 = minimize(polish_objective, init_state2, method='L-BFGS-B',
                                 options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
                if result2.fun < best_side:
                    final_state2 = result2.x
                    hexes2 = [(final_state2[3*i], final_state2[3*i+1], 1.0, final_state2[3*i+2]) for i in range(n_hex)]
                    if _all_disjoint(hexes2):
                        center2, side2 = _lp_outer_side(final_state2, best_oangle, n_hex)
                        if side2 < best_side:
                            best_side = side2
                            best_positions = [(h[0], h[1]) for h in hexes2]
                            best_angles = [h[3] for h in hexes2]
                            best_center = center2
            except Exception:
                pass
    
    # Final verification and adjustment
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i]) for i in range(len(best_positions))]
    
    if not _all_disjoint(inner_hexes):
        # Scale up positions slightly
        for scale in [1.001, 1.005, 1.01, 1.02]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                best_center, best_side = _compute_outer_for_positions(best_positions, best_angles, best_oangle)
                break
    
    # Verify containment with small tolerance
    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break
    
    if not valid:
        # Increase side until valid
        for _ in range(200):
            best_side *= 1.0005
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break
    
    # Format output
    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]] for i in range(len(inner_hexes))]
    
    return inner, (best_center[0], best_center[1]), best_side, best_oangle

"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Direct nonlinear optimization with exact support functions and SAT-based overlap.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
DEADLINE_MARGIN = 10.0


def _hex_support(ang_deg, nx, ny):
    """Exact support function of unit hexagon (flat-top) with rotation ang_deg in direction (nx,ny)."""
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Rotate direction into hexagon's local frame
    lx = nx * ca + ny * sa
    ly = -nx * sa + ny * ca
    # Hexagon vertices in local frame (flat-top): angles 0,60,120,180,240,300
    # Support = max over vertices of (vx*lx + vy*ly)
    # Vertices at angles 0,60,120,180,240,300, radius 1
    max_val = 0.0
    for k in range(6):
        va = math.radians(60 * k)
        vx = math.cos(va)
        vy = math.sin(va)
        val = vx * lx + vy * ly
        if val > max_val:
            max_val = val
    return max_val


def _hex_support_fast(ang_deg, nx, ny):
    """Faster: precompute vertex directions, use max dot product."""
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    lx = nx * ca + ny * sa
    ly = -nx * sa + ny * ca
    # 6 vertex directions (unit vectors at 0,60,...,300 degrees)
    # Precomputed cos/sin
    max_val = 0.0
    # Unroll for speed
    vals = (lx,
            lx * 0.5 - ly * 0.8660254037844386,
            -lx * 0.5 - ly * 0.8660254037844386,
            -lx,
            -lx * 0.5 + ly * 0.8660254037844386,
            lx * 0.5 + ly * 0.8660254037844386)
    for v in vals:
        if v > max_val:
            max_val = v
    return max_val


def _overlap_penalty(pos_i, ang_i, pos_j, ang_j):
    """
    Compute overlap between two unit hexagons using SAT.
    Returns 0 if separated, positive value = penetration depth (approximate).
    """
    dx = pos_j[0] - pos_i[0]
    dy = pos_j[1] - pos_i[1]
    dist_sq = dx * dx + dy * dy
    # Quick rejection: max distance between centers for possible overlap
    if dist_sq > 4.0 + 1e-12:
        return 0.0

    # Edge normals for hex_i: angles ang_i + 30 + 60*k
    # Edge normals for hex_j: angles ang_j + 30 + 60*k
    max_pen = 0.0
    for k in range(6):
        # Normal direction from hex_i's k-th edge
        na = math.radians(ang_i + 30 + 60 * k)
        nx = math.cos(na)
        ny = math.sin(na)
        # Support of hex_i in direction n
        s_i = _hex_support_fast(ang_i, nx, ny)
        # Support of hex_j in direction -n
        s_j = _hex_support_fast(ang_j, -nx, -ny)
        # Projection of center difference
        proj = dx * nx + dy * ny
        pen = s_i + s_j - proj
        if pen > max_pen:
            max_pen = pen

    for k in range(6):
        # Normal direction from hex_j's k-th edge
        na = math.radians(ang_j + 30 + 60 * k)
        nx = math.cos(na)
        ny = math.sin(na)
        s_i = _hex_support_fast(ang_i, nx, ny)
        s_j = _hex_support_fast(ang_j, -nx, -ny)
        proj = dx * nx + dy * ny
        pen = s_i + s_j - proj
        if pen > max_pen:
            max_pen = pen

    if max_pen <= 1e-10:
        return 0.0
    return max_pen


def _outer_side_lp(pos, ang, oangle):
    """
    Compute minimum outer hexagon side length containing all inner hexagons.
    Uses LP: minimize t s.t. for each inner vertex v and each outer face normal n_k:
    (v - center) . n_k <= t * apothem
    where t*apothem is the inradius, side = 2*inradius/sqrt(3)
    """
    n_hex = len(pos)
    # Variables: cx, cy, t (inradius)
    # Constraints: for each inner hex vertex, for each outer face normal:
    #   -nx*(vx - cx) - ny*(vy - cy) <= t*APOTHEM
    #   => nx*cx + ny*cy - APOTHEM*t >= nx*vx + ny*vy
    #   => -nx*cx - ny*cy + APOTHEM*t <= -(nx*vx + ny*vy)

    A_ub = []
    b_ub = []
    for i in range(n_hex):
        cx_h, cy_h = pos[i]
        ang_h = ang[i]
        # 6 vertices of inner hexagon i
        for k in range(6):
            va = math.radians(ang_h + 60 * k)
            vx = cx_h + math.cos(va)
            vy = cy_h + math.sin(va)
            for m in range(6):
                na = math.radians(oangle + 30 + 60 * m)
                nx = math.cos(na)
                ny = math.sin(na)
                A_ub.append([-nx, -ny, APOTHEM])
                b_ub.append(-(nx * vx + ny * vy))
    # t >= 0
    A_ub.append([0, 0, -1])
    b_ub.append(0)

    from scipy.optimize import linprog
    c = [0, 0, 1]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    if res.success:
        inradius = res.x[2]
        side = inradius * 2.0 / APOTHEM
        center = (res.x[0], res.x[1])
        return center, side
    return None, float('inf')


def _outer_side_direct(pos, ang, oangle):
    """
    Compute outer hexagon side without LP (faster, approximate).
    For each of 6 face normals, find max projection of all inner hex vertices.
    The inradius must be >= max projection for all faces.
    Center is chosen to minimize the max inradius.
    """
    n_hex = len(pos)
    # Collect all vertices
    all_vx = []
    all_vy = []
    for i in range(n_hex):
        cx_h, cy_h = pos[i]
        ang_h = ang[i]
        for k in range(6):
            va = math.radians(ang_h + 60 * k)
            all_vx.append(cx_h + math.cos(va))
            all_vy.append(cy_h + math.sin(va))

    all_vx = np.array(all_vx)
    all_vy = np.array(all_vy)

    # For each face normal direction, compute max and min projection
    # inradius_k = (max_proj - min_proj) / 2
    # center_k = (max_proj + min_proj) / 2 along that normal
    # We need a single center that works for all faces.
    # Simple approach: use centroid, compute required inradius
    cx_mean = float(np.mean(all_vx))
    cy_mean = float(np.mean(all_vy))

    max_inradius = 0.0
    for m in range(6):
        na = math.radians(oangle + 30 + 60 * m)
        nx = math.cos(na)
        ny = math.sin(na)
        projs = all_vx * nx + all_vy * ny
        max_p = float(np.max(projs))
        min_p = float(np.min(projs))
        # Inradius needed if center is at centroid
        ir = max(max_p - (cx_mean * nx + cy_mean * ny),
                 (cx_mean * nx + cy_mean * ny) - min_p)
        if ir > max_inradius:
            max_inradius = ir

    # Better: optimize center for this fixed oangle
    # Minimize max over faces of (max_proj - center_proj) and (center_proj - min_proj)
    # This is: minimize max_k (max_p_k - c.n_k, c.n_k - min_p_k)
    # = minimize max_k (max(max_p_k - c.n_k, c.n_k - min_p_k))
    # = minimize max_k (max_p_k - min_p_k)/2 + |c.n_k - (max_p_k+min_p_k)/2|
    # So optimal c minimizes max_k |c.n_k - mid_k| where mid_k = (max_p_k+min_p_k)/2
    # This is a 2D problem. Use simple iteration.
    mids = []
    half_widths = []
    normals = []
    for m in range(6):
        na = math.radians(oangle + 30 + 60 * m)
        nx = math.cos(na)
        ny = math.sin(na)
        projs = all_vx * nx + all_vy * ny
        max_p = float(np.max(projs))
        min_p = float(np.min(projs))
        mids.append((max_p + min_p) / 2.0)
        half_widths.append((max_p - min_p) / 2.0)
        normals.append((nx, ny))

    # Optimize center to minimize max_k |c.n_k - mid_k|
    cx, cy = cx_mean, cy_mean
    for _ in range(50):
        # Compute gradient-like update
        # f(c) = max_k |c.n_k - mid_k|
        # Subgradient: n_k * sign(c.n_k - mid_k) for argmax k
        max_val = -1
        argmax_k = 0
        for k in range(6):
            val = abs(cx * normals[k][0] + cy * normals[k][1] - mids[k])
            if val > max_val:
                max_val = val
                argmax_k = k
        nx, ny = normals[argmax_k]
        proj = cx * nx + cy * ny
        if proj > mids[argmax_k]:
            cx -= 0.1 * nx
            cy -= 0.1 * ny
        else:
            cx += 0.1 * nx
            cy += 0.1 * ny

    # Final inradius
    max_inradius = 0.0
    for k in range(6):
        nx, ny = normals[k]
        c_proj = cx * nx + cy * ny
        ir = max(max_p_k - c_proj, c_proj - min_p_k) if False else 0
        # Recompute properly
    max_inradius = 0.0
    for k in range(6):
        nx, ny = normals[k]
        c_proj = cx * nx + cy * ny
        projs = all_vx * nx + all_vy * ny
        max_p = float(np.max(projs))
        min_p = float(np.min(projs))
        ir = max(max_p - c_proj, c_proj - min_p)
        if ir > max_inradius:
            max_inradius = ir

    side = max_inradius * 2.0 / APOTHEM
    return (cx, cy), side


def _total_overlap(pos, ang):
    """Sum of squared overlaps for all pairs."""
    n = len(pos)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            p = _overlap_penalty(pos[i], ang[i], pos[j], ang[j])
            if p > 0:
                penalty += p * p
    return penalty


def _build_topology_hexagonal():
    """Central hex + 6 ring + 4 extra in ring 2."""
    positions = [(0.0, 0.0)]
    # Ring 1: 6 hexagons at distance 2*APOTHEM = SQRT3
    for k in range(6):
        a = math.radians(60 * k)
        positions.append((SQRT3 * math.cos(a), SQRT3 * math.sin(a)))
    # Ring 2: pick 4 from the 12 available
    # Positions at axial (2,0),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,0) etc.
    # In xy: axial (q,r) -> x=1.5q, y=sqrt3*(r+q/2)
    extra_axial = [(2, 0), (2, -1), (1, -2), (-1, -2)]
    for q, r in extra_axial:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        positions.append((x, y))
    angles = [0.0] * 11
    return positions, angles


def _build_topology_hexagonal_v2():
    """Different 4 in ring 2: symmetric placement."""
    positions = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60 * k)
        positions.append((SQRT3 * math.cos(a), SQRT3 * math.sin(a)))
    # 4 extra: at (2,0), (0,-2), (-2,0), (0,2) in axial -> but those are 4 apart
    # Better: (2,0), (1,-2), (-1,-2), (-2,0) -- wait that's only 4
    # Let's use: (2,0), (2,-1), (-2,0), (-2,1) for more symmetry
    extra_axial = [(2, 0), (2, -1), (-2, 0), (-2, 1)]
    for q, r in extra_axial:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        positions.append((x, y))
    angles = [0.0] * 11
    return positions, angles


def _build_topology_hexagonal_v3():
    """Another arrangement: 4 extras at alternating ring-2 positions."""
    positions = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60 * k)
        positions.append((SQRT3 * math.cos(a), SQRT3 * math.sin(a)))
    # Ring 2 positions: (2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-2),(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),(-1,2),(0,2),(1,2)
    # Pick 4 spread out: (2,0), (0,-2), (-2,0), (0,2)
    extra_axial = [(2, 0), (0, -2), (-2, 0), (0, 2)]
    for q, r in extra_axial:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        positions.append((x, y))
    angles = [0.0] * 11
    return positions, angles


def _build_topology_hexagonal_v4():
    """4 extras: (2,-1),(1,-2),(-1,2),(-2,1) - diagonal arrangement."""
    positions = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60 * k)
        positions.append((SQRT3 * math.cos(a), SQRT3 * math.sin(a)))
    extra_axial = [(2, -1), (1, -2), (-1, 2), (-2, 1)]
    for q, r in extra_axial:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        positions.append((x, y))
    angles = [0.0] * 11
    return positions, angles


def _build_topology_hexagonal_v5():
    """4 extras: (2,0),(0,-2),(-2,0),(0,2) but rotated arrangement."""
    positions = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(30 + 60 * k)  # Pointy-top orientation
        positions.append((SQRT3 * math.cos(a), SQRT3 * math.sin(a)))
    extra_axial = [(2, 0), (0, -2), (-2, 0), (0, 2)]
    for q, r in extra_axial:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        positions.append((x, y))
    angles = [0.0] * 11
    return positions, angles


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # Topologies to try
    topologies = [
        _build_topology_hexagonal(),
        _build_topology_hexagonal_v2(),
        _build_topology_hexagonal_v3(),
        _build_topology_hexagonal_v4(),
        _build_topology_hexagonal_v5(),
    ]

    oangles_to_try = [0.0, 15.0, 30.0, -15.0, -30.0]

    # Phase 1: Evaluate initial topologies with LP
    for pos, ang in topologies:
        if time.time() > deadline - 60:
            break
        for oangle in oangles_to_try:
            if time.time() > deadline - 60:
                break
            center, side = _outer_side_lp(pos, ang, oangle)
            if side < best_side:
                best_side = side
                best_positions = pos[:]
                best_angles = ang[:]
                best_center = center
                best_oangle = oangle

    if best_positions is None:
        pos, ang = topologies[0]
        best_positions = pos
        best_angles = ang
        best_center, best_side = _outer_side_lp(best_positions, best_angles, 0.0)
        best_oangle = 0.0

    n_hex = len(best_positions)

    # Phase 2: Nonlinear optimization
    def objective(state, oangle):
        """Minimize outer side + overlap penalty."""
        pos = [(state[3 * i], state[3 * i + 1]) for i in range(n_hex)]
        ang = [state[3 * i + 2] for i in range(n_hex)]

        # Compute outer side (fast approximation)
        center, side = _outer_side_direct(pos, ang, oangle)
        if side == float('inf'):
            return 1e6

        # Overlap penalty
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                p = _overlap_penalty(pos[i], ang[i], pos[j], ang[j])
                if p > 0:
                    penalty += p * p

        return side + 500.0 * penalty

    def objective_grad(state, oangle):
        """Same as objective but with higher penalty weight for polish."""
        pos = [(state[3 * i], state[3 * i + 1]) for i in range(n_hex)]
        ang = [state[3 * i + 2] for i in range(n_hex)]
        center, side = _outer_side_direct(pos, ang, oangle)
        if side == float('inf'):
            return 1e6
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                p = _overlap_penalty(pos[i], ang[i], pos[j], ang[j])
                if p > 0:
                    penalty += p * p
        return side + 2000.0 * penalty

    # Try optimization from best initial
    init_state = []
    for i in range(n_hex):
        init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

    # Phase 2a: Coarse optimization
    if time.time() < deadline - 30:
        try:
            res = minimize(lambda s: objective(s, best_oangle), init_state,
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9,
                                    'adaptive': True})
            if res.fun < best_side + 100:
                fs = res.x
                pos = [(fs[3 * i], fs[3 * i + 1]) for i in range(n_hex)]
                ang = [fs[3 * i + 2] for i in range(n_hex)]
                # Verify with LP
                center, side = _outer_side_lp(pos, ang, best_oangle)
                if side < best_side:
                    # Check no overlaps
                    has_overlap = False
                    for i in range(n_hex):
                        for j in range(i + 1, n_hex):
                            if _overlap_penalty(pos[i], ang[i], pos[j], ang[j]) > 1e-8:
                                has_overlap = True
                                break
                        if has_overlap:
                            break
                    if not has_overlap:
                        best_side = side
                        best_positions = pos
                        best_angles = ang
                        best_center = center
        except Exception:
            pass

    # Phase 2b: Try different outer angles
    if time.time() < deadline - 25:
        for oangle in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
                       -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]:
            if time.time() > deadline - 25:
                break
            center, side = _outer_side_lp(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle

    # Phase 2c: Fine optimization with refined oangle
    if time.time() < deadline - 15:
        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        try:
            res = minimize(lambda s: objective(s, best_oangle), init_state,
                           method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11,
                                    'adaptive': True})
            if res.fun < best_side + 50:
                fs = res.x
                pos = [(fs[3 * i], fs[3 * i + 1]) for i in range(n_hex)]
                ang = [fs[3 * i + 2] for i in range(n_hex)]
                center, side = _outer_side_lp(pos, ang, best_oangle)
                if side < best_side:
                    has_overlap = False
                    for i in range(n_hex):
                        for j in range(i + 1, n_hex):
                            if _overlap_penalty(pos[i], ang[i], pos[j], ang[j]) > 1e-8:
                                has_overlap = True
                                break
                        if has_overlap:
                            break
                    if not has_overlap:
                        best_side = side
                        best_positions = pos
                        best_angles = ang
                        best_center = center
        except Exception:
            pass

    # Phase 2d: Polish with high penalty
    if time.time() < deadline - 8:
        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        try:
            res = minimize(lambda s: objective_grad(s, best_oangle), init_state,
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < best_side + 20:
                fs = res.x
                pos = [(fs[3 * i], fs[3 * i + 1]) for i in range(n_hex)]
                ang = [fs[3 * i + 2] for i in range(n_hex)]
                center, side = _outer_side_lp(pos, ang, best_oangle)
                if side < best_side:
                    has_overlap = False
                    for i in range(n_hex):
                        for j in range(i + 1, n_hex):
                            if _overlap_penalty(pos[i], ang[i], pos[j], ang[j]) > 1e-8:
                                has_overlap = True
                                break
                        if has_overlap:
                            break
                    if not has_overlap:
                        best_side = side
                        best_positions = pos
                        best_angles = ang
                        best_center = center
        except Exception:
            pass

    # Phase 3: Try optimizing from other topologies if time remains
    if time.time() < deadline - 10:
        for pos_init, ang_init in topologies[1:]:
            if time.time() > deadline - 10:
                break
            init_state = []
            for i in range(n_hex):
                init_state.extend([pos_init[i][0], pos_init[i][1], ang_init[i]])
            try:
                res = minimize(lambda s: objective(s, best_oangle), init_state,
                               method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.fun < best_side + 50:
                    fs = res.x
                    pos = [(fs[3 * i], fs[3 * i + 1]) for i in range(n_hex)]
                    ang = [fs[3 * i + 2] for i in range(n_hex)]
                    center, side = _outer_side_lp(pos, ang, best_oangle)
                    if side < best_side:
                        has_overlap = False
                        for i in range(n_hex):
                            for j in range(i + 1, n_hex):
                                if _overlap_penalty(pos[i], ang[i], pos[j], ang[j]) > 1e-8:
                                    has_overlap = True
                                    break
                            if has_overlap:
                                break
                        if not has_overlap:
                            best_side = side
                            best_positions = pos
                            best_angles = ang
                            best_center = center
            except Exception:
                pass

    # Final verification and safety
    # Ensure no overlaps with small epsilon
    has_overlap = False
    for i in range(n_hex):
        for j in range(i + 1, n_hex):
            if _overlap_penalty(best_positions[i], best_angles[i],
                                best_positions[j], best_angles[j]) > 1e-6:
                has_overlap = True
                break
        if has_overlap:
            break

    if has_overlap:
        # Slightly separate: scale positions from centroid
        cx = sum(p[0] for p in best_positions) / n_hex
        cy = sum(p[1] for p in best_positions) / n_hex
        for scale in [1.001, 1.005, 1.01, 1.02, 1.03, 1.05]:
            new_pos = [(cx + (p[0] - cx) * scale, cy + (p[1] - cy) * scale)
                       for p in best_positions]
            ok = True
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    if _overlap_penalty(new_pos[i], best_angles[i],
                                        new_pos[j], best_angles[j]) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                best_positions = new_pos
                break

    # Recompute final outer hexagon
    best_center, best_side = _outer_side_lp(best_positions, best_angles, best_oangle)

    # Ensure all inner hexagons are inside outer
    # Add small safety margin to side
    best_side *= 1.0001

    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]]
             for i in range(n_hex)]

    return inner, (best_center[0], best_center[1]), best_side, best_oangle

"""Hexagon packing: 11 unit hexagons inside a regular hexagon.
Structural change: direct Nelder-Mead search over all 11 hexagon positions
and outer hexagon parameters, using a fast LP-based enclosing-hexagon computation.
"""
import math
import time
import random
import numpy as np

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 145.0  # safety margin before 150s limit

# Unit hexagon vertices (side=1, flat-top, centered at origin)
_HEX_VERTS = []
for k in range(6):
    a = math.radians(60 * k)
    _HEX_VERTS.append((math.cos(a), math.sin(a)))
_HEX_VERTS = np.array(_HEX_VERTS)

# Normal vectors for the 6 sides of a unit hexagon (flat-top)
# Each side has an outward normal; the apothem is sqrt(3)/2
_SIDE_NORMALS = []
_apothem = SQRT3 / 2
for k in range(6):
    a = math.radians(30 + 60 * k)
    _SIDE_NORMALS.append((math.cos(a), math.sin(a)))
_SIDE_NORMALS = np.array(_SIDE_NORMALS)


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon as numpy array."""
    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    verts = _HEX_VERTS * side
    verts = verts @ rot.T
    verts += np.array([cx, cy])
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2
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
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
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


def _min_outer_side_lp(hexes, ox, oy, oangle_deg=0.0):
    """
    Find minimum side length of outer hexagon containing all hexagons.
    The outer hexagon has fixed center (ox,oy) and rotation oangle_deg.
    We minimize the side length s such that all inner hex vertices are inside.
    
    For a regular hexagon with center (ox,oy), rotation theta, and side length s,
    a point (px,py) is inside iff for all k=0..5:
        (px-ox, py-oy) rotated by -theta, dot normal_k <= s * apothem
    where apothem = s * sqrt(3)/2.
    
    So the constraint is: proj_k <= s * sqrt(3)/2 for all k.
    Thus s >= max_k(proj_k) * 2 / sqrt(3).
    """
    a = -math.radians(oangle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    max_val = 0.0
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
    return max_val * 2.0 / SQRT3


def _axial_to_xy(q, r):
    """Convert axial hex coordinates to Cartesian (flat-top orientation)."""
    x = 1.5 * q
    y = SQRT3 * (r + q / 2.0)
    return (x, y)


def _initial_positions():
    """Generate initial positions: center + ring1 (6) + ring2 (4 selected)."""
    ring1_axial = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring2_axial = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
                   (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
                   (0, 2), (1, 2)]
    
    # Select 4 ring2 positions that form a compact cluster
    # Try several combinations and pick the best initial one
    best_combo = None
    best_side = float('inf')
    
    import itertools
    all_combos = list(itertools.combinations(range(len(ring2_axial)), 4))
    
    for combo in all_combos:
        ring2_selected = [ring2_axial[i] for i in combo]
        all_axial = [(0, 0)] + ring1_axial + ring2_selected
        all_xy = [_axial_to_xy(q, r) for q, r in all_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in all_xy]
        cx_mean = sum(x for x, y in all_xy) / N
        cy_mean = sum(y for x, y in all_xy) / N
        side = _min_outer_side_lp(inner_hexes, cx_mean, cy_mean, 0.0)
        if side < best_side:
            best_side = side
            best_combo = combo
    
    ring2_selected = [ring2_axial[i] for i in best_combo]
    all_axial = [(0, 0)] + ring1_axial + ring2_selected
    all_xy = [_axial_to_xy(q, r) for q, r in all_axial]
    
    # Parameters: for each hex: (x, y, angle_deg), plus outer: (cx, cy, oangle)
    # Total: 11*3 + 3 = 36 parameters
    params = []
    for x, y in all_xy:
        params.extend([x, y, 0.0])
    cx_mean = sum(x for x, y in all_xy) / N
    cy_mean = sum(y for x, y in all_xy) / N
    params.extend([cx_mean, cy_mean, 0.0])
    
    return params, best_side


def _unpack(params):
    """Unpack parameters into inner hexagons and outer hexagon."""
    inner_hexes = []
    for i in range(N):
        x = params[3 * i]
        y = params[3 * i + 1]
        a = params[3 * i + 2]
        inner_hexes.append((x, y, 1.0, a))
    ox = params[3 * N]
    oy = params[3 * N + 1]
    oangle = params[3 * N + 2]
    return inner_hexes, (ox, oy, oangle)


def _objective(params, deadline):
    """Compute the outer hexagon side length with penalty for overlaps."""
    if time.time() > deadline:
        return 1e10
    
    inner_hexes, (ox, oy, oangle) = _unpack(params)
    
    # Compute minimum outer side
    side = _min_outer_side_lp(inner_hexes, ox, oy, oangle)
    
    # Check disjointness with penalty
    penalty = 0.0
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                penalty += 5.0
    
    return side + penalty


def _nelder_mead(f, x0, max_iter=500, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
                 time_budget=None):
    """
    Nelder-Mead simplex optimization.
    f: objective function
    x0: initial point (list or numpy array)
    """
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    
    # Build initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.1 if x0[i] == 0 else 0.05 * x0[i]
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [f(x, time_budget) for x in simplex]
    
    for iteration in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
        
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i].copy() for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < 1e-12:
            break
        
        xw = simplex[-1]  # worst
        xg = simplex[0]   # best
        xc = np.mean(simplex[:-1], axis=0)  # centroid of all but worst
        
        # Reflection
        xr = xc + alpha * (xc - xw)
        fr = f(xr, time_budget)
        
        if fr < fvals[0]:
            # Expansion
            xe = xc + gamma * (xc - xw)
            fe = f(xe, time_budget)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc2 = xc + 0.5 * (xr - xc)
                fc2 = f(xc2, time_budget)
                if fc2 <= fr:
                    simplex[-1] = xc2
                    fvals[-1] = fc2
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = xg + sigma * (simplex[i] - xg)
                        fvals[i] = f(simplex[i], time_budget)
            else:
                # Inside contraction
                xc2 = xc + 0.5 * (xw - xc)
                fc2 = f(xc2, time_budget)
                if fc2 <= fvals[-1]:
                    simplex[-1] = xc2
                    fvals[-1] = fc2
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = xg + sigma * (simplex[i] - xg)
                        fvals[i] = f(simplex[i], time_budget)
    
    # Return best point
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Get initial positions
    params, initial_side = _initial_positions()
    
    best_params = params.copy()
    best_side = initial_side
    
    # Nelder-Mead optimization
    try:
        result_params, result_val = _nelder_mead(
            _objective,
            params,
            max_iter=200,
            time_budget=deadline - 30.0
        )
        if result_val < best_side:
            best_params = result_params
            best_side = result_val
    except Exception:
        pass
    
    # Second round with perturbation
    if time.time() < deadline - 15.0:
        random.seed(42)
        perturbed = best_params.copy()
        for i in range(len(perturbed)):
            if i < 3 * N:  # only perturb inner hex positions/angles
                perturbed[i] += random.gauss(0, 0.01)
        
        try:
            result_params, result_val = _nelder_mead(
                _objective,
                perturbed,
                max_iter=100,
                time_budget=deadline - 10.0
            )
            if result_val < best_side:
                best_params = result_params
                best_side = result_val
        except Exception:
            pass
    
    # Unpack best parameters
    inner_hexes, (ox, oy, oangle) = _unpack(best_params)
    
    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # Try to fix by adjusting positions
        for attempt in range(10):
            fixed = True
            for i in range(N):
                for j in range(i + 1, N):
                    if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                        # Push apart
                        dx = inner_hexes[j][0] - inner_hexes[i][0]
                        dy = inner_hexes[j][1] - inner_hexes[i][1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < 1e-9:
                            dx, dy = 0.01, 0.01
                            dist = math.sqrt(2) * 0.01
                        push = 0.01 / dist
                        inner_hexes[i] = (inner_hexes[i][0] - push * dx,
                                         inner_hexes[i][1] - push * dy,
                                         inner_hexes[i][2], inner_hexes[i][3])
                        inner_hexes[j] = (inner_hexes[j][0] + push * dx,
                                         inner_hexes[j][1] + push * dy,
                                         inner_hexes[j][2], inner_hexes[j][3])
                        fixed = False
            if fixed:
                break
    
    # Recompute outer side
    best_side = _min_outer_side_lp(inner_hexes, ox, oy, oangle)
    
    # Verify all inner hexagons are inside outer
    outer_hex = (ox, oy, best_side, oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break
    
    if not valid:
        # Binary search for minimum valid side length
        lo, hi = best_side, best_side * 1.5
        for _ in range(50):
            mid = (lo + hi) / 2
            test_outer = (ox, oy, mid, oangle)
            ok = all(_inner_in_outer(h, test_outer) for h in inner_hexes)
            if ok:
                hi = mid
            else:
                lo = mid
        best_side = hi
    
    # Format output
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    
    return inner, (ox, oy), best_side, oangle

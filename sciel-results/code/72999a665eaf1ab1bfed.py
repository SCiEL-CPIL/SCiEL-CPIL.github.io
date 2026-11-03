import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _check_disjoint(inner_hexes):
    """Check if inner hexagons are disjoint. Returns min gap (positive is good)."""
    min_gap = float('inf')
    s3 = math.sqrt(3)
    for i in range(N):
        for j in range(i + 1, N):
            dx = inner_hexes[i][0] - inner_hexes[j][0]
            dy = inner_hexes[i][1] - inner_hexes[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            gap = dist - s3
            if gap < min_gap:
                min_gap = gap
    return min_gap

def _surrogate_side_length(inner_hexes):
    """Fast surrogate for the minimum enclosing hexagon side length.
    Uses the maximum width in 6 fixed directions (0, 30, 60, 90, 120, 150).
    """
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    if not all_vertices:
        return 0.0
        
    # 6 directions for the normals of a regular hexagon
    directions = []
    for k in range(6):
        a = math.radians(30 * k)
        directions.append((math.cos(a), math.sin(a)))
        
    max_width = 0.0
    for nx, ny in directions:
        proj = [vx * nx + vy * ny for vx, vy in all_vertices]
        width = max(proj) - min(proj)
        if width > max_width:
            max_width = width
            
    # Side length S = width / (sqrt(3)/2) = width * 2 / sqrt(3)
    # Wait, width of hexagon with side S in normal direction is 2 * apothem = 2 * S * sqrt(3)/2 = S * sqrt(3).
    # So S = width / sqrt(3).
    return max_width / math.sqrt(3)

def _lp_min_enclosing(inner_hexes, angle_deg):
    """LP for the minimum inradius enclosing hexagon of fixed angle.
    Returns (cx, cy, side) or None on failure.
    """
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)

    c = [0.0, 0.0, 1.0]  # minimize t (inradius)
    A_ub = []
    b_ub = []

    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)

        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)

    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t / (math.sqrt(3) / 2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _min_enclosing_hex_side(inner_hexes):
    """Find the minimum enclosing regular hexagon over a fine angle sweep.
    Returns (cx, cy, side, angle_deg).
    """
    best_side = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_angle = 0.0

    # Coarse sweep
    for deg in range(0, 31, 1):
        if time.time() > _DEADLINE - 15.0:
            break
        r = _lp_min_enclosing(inner_hexes, deg)
        if r and r[2] < best_side:
            best_side, best_cx, best_cy, best_angle = r[2], r[0], r[1], deg

    # Fine sweep
    if time.time() < _DEADLINE - 15.0:
        lo = max(0.0, best_angle - 1.0)
        hi = min(30.0, best_angle + 1.0)
        n_steps = 41
        for i in range(n_steps):
            deg = lo + (hi - lo) * i / (n_steps - 1)
            if time.time() > _DEADLINE - 15.0:
                break
            r = _lp_min_enclosing(inner_hexes, deg)
            if r and r[2] < best_side:
                best_side, best_cx, best_cy, best_angle = r[2], r[0], r[1], deg

    return best_cx, best_cy, best_side, best_angle

def _nelder_mead_polish(inner_hexes, max_iter=300, tol=1e-5):
    """Polish inner positions/angles using a fast surrogate objective."""
    def objective(params):
        new_inner = []
        for i in range(N):
            new_inner.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        
        # Disjointness penalty
        s3 = math.sqrt(3)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = new_inner[i][0] - new_inner[j][0]
                dy = new_inner[i][1] - new_inner[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < s3:
                    penalty += (s3 - dist) ** 2
        
        side = _surrogate_side_length(new_inner)
        return side + 100.0 * penalty

    x0 = []
    for h in inner_hexes:
        x0.extend([h[0], h[1], h[2]])

    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': tol,
                                   'fatol': tol, 'adaptive': True})
        if result.fun < 100.0: # Check if penalty is low
            polished = []
            for i in range(N):
                polished.append([result.x[i * 3], result.x[i * 3 + 1], result.x[i * 3 + 2]])
            return polished
    except Exception:
        pass
    return inner_hexes

def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Initial placement
    sqrt3 = math.sqrt(3)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            candidates.append((x * x + y * y, x, y))
    candidates.sort(key=lambda t: t[0])

    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break

    inner = [[pos[0], pos[1], 30.0] for pos in positions]

    # Initial enclosing hexagon
    if time.time() > _DEADLINE - 10.0:
        xs = [h[0] for h in inner]
        ys = [h[1] for h in inner]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_dist = 0.0
        for h in inner:
            d = math.sqrt((h[0] - cx) ** 2 + (h[1] - cy) ** 2)
            if d > max_dist:
                max_dist = d
        side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
        return inner, (cx, cy), side, 0.0

    cx, cy, side, angle = _min_enclosing_hex_side(inner)
    best_side = side
    best_inner = [h[:] for h in inner]
    best_center = (cx, cy)
    best_angle = angle

    # Continuous inner optimization with fast surrogate
    if time.time() < _DEADLINE - 40.0:
        polished = _nelder_mead_polish(inner, max_iter=300, tol=1e-5)
        if polished is not inner:
            pcx, pcy, pside, pangle = _min_enclosing_hex_side(polished)
            if pside < best_side - 1e-7:
                best_side = pside
                best_inner = polished
                best_center = (pcx, pcy)
                best_angle = pangle

    # Final high-precision LP polish
    if time.time() < _DEADLINE - 20.0:
        lo = max(0.0, best_angle - 0.5)
        hi = min(30.0, best_angle + 0.5)
        n_steps = 81
        for i in range(n_steps):
            deg = lo + (hi - lo) * i / (n_steps - 1)
            if time.time() > _DEADLINE - 20.0:
                break
            r = _lp_min_enclosing(best_inner, deg)
            if r and r[2] < best_side - 1e-9:
                best_side, best_center, best_angle = r[2], (r[0], r[1]), deg

    # Optional second polish pass
    if time.time() < _DEADLINE - 25.0:
        polished2 = _nelder_mead_polish(best_inner, max_iter=100, tol=1e-6)
        if polished2 is not best_inner:
            pcx, pcy, pside, pangle = _min_enclosing_hex_side(polished2)
            if pside < best_side - 1e-8:
                best_side = pside
                best_inner = polished2
                best_center = (pcx, pcy)
                best_angle = pangle
                lo = max(0.0, best_angle - 0.3)
                hi = min(30.0, best_angle + 0.3)
                for i in range(41):
                    deg = lo + (hi - lo) * i / 40
                    if time.time() > _DEADLINE - 15.0:
                        break
                    r = _lp_min_enclosing(best_inner, deg)
                    if r and r[2] < best_side - 1e-9:
                        best_side, best_center, best_angle = r[2], (r[0], r[1]), deg

    return best_inner, (best_center[0], best_center[1]), best_side, best_angle

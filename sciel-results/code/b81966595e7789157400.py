"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = math.radians(angle_deg)
    return np.array([[cx + side * math.cos(a + i * math.pi / 3),
                      cy + side * math.sin(a + i * math.pi / 3)] for i in range(6)])

def _polygon_in_hexagon(verts, ocx, ocy, oside, oangle_deg):
    """Check if all vertices of polygon are inside the outer hexagon."""
    oa = math.radians(oangle_deg)
    apothem = oside * math.sqrt(3) / 2.0
    for i in range(6):
        theta = oa + i * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        for v in verts:
            dx = v[0] - ocx
            dy = v[1] - ocy
            d = dx * nx + dy * ny
            if d > apothem + 1e-9:
                return False
    return True

def _hexagons_disjoint(v1, v2, tol=1e-9):
    """Check if two convex hexagons are disjoint using SAT."""
    for poly in (v1, v2):
        for i in range(6):
            j = (i + 1) % 6
            edge = poly[j] - poly[i]
            nx = -edge[1]
            ny = edge[0]
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-12:
                continue
            nx /= norm
            ny /= norm
            proj1 = v1[:, 0] * nx + v1[:, 1] * ny
            proj2 = v2[:, 0] * nx + v2[:, 1] * ny
            if proj1.max() < proj2.min() + tol or proj2.max() < proj1.min() + tol:
                return True
    return False

def _compute_penalty(params):
    """Compute objective: outer side length + penalties for constraint violations."""
    inner_params = params[:36].reshape(12, 3)
    ocx, ocy, oside, oangle = params[36], params[37], params[38], params[39]

    if oside < 1.5:
        return 1e5 + (1.5 - oside) * 1e4

    penalty = 0.0
    verts_list = [_hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
                  for i in range(12)]

    # Containment penalty
    oa = math.radians(oangle)
    apothem = oside * math.sqrt(3) / 2.0
    for i in range(12):
        for j in range(6):
            theta = oa + j * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            for v in verts_list[i]:
                dx = v[0] - ocx
                dy = v[1] - ocy
                d = dx * nx + dy * ny
                if d > apothem:
                    penalty += (d - apothem) ** 2

    # Disjointness penalty
    for i in range(12):
        for j in range(i + 1, 12):
            if not _hexagons_disjoint(verts_list[i], verts_list[j]):
                min_dist = float('inf')
                for v1 in verts_list[i]:
                    for v2 in verts_list[j]:
                        dd = (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2
                        if dd < min_dist:
                            min_dist = dd
                min_dist = math.sqrt(min_dist)
                if min_dist < 0.01:
                    penalty += (0.01 - min_dist) ** 2 * 1e5

    return oside + penalty

def _is_valid(inner, ocx, ocy, oside, oangle):
    """Validate the configuration."""
    if oside < 1.0:
        return False
    verts_list = [_hex_vertices(inner[i][0], inner[i][1], inner[i][2]) for i in range(12)]
    for i in range(12):
        if not _polygon_in_hexagon(verts_list[i], ocx, ocy, oside, oangle):
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _hexagons_disjoint(verts_list[i], verts_list[j]):
                return False
    return True

def construct():
    global _START_TIME
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        inner = [[0.0, 0.0, 0.0] for _ in range(12)]
        return inner, (0.0, 0.0), 15.0, 0.0

    sqrt3 = math.sqrt(3)

    # Initial configuration: hexagonal lattice, 4 rows x 3 cols
    positions = []
    for row in range(4):
        for col in range(3):
            x = col * 1.5 + (0.75 if row % 2 == 1 else 0.0)
            y = row * sqrt3 / 2
            positions.append([x, y, 0.0])

    # Center
    cx = sum(p[0] for p in positions) / 12
    cy = sum(p[1] for p in positions) / 12
    for p in positions:
        p[0] -= cx
        p[1] -= cy

    max_r = max(math.sqrt(p[0] ** 2 + p[1] ** 2) for p in positions) + 1.5
    oside_init = max_r / (sqrt3 / 2) + 0.5

    params0 = np.zeros(40)
    for i in range(12):
        params0[i * 3] = positions[i][0]
        params0[i * 3 + 1] = positions[i][1]
        params0[i * 3 + 2] = positions[i][2]
    params0[36] = 0.0
    params0[37] = 0.0
    params0[38] = oside_init
    params0[39] = 0.0

    best_params = params0.copy()
    best_value = _compute_penalty(params0)

    # Nelder-Mead optimization
    remaining = _TIME_LIMIT - (time.time() - _START_TIME) - 5
    if remaining > 2:
        try:
            result = minimize(_compute_penalty, params0, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5,
                                       'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x
        except Exception:
            pass

    # Extract result
    inner = []
    for i in range(12):
        inner.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]),
                      float(best_params[i * 3 + 2])])
    ocx = float(best_params[36])
    ocy = float(best_params[37])
    oside = float(best_params[38])
    oangle = float(best_params[39])

    # Validate
    if not _is_valid(inner, ocx, ocy, oside, oangle):
        # Try a larger safe hexagon
        all_verts = []
        for i in range(12):
            all_verts.append(_hex_vertices(inner[i][0], inner[i][1], inner[i][2]))
        max_dist = 0
        for vlist in all_verts:
            for v in vlist:
                d = math.sqrt(v[0] ** 2 + v[1] ** 2)
                if d > max_dist:
                    max_dist = d
        safe_oside = max_dist / (sqrt3 / 2) + 0.1
        return inner, (0.0, 0.0), safe_oside, 0.0

    return inner, (ocx, ocy), oside, oangle

# EVOLVE-BLOCK-END

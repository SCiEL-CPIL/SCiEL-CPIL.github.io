import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _min_outer_side(inner_hexes, outer_center, outer_angle_deg):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    ocx, ocy = outer_center
    oa = np.radians(outer_angle_deg)
    max_proj = 0.0
    for hx, hy, ha in inner_hexes:
        verts = _hex_vertices(hx, hy, 1.0, ha)
        for v in verts:
            dx = v[0] - ocx
            dy = v[1] - ocy
            for k in range(6):
                ang = oa + k * np.pi / 3
                proj = dx * np.cos(ang) + dy * np.sin(ang)
                if proj > max_proj:
                    max_proj = proj
    return 2.0 * max_proj / np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    # Initial configuration: compact hexagonal cluster
    d = np.sqrt(3)  # center distance for edge-to-edge hexagons
    positions = [(0.0, 0.0)]
    for i in range(6):
        a = i * np.pi / 3
        positions.append((d * np.cos(a), d * np.sin(a)))
    for i in range(6):
        a = i * np.pi / 3
        positions.append((2 * d * np.cos(a), 2 * d * np.sin(a)))
        if len(positions) >= N:
            break
    positions = positions[:N]

    inner = [[p[0], p[1], 0.0] for p in positions]

    def objective(params):
        n_inner = N
        inner_params = params[:3 * n_inner]
        outer_params = params[3 * n_inner:]
        ocx, ocy, oa = outer_params
        hexes = []
        for i in range(n_inner):
            x = inner_params[3 * i]
            y = inner_params[3 * i + 1]
            a = inner_params[3 * i + 2]
            hexes.append((x, y, a))
        return _min_outer_side(hexes, (ocx, ocy), oa)

    x0 = []
    for hx, hy, ha in inner:
        x0.extend([hx, hy, ha])
    x0.extend([0.0, 0.0, 0.0])

    best_result = None
    best_val = objective(x0)

    try:
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'maxfun': 8000})
        if result.fun < best_val:
            best_val = result.fun
            best_result = result.x
        elif best_result is None:
            best_result = x0
    except Exception:
        best_result = x0 if best_result is None else best_result

    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        for restart in range(8):
            if time.time() > deadline - 25.0:
                break
            rng = np.random.default_rng(restart * 137 + 1)
            x0r = np.array(x0, dtype=float)
            x0r[:3 * N] += rng.normal(0, 0.8, size=3 * N)
            x0r[3 * N + 2] = rng.uniform(0, 60)

            try:
                result = minimize(objective, x0r, method='L-BFGS-B', options={'maxiter': 150, 'maxfun': 4000})
                if result.fun < best_val:
                    best_val = result.fun
                    best_result = result.x
            except Exception:
                pass

    if best_result is None:
        best_result = x0

    inner_out = []
    for i in range(N):
        x = best_result[3 * i]
        y = best_result[3 * i + 1]
        a = best_result[3 * i + 2]
        inner_out.append([float(x), float(y), float(a)])

    ocx = float(best_result[3 * N])
    ocy = float(best_result[3 * N + 1])
    oa = float(best_result[3 * N + 2])
    side = float(_min_outer_side(inner_out, (ocx, ocy), oa))

    return inner_out, (ocx, ocy), side, oa
# EVOLVE-BLOCK-END

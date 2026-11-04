import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s
APOTHEM_UNIT = math.sqrt(3) / 2.0
EPS = 1e-9

# Pre-compute unit hexagon vertices and edge normals (for angle 0)
_UNIT_VERTS = []
_UNIT_NORMALS = []
for k in range(6):
    theta = math.radians(60 * k)
    _UNIT_VERTS.append((math.cos(theta), math.sin(theta)))
    theta_n = math.radians(30 + 60 * k)
    _UNIT_NORMALS.append((math.cos(theta_n), math.sin(theta_n)))

# Convert to numpy arrays for speed
_UNIT_VERTS_ARR = np.array(_UNIT_VERTS)  # (6, 2)
_UNIT_NORMALS_ARR = np.array(_UNIT_NORMALS)  # (6, 2)


def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts


def _get_axes(verts):
    """Return edge normal axes for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes


def _project(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)


def _sat_overlap_depth(verts1, verts2, axes1, axes2):
    """Returns the overlap depth if polygons overlap, 0 otherwise."""
    min_overlap = float('inf')
    for axis in axes1:
        min1, max1 = _project(verts1, axis)
        min2, max2 = _project(verts2, axis)
        if max1 <= min2 or max2 <= min1:
            return 0.0
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    for axis in axes2:
        min1, max1 = _project(verts1, axis)
        min2, max2 = _project(verts2, axis)
        if max1 <= min2 or max2 <= min1:
            return 0.0
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    return min_overlap


def _objective(params):
    """
    Objective function: minimize outer hexagon side length + penalty for overlaps.
    params: [x0..x11, y0..y11, angle0..angle11]
    """
    xs = params[0:N]
    ys = params[N:2*N]
    angles = params[2*N:3*N]

    verts_list = []
    axes_list = []

    for i in range(N):
        v = _hex_vertices(xs[i], ys[i], angles[i])
        verts_list.append(v)
        axes_list.append(_get_axes(v))

    # Calculate required outer side length assuming outer is centered at (0,0) and angle 0
    # We'll optimize the outer hexagon separately later.
    max_proj = 0.0
    for v in verts_list:
        for (vx, vy) in v:
            for k in range(6):
                theta_n = math.radians(30 + 60 * k)
                nx = math.cos(theta_n)
                ny = math.sin(theta_n)
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj

    side_length = max_proj / APOTHEM_UNIT

    # Calculate overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(verts_list[i], verts_list[j], axes_list[i], axes_list[j])
            if depth > 0:
                penalty += depth * depth

    return side_length + 100.0 * penalty


def _min_enclosing_hex_side(xs, ys, angles, cx, cy, angle_deg):
    """
    Compute the side length of the smallest regular hexagon with center (cx, cy) and rotation angle_deg
    that encloses all inner hexagons.
    """
    a = math.radians(angle_deg)
    max_proj = 0.0
    for i in range(N):
        v = _hex_vertices(xs[i], ys[i], angles[i])
        for (vx, vy) in v:
            for k in range(6):
                theta_n = a + math.radians(30 + 60 * k)
                nx = math.cos(theta_n)
                ny = math.sin(theta_n)
                proj = (vx - cx) * nx + (vy - cy) * ny
                if proj > max_proj:
                    max_proj = proj
    return max_proj / APOTHEM_UNIT


def _optimize_outer_hexagon(xs, ys, angles):
    """
    Find the best outer hexagon (center and angle) for the given inner configuration.
    Returns (cx, cy, angle, side).
    """
    # We can optimize over (cx, cy, angle).
    # Due to symmetry, we can restrict angle to [0, 60) and center to a small region.
    
    def _outer_obj(params):
        cx, cy, angle = params
        return _min_enclosing_hex_side(xs, ys, angles, cx, cy, angle)

    # Initial guess: centroid of inner hexagons
    cx0 = np.mean(xs)
    cy0 = np.mean(ys)
    angle0 = 0.0

    # Try a few starting points for the outer optimization
    best_side = float('inf')
    best_params = None

    # Grid search over angles
    for angle_start in np.linspace(0, 60, 13):
        x0 = [cx0, cy0, angle_start]
        try:
            res = minimize(_outer_obj, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        return cx0, cy0, angle0, _min_enclosing_hex_side(xs, ys, angles, cx0, cy0, angle0)

    cx_best, cy_best, angle_best = best_params
    return cx_best, cy_best, angle_best, best_side


def construct():
    t0 = time.time()

    def make_flower():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        r2 = r1 * 1.732
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
        return np.array([p[0] for p in pts] + [p[1] for p in pts] + [p[2] for p in pts])

    def make_lattice():
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
        return np.array([p[0] for p in pts] + [p[1] for p in pts] + [p[2] for p in pts])

    initializers = [make_flower, make_lattice]

    best_side = float('inf')
    best_params = None
    best_outer = (0.0, 0.0, 0.0, 0.0)  # cx, cy, angle, side

    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 60:
            break

        x0 = init_func()

        # Run Nelder-Mead
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})

            if res.fun < best_side:
                # Check validity of the result
                params = res.x
                xs = params[0:N]
                ys = params[N:2*N]
                angles = params[2*N:3*N]

                verts_list = [_hex_vertices(xs[i], ys[i], angles[i]) for i in range(N)]
                axes_list = [_get_axes(v) for v in verts_list]

                is_valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        depth = _sat_overlap_depth(verts_list[i], verts_list[j], axes_list[i], axes_list[j])
                        if depth > 1e-6:
                            is_valid = False
                            break
                    if not is_valid:
                        break

                if is_valid:
                    # Calculate precise side length
                    max_proj = 0.0
                    for v in verts_list:
                        for (vx, vy) in v:
                            for k in range(6):
                                theta_n = math.radians(30 + 60 * k)
                                nx = math.cos(theta_n)
                                ny = math.sin(theta_n)
                                proj = vx * nx + vy * ny
                                if proj > max_proj:
                                    max_proj = proj
                    side = max_proj / APOTHEM_UNIT
                    if side < best_side:
                        best_side = side
                        best_params = params
        except Exception as e:
            pass

    # If no valid solution found, use a fallback
    if best_params is None:
        best_params = make_flower()
        best_side = _objective(best_params)

    # Final refinement: optimize outer hexagon
    xs = best_params[0:N]
    ys = best_params[N:2*N]
    angles = best_params[2*N:3*N]

    cx_best, cy_best, angle_best, side_best = _optimize_outer_hexagon(xs, ys, angles)

    # Return the configuration
    inner = []
    for i in range(N):
        x = float(best_params[i])
        y = float(best_params[N + i])
        a = float(best_params[2*N + i])
        inner.append([x, y, a])

    return inner, (cx_best, cy_best), side_best, angle_best


if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)

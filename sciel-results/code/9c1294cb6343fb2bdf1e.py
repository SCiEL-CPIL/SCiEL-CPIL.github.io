import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 148.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum enclosing regular hexagon side length using linear programming.
    Variables: [cx, cy, r] where r is the apothem (distance from center to each side).
    Side length = 2*r/sqrt(3).
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6,2)

    M = verts.shape[0]
    c_obj = np.array([0.0, 0.0, 1.0])

    A_rows = []
    b_rows = []
    for k in range(6):
        nkx, nky = normals[k]
        nv = verts[:, 0] * nkx + verts[:, 1] * nky
        A_block = np.column_stack([
            np.full(M, -nkx),
            np.full(M, -nky),
            np.full(M, -1.0)
        ])
        b_block = -nv
        A_rows.append(A_block)
        b_rows.append(b_block)

    A_ub = np.vstack(A_rows)
    b_ub = np.concatenate(b_rows)

    A_ub = np.vstack([A_ub, np.array([[0.0, 0.0, -1.0]])])
    b_ub = np.append(b_ub, 0.0)

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')

    if res.status != 0:
        return None

    cx, cy, r = res.x
    if r < 0:
        r = 0.0
    side = r * 2.0 / _SQRT3
    return cx, cy, side


def _compute_side_for_config(positions, angles, outer_angle_deg):
    """Compute enclosing hex side for given inner positions/angles and outer angle."""
    all_verts = np.vstack([_hex_vertices(px, py, ang) for px, py, ang in zip(positions, angles)])
    res = _min_enclosing_hex_side_lp(all_verts, outer_angle_deg)
    if res is None:
        return float('inf'), (0.0, 0.0)
    cx, cy, side = res
    return side, (cx, cy)


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.01
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 1:
        # Row-based: 3-4-4 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        # 2-4-2-3 pattern
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * sqrt3 * scale - sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        # 3-5-3 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        # Center + 6 ring + 4 more, symmetric
        positions = []
        positions.append((0, 0))
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        for ti, tj in ring1:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        ring2 = [(2, 0), (1, 1), (-1, 2), (-2, 0), (-1, -1), (0, -2)]
        for ti, tj in ring2[:4]:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        # 2-3-3-3 pattern
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * sqrt3 * scale - sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        # 4-4-3 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        # 5-6 pattern: two rows of 5 and 6
        positions = []
        y1 = -0.75 * scale
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.75 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        # 6-5 pattern
        positions = []
        y1 = -0.75 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.75 * scale
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 10:
        # Diamond/rhombus arrangement
        positions = []
        positions.append((0, 0))
        positions.append((sqrt3 * scale, 0))
        positions.append((-sqrt3 * scale, 0))
        positions.append((0, 3.0 * scale))
        positions.append((0, -3.0 * scale))
        positions.append((sqrt3/2 * scale, 1.5 * scale))
        positions.append((-sqrt3/2 * scale, 1.5 * scale))
        positions.append((sqrt3/2 * scale, -1.5 * scale))
        positions.append((-sqrt3/2 * scale, -1.5 * scale))
        positions.append((sqrt3 * scale, 3.0 * scale))
        positions.append((-sqrt3 * scale, 3.0 * scale))
        return positions[:N]
    elif layout_idx == 11:
        # Tighter packing: reduce scale
        scale2 = 1.0
        u2 = (sqrt3 * scale2, 0.0)
        v2 = (sqrt3 / 2 * scale2, 1.5 * scale2)
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti * u2[0] + tj * v2[0]
            y = ti * u2[1] + tj * v2[1]
            positions.append((x, y))
        return positions[:N]
    else:
        # Fallback: dense grid sorted by distance from origin
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y, i, j))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return [(x, y) for x, y, i, j in coords_sorted[:N]]


def _polish(positions, angles, outer_angle_deg, center, max_time=20.0):
    """
    Polish the configuration using SLSQP to jointly optimize inner positions,
    inner angles, outer center, and outer angle.
    """
    t0 = time.time()
    
    def objective(params):
        if time.time() - t0 > max_time:
            return 1e10
        # params: [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, cx, cy, outer_angle]
        # N*3 + 3 = 36 params
        inner_params = params[:N*3]
        cx = params[N*3]
        cy = params[N*3+1]
        oa = params[N*3+2]
        
        # Compute all vertices
        all_verts = np.empty((N*6, 2))
        for i in range(N):
            px = inner_params[i*3]
            py = inner_params[i*3+1]
            pa = math.degrees(inner_params[i*3+2])
            all_verts[i*6:(i+1)*6] = _hex_vertices(px, py, pa)
        
        # Compute enclosing hex side with given outer angle and center
        base_angle = math.radians(oa + 30)
        angles_out = base_angle + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
        
        # For each vertex and normal: n_k . (v - c) <= r
        # r = max over all (k, v) of n_k . (v - c)
        max_proj = 0.0
        for k in range(6):
            proj = all_verts[:, 0] * normals[k, 0] + all_verts[:, 1] * normals[k, 1]
            proj -= cx * normals[k, 0] + cy * normals[k, 1]
            if proj.max() > max_proj:
                max_proj = proj.max()
        
        if max_proj < 0:
            max_proj = 0.0
        
        side = max_proj * 2.0 / _SQRT3
        return side
    
    # Initial params
    x0 = np.zeros(N*3 + 3)
    for i in range(N):
        x0[i*3] = positions[i][0]
        x0[i*3+1] = positions[i][1]
        x0[i*3+2] = math.radians(angles[i])
    x0[N*3] = center[0]
    x0[N*3+1] = center[1]
    x0[N*3+2] = math.radians(outer_angle_deg)
    
    try:
        res = minimize(objective, x0, method='SLSQP', options={'maxiter': 50, 'ftol': 1e-8})
        if res.success or res.fun < float('inf'):
            final_params = res.x
            # Extract results
            final_positions = [(final_params[i*3], final_params[i*3+1]) for i in range(N)]
            final_angles = [math.degrees(final_params[i*3+2]) for i in range(N)]
            final_cx = final_params[N*3]
            final_cy = final_params[N*3+1]
            final_oa = math.degrees(final_params[N*3+2])
            final_side = res.fun
            return final_positions, final_angles, final_cx, final_cy, final_oa, final_side
    except Exception:
        pass
    
    return None


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_positions = None
    best_angles = None

    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]

    # Phase 1: For each layout, sweep outer angles and try a few inner angle configs
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 30.0:
            break

        # Try different inner angle configurations
        inner_angle_options = [30.0, 0.0, 15.0, 45.0]
        
        for inner_angle in inner_angle_options:
            if time.time() > _DEADLINE - 30.0:
                break

            angles = [inner_angle] * N
            all_vertices = np.vstack([_hex_vertices(px, py, ang) for px, py, ang in zip(positions, angles)])

            best_local_side = float('inf')
            best_local_center = (0.0, 0.0)
            best_local_angle = 0.0

            # Sweep over outer angles (0 to 30 degrees, step 1.0 for speed)
            for outer_angle_deg in np.arange(0, 30.0, 1.0):
                if time.time() > _DEADLINE - 30.0:
                    break

                res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_local_side:
                        best_local_side = side
                        best_local_center = (cx, cy)
                        best_local_angle = outer_angle_deg

            if best_local_side < best_side:
                best_side = best_local_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_local_center
                best_outer_angle = best_local_angle
                best_positions = positions
                best_angles = angles

    # Phase 2: Fine sweep on best configuration
    if best_positions is not None and time.time() < _DEADLINE - 40.0:
        angles = best_angles
        all_vertices = np.vstack([_hex_vertices(px, py, ang) for px, py, ang in zip(best_positions, angles)])

        # Fine sweep around best outer angle
        for outer_angle_deg in np.arange(best_outer_angle - 5.0, best_outer_angle + 5.0, 0.25):
            if time.time() > _DEADLINE - 40.0:
                break
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_outer_angle = outer_angle_deg

        # Try different inner angles for the best layout
        for inner_angle in np.arange(0, 30, 5):
            if time.time() > _DEADLINE - 40.0:
                break
            angles_try = [inner_angle] * N
            all_v = np.vstack([_hex_vertices(px, py, ang) for px, py, ang in zip(best_positions, angles_try)])
            for outer_angle_deg in np.arange(0, 30.0, 1.0):
                if time.time() > _DEADLINE - 40.0:
                    break
                res = _min_enclosing_hex_side_lp(all_v, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_side:
                        best_side = side
                        best_center = (cx, cy)
                        best_outer_angle = outer_angle_deg
                        best_inner = [[pos[0], pos[1], inner_angle] for pos in best_positions]
                        best_angles = angles_try

    # Phase 3: Polish with SLSQP
    if best_positions is not None and time.time() < _DEADLINE - 15.0:
        polish_result = _polish(best_positions, best_angles, best_outer_angle, best_center, max_time=25.0)
        if polish_result is not None:
            pp, pa, pcx, pcy, poa, pside = polish_result
            if pside < best_side:
                best_side = pside
                best_inner = [[pp[i][0], pp[i][1], pa[i]] for i in range(N)]
                best_center = (pcx, pcy)
                best_outer_angle = poa

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle

"""Combined program: LP outer-hexagon minimization + lattice layouts + high-precision angle polish."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin within 150s limit
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    angles = angle_rad + np.arange(6) * math.pi / 3
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons as a numpy array (N*6, 2)."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        verts[idx * 6: idx * 6 + 6] = _hex_vertices(ix, iy, iangle)
    return verts

def _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg):
    """For a fixed outer hexagon angle, find optimal center and side length using LP.
    Returns (cx, cy, side) or None on failure.
    """
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return (0.0, 0.0, 0.0)
    
    # Normals to the 6 sides of the outer hexagon
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    
    # Vectorized constraint construction
    v_dot_n = all_vertices @ normals.T  # (n_pts, 6)
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        
        # Row 1: -nx*cx - ny*cy - R <= -v_dot
        A_rows.append(np.column_stack([np.full(n_pts, -nx), np.full(n_pts, -ny), np.full(n_pts, -1.0)]))
        b_vals.append(-v_dots)
        
        # Row 2: nx*cx + ny*cy - R <= v_dot
        A_rows.append(np.column_stack([np.full(n_pts, nx), np.full(n_pts, ny), np.full(n_pts, -1.0)]))
        b_vals.append(v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    # R >= 0
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3 / 2)
            return (cx, cy, side)
    except Exception:
        pass
    
    # Fallback: use centroid and compute required R
    cx_mean = all_vertices[:, 0].mean()
    cy_mean = all_vertices[:, 1].mean()
    centered = all_vertices - np.array([cx_mean, cy_mean])
    projs = centered @ normals.T  # (n_pts, 6)
    R_opt = projs.max()
    if R_opt < 0:
        R_opt = 0.0
    side = R_opt / (_SQRT3 / 2)
    return (cx_mean, cy_mean, side)

def _min_enclosing_hex_side(inner_hexes, all_vertices=None):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    if all_vertices is None:
        all_vertices = _compute_all_vertices(inner_hexes)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # First pass: coarse sweep
    for angle_deg in np.arange(0, 60, 2.0):
        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Second pass: fine sweep
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.25):
            res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)
    
    # Third pass: even finer sweep
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), 0.05):
            res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)
    
    # Fourth pass: Nelder-Mead polish
    if best_side < float('inf') and time.time() < _DEADLINE - 15.0:
        def objective(params):
            ang_p = params[2] % 60.0
            res = _min_enclosing_hex_side_lp(all_vertices, float(ang_p))
            if res:
                return res[2]
            return 1e6
        
        x0 = np.array([best_center[0], best_center[1], best_angle])
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            if res.success:
                ang_nm = float(res.x[2] % 60.0)
                cx_v, cy_v, side_v = _min_enclosing_hex_side_lp(all_vertices, ang_nm)
                if side_v < best_side:
                    best_side = side_v
                    best_center = (cx_v, cy_v)
                    best_angle = ang_nm
        except Exception:
            pass
    
    # Fifth pass: High-precision grid search
    if best_side < float('inf') and time.time() < _DEADLINE - 5.0:
        window = 0.1  # degrees
        steps = 200
        angles_hp = np.linspace(best_angle - window, best_angle + window, steps)
        for ang in angles_hp:
            ang_mod = ang % 60.0
            if ang_mod < 0:
                ang_mod += 60.0
            res = _min_enclosing_hex_side_lp(all_vertices, float(ang_mod))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(ang_mod)
    
    return best_center[0], best_center[1], best_side, best_angle

def _build_positions(layout_idx, scale=1.0000001):
    """Generate 11 positions for given layout index."""
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 more
        positions = [(0, 0)]
        for k in range(6):
            positions.append((k, 0))
        # Add 4 more positions
        positions.append((1, 1))
        positions.append((-1, 1))
        positions.append((1, -1))
        positions.append((-1, -1))
        
    elif layout_idx == 1:
        # Linear arrangement
        positions = [(i, 0) for i in range(-5, 6)]
        
    elif layout_idx == 2:
        # 2-row arrangement
        positions = [(i, 0) for i in range(-3, 4)]
        positions += [(i, 1) for i in range(-3, 1)]
        
    elif layout_idx == 3:
        # 3-row staggered
        positions = [(i, 0) for i in range(-2, 3)]
        positions += [(i, 1) for i in range(-2, 3)]
        positions += [(i, 2) for i in range(-2, 3)]
        
    else:
        # Default: center + ring
        positions = [(0, 0)]
        for k in range(6):
            positions.append((k, 0))
        positions.append((1, 1))
        positions.append((-1, 1))
        positions.append((1, -1))
    
    # Convert lattice coords to Cartesian
    cart_positions = []
    for i, j in positions:
        x = i * u[0] + j * v[0]
        y = i * u[1] + j * v[1]
        cart_positions.append((x, y))
    
    return cart_positions

def _on_segment(a, b, p, eps=1e-12):
    """Check if point p is on segment a-b."""
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    c1 = cross(p3, p4, p1)
    c2 = cross(p3, p4, p2)
    c3 = cross(p1, p2, p3)
    c4 = cross(p1, p2, p4)

    if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
       ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
        return True

    if abs(c1) < 1e-12 and _on_segment(p3, p4, p1):
        return True
    if abs(c2) < 1e-12 and _on_segment(p3, p4, p2):
        return True
    if abs(c3) < 1e-12 and _on_segment(p1, p2, p3):
        return True
    if abs(c4) < 1e-12 and _on_segment(p1, p2, p4):
        return True
    return False

def _point_in_convex_polygon(p, polygon, strict=False):
    """Check if point p is inside convex polygon."""
    px, py = p
    n = len(polygon)
    sign = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if strict:
            if cross < 0:
                return False
        else:
            if cross < -1e-12:
                return False
        if cross > 0:
            if sign < 0:
                return False
            sign = 1
        elif cross < 0:
            if sign > 0:
                return False
            sign = -1
    return True

def _hexagons_overlap(verts1, verts2):
    """Check if two convex hexagons overlap (interior intersection)."""
    if _point_in_convex_polygon(verts1[0], verts2, strict=True):
        return True
    if _point_in_convex_polygon(verts2[0], verts1, strict=True):
        return True

    for k in range(6):
        a1 = verts1[k]
        b1 = verts1[(k + 1) % 6]
        for l in range(6):
            a2 = verts2[l]
            b2 = verts2[(l + 1) % 6]
            if _segments_intersect(a1, b1, a2, b2):
                return True
    return False

def _check_disjoint(inner_hexes, tol=1e-9):
    """Check that all inner hexagons are disjoint."""
    n = len(inner_hexes)
    all_verts = []
    for ix, iy, iangle in inner_hexes:
        all_verts.append(_hex_vertices(ix, iy, iangle))

    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap(all_verts[i], all_verts[j]):
                return False
    return True

def construct():
    if time.time() > _DEADLINE - 5.0:
        # Emergency fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Phase 1: Try multiple lattice layouts
    layouts_to_try = [0, 1, 2, 3]
    scales = [1.0000001, 1.00001, 1.0001]
    
    for layout in layouts_to_try:
        if time.time() > _DEADLINE - 30.0:
            break
        
        for scale in scales:
            if time.time() > _DEADLINE - 20.0:
                break
            
            positions = _build_positions(layout, scale)
            if len(positions) != N:
                continue
            
            for inner_angle in [0.0, 30.0]:
                if time.time() > _DEADLINE - 10.0:
                    break
                
                inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                
                # Quick check for disjointness
                if not _check_disjoint(inner):
                    continue
                
                # Find best enclosing hexagon
                all_vertices = _compute_all_vertices(inner)
                cx, cy, side, angle = _min_enclosing_hex_side(inner, all_vertices)
                
                if side < best_side:
                    best_side = side
                    best_inner = inner
                    best_center = (cx, cy)
                    best_angle = angle

    # Phase 2: Local refinement of positions
    if best_inner is not None and time.time() < _DEADLINE - 20.0:
        n = len(best_inner)
        x0 = []
        for ix, iy, ia in best_inner:
            x0.extend([ix, iy, ia])
        x0 = np.array(x0, dtype=float)

        def objective(params):
            if time.time() > _DEADLINE - 5.0:
                return 1e9
            pos = [(params[3*i], params[3*i+1]) for i in range(n)]
            ang = [params[3*i+2] for i in range(n)]
            inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(n)]

            if not _check_disjoint(inner):
                return 1e6

            try:
                all_vertices = _compute_all_vertices(inner)
                _, _, side, _ = _min_enclosing_hex_side(inner, all_vertices)
                return side
            except Exception:
                return 1e9

        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < 1e5:
                new_inner = []
                for i in range(n):
                    new_inner.append([res.x[3*i], res.x[3*i+1], res.x[3*i+2]])
                if _check_disjoint(new_inner):
                    all_vertices = _compute_all_vertices(new_inner)
                    cx, cy, side, angle = _min_enclosing_hex_side(new_inner, all_vertices)
                    if side < best_side - 1e-10:
                        best_side = side
                        best_inner = new_inner
                        best_center = (cx, cy)
                        best_angle = angle
        except Exception:
            pass

    # Phase 3: Final high-precision polish on the outer angle
    if best_inner is not None and time.time() < _DEADLINE - 10.0:
        all_vertices = _compute_all_vertices(best_inner)
        
        # Very fine grid search around current best angle
        window = 0.05
        steps = 500
        angles_hp = np.linspace(best_angle - window, best_angle + window, steps)
        
        for ang in angles_hp:
            if time.time() > _DEADLINE - 5.0:
                break
            ang_mod = ang % 60.0
            if ang_mod < 0:
                ang_mod += 60.0
            res = _min_enclosing_hex_side_lp(all_vertices, float(ang_mod))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(ang_mod)
        
        # Nelder-Mead final polish
        if time.time() < _DEADLINE - 5.0:
            def polish_obj(x):
                if time.time() > _DEADLINE - 3.0:
                    return 1e9
                _, _, side, _ = _min_enclosing_hex_side_lp(all_vertices, float(x[0]))
                return side

            try:
                res_polish = minimize(polish_obj, np.array([best_angle]), method='Nelder-Mead',
                                     options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 300, 'adaptive': True})
                if res_polish.fun < best_side - 1e-12:
                    best_angle = float(res_polish.x[0] % 60.0)
                    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, best_angle)
                    best_side = side
                    best_center = (cx, cy)
            except Exception:
                pass

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_angle

"""Combined program: LP-based outer hexagon + L-BFGS-B inner optimization with smooth surrogate."""
import math
import time
import numpy as np
import itertools
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 148.0
_SQRT3 = math.sqrt(3)

def _hex_vertices_array(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg.
    Returns numpy array shape (6, 2)."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons as a numpy array (N*6, 2)."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + math.pi * k / 3
            verts[idx * 6 + k, 0] = ix + math.cos(a)
            verts[idx * 6 + k, 1] = iy + np.sin(a)
    return verts

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex direction at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    v_dot_n = verts @ normals.T

    A_rows = []
    b_vals = []

    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]

        # (v - c).n <= R  =>  -nx*cx - ny*cy - R <= -v.n
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)

        # (c - v).n <= R  =>  nx*cx + ny*cy - R <= v.n
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
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
            return cx, cy, side
    except Exception:
        pass
    return None

def _smooth_surrogate(params, outer_angle_deg, alpha=50.0):
    """
    Smooth approximation of the maximum width of the point cloud defined by params.
    params: [x1, y1, a1, x2, y2, a2, ...] (3*N parameters)
    Returns a scalar value to be minimized.
    """
    n = len(params) // 3
    verts = np.zeros((n * 6, 2))
    for i in range(n):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        angle_rad = math.radians(a)
        for k in range(6):
            ang = angle_rad + math.pi * k / 3
            verts[i*6 + k, 0] = x + math.cos(ang)
            verts[i*6 + k, 1] = y + math.sin(ang)
    
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    v_dot_n = verts @ normals.T # (M, 6)
    
    # For each normal direction k, compute soft max and soft min
    # U_k = (1/alpha) * log(sum(exp(alpha * v_dot_n[:, k])))
    # L_k = -(1/alpha) * log(sum(exp(-alpha * v_dot_n[:, k])))
    # W_k = U_k - L_k
    # Objective = (1/alpha) * log(sum(exp(alpha * W_k)))
    
    # Vectorized computation
    # v_dot_n shape (M, 6)
    # We need to compute log-sum-exp for each column
    
    # To avoid overflow, subtract max
    max_vdn = np.max(v_dot_n, axis=0) # (6,)
    min_vdn = np.min(v_dot_n, axis=0) # (6,)
    
    # U_k
    exp_vdn = np.exp(alpha * (v_dot_n - max_vdn))
    log_sum_exp_pos = np.log(np.sum(exp_vdn, axis=0))
    U = (log_sum_exp_pos + alpha * max_vdn) / alpha
    
    # L_k
    exp_neg_vdn = np.exp(-alpha * (v_dot_n - min_vdn))
    log_sum_exp_neg = np.log(np.sum(exp_neg_vdn, axis=0))
    L = -(log_sum_exp_neg - alpha * min_vdn) / alpha # Note: - (1/a) log sum exp(-a v) = (1/a) log sum exp(a v) - max... wait.
    # Let's re-derive L_k carefully.
    # min_i v_i = - max_i (-v_i)
    # max_i (-v_i) = (1/a) log sum exp(a * (-v_i))
    # So min_i v_i = - (1/a) log sum exp(-a * v_i)
    
    # Correct L calculation:
    # -v_dot_n shape (M, 6)
    # max_neg = max(-v_dot_n) = -min(v_dot_n)
    # exp_term = exp(alpha * (-v_dot_n - max_neg))
    # log_sum = log(sum(exp_term))
    # max_neg_val = (log_sum + alpha * max_neg) / alpha
    # min_val = - max_neg_val
    
    max_neg = np.max(-v_dot_n, axis=0)
    exp_neg = np.exp(alpha * (-v_dot_n - max_neg))
    log_sum_neg = np.log(np.sum(exp_neg, axis=0))
    max_neg_val = (log_sum_neg + alpha * max_neg) / alpha
    L = -max_neg_val
    
    W = U - L # (6,)
    
    # Final soft max over W
    max_W = np.max(W)
    exp_W = np.exp(alpha * (W - max_W))
    log_sum_W = np.log(np.sum(exp_W))
    obj = (log_sum_W + alpha * max_W) / alpha
    
    return obj

def _optimize_inner_positions(initial_positions, initial_angles, outer_angle_deg, alpha=50.0, max_iter=200):
    """
    Optimize inner hexagon positions and angles using L-BFGS-B on the smooth surrogate.
    Returns optimized positions and angles.
    """
    n = len(initial_positions)
    x0 = []
    for i in range(n):
        x0.extend([initial_positions[i][0], initial_positions[i][1], initial_angles[i]])
    x0 = np.array(x0)
    
    # Bounds: keep within a reasonable box to prevent divergence
    bounds = [(-10, 10), (-10, 10), (0, 60)] * n
    
    try:
        res = minimize(_smooth_surrogate, x0, args=(outer_angle_deg, alpha), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success:
            opt_params = res.x
            opt_positions = []
            opt_angles = []
            for i in range(n):
                opt_positions.append((opt_params[i*3], opt_params[i*3+1]))
                opt_angles.append(opt_params[i*3+2])
            return opt_positions, opt_angles
    except Exception:
        pass
    return initial_positions, initial_angles

def _build_layouts():
    """Generate multiple candidate layouts for the 11 inner hexagons.
    Returns list of (positions_list, inner_angle)."""
    layouts = []
    # Use tight scale to allow L-BFGS to find tighter packing
    scale = 1.0001
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)

    # Layout 0: hexagonal flower + 4 from second ring (compact)
    target = [(0, 0)]
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    target.extend(ring1)
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
    target.extend(ring2_sorted[:4])

    for inner_angle in [0.0, 30.0]:
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        layouts.append((positions[:N], inner_angle))

    # Layout 1: row-based 3-4-4
    positions = []
    y1 = -1.5 * scale
    for k in range(-1, 2):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 3):
        x = k * _SQRT3 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-1, 3):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y3))
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions[:N], inner_angle))

    # Layout 2: 4-3-4 shifted
    positions = []
    y1 = -1.5 * scale
    for k in range(-2, 2):
        x = k * _SQRT3 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 2):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-2, 2):
        x = k * _SQRT3 * scale
        positions.append((x, y3))
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions[:N], inner_angle))

    # Layout 3: greedy nearest-neighbor from origin on triangular lattice
    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            coords.append((x, y))
    coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
    positions = coords_sorted[:N]
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions, inner_angle))

    return layouts

def _refine_layout(positions, inner_angle):
    """
    Refine a specific layout by optimizing inner positions with L-BFGS-B
    and then finding the best outer angle with LP.
    """
    if time.time() > _DEADLINE - 20.0:
        return None
    
    # 1. Optimize inner positions using smooth surrogate
    # Try a few outer angles for the surrogate optimization
    best_opt_side = float('inf')
    best_opt_inner = None
    best_opt_center = None
    best_opt_outer_angle = None
    
    # Initial guess for outer angle is 0
    # We can also try 30
    for trial_outer_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 15.0:
            break
            
        opt_positions, opt_angles = _optimize_inner_positions(positions, [inner_angle]*N, trial_outer_angle)
        
        # Construct inner hexes from optimized positions
        inner_hexes = [[opt_positions[i][0], opt_positions[i][1], opt_angles[i]] for i in range(N)]
        verts = _compute_all_vertices(inner_hexes)
        
        # 2. Find best outer angle with LP for this optimized inner layout
        best_local_side = float('inf')
        best_local_center = None
        best_local_angle = None
        
        # Coarse search
        for angle_deg in np.arange(0, 60, 2.0):
            if time.time() > _DEADLINE - 10.0:
                break
            res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = float(angle_deg)
        
        # Fine search
        if best_local_angle is not None:
            for angle_deg in np.arange(max(0.0, best_local_angle - 2), min(60.0, best_local_angle + 2), 0.1):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
                if res:
                    cx, cy, side = res
                    if side < best_local_side:
                        best_local_side = side
                        best_local_center = (cx, cy)
                        best_local_angle = float(angle_deg)
            
            if best_local_side < best_opt_side:
                best_opt_side = best_local_side
                best_opt_inner = inner_hexes
                best_opt_center = best_local_center
                best_opt_outer_angle = best_local_angle

    return best_opt_side, best_opt_inner, best_opt_center, best_opt_outer_angle

def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    layouts = _build_layouts()

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    for positions, inner_angle in layouts:
        if time.time() > _DEADLINE - 25.0:
            break

        result = _refine_layout(positions, inner_angle)
        if result:
            side, inner, center, outer_angle = result
            if side < best_side:
                best_side = side
                best_inner = inner
                best_center = center
                best_outer_angle = outer_angle

    # Fallback if optimization failed or no layout was processed
    if best_inner is None:
        scale = 1.000001
        u = (_SQRT3 * scale, 0.0)
        v = (_SQRT3 / 2 * scale, 1.5 * scale)
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        best_inner = [[coords_sorted[i][0], coords_sorted[i][1], 0.0] for i in range(N)]
        verts = _compute_all_vertices(best_inner)
        # Quick LP for fallback
        best_fb_side = float('inf')
        best_fb_center = (0,0)
        best_fb_angle = 0
        for angle_deg in np.arange(0, 60, 5.0):
            res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_fb_side:
                    best_fb_side = side
                    best_fb_center = (cx, cy)
                    best_fb_angle = float(angle_deg)
        best_side = best_fb_side
        best_center = best_fb_center
        best_outer_angle = best_fb_angle

    return best_inner, best_center, best_side, best_outer_angle

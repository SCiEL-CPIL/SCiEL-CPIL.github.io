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


def _hex_vertices_batch(positions, angle_deg):
    """Vertices for multiple hexagons at given positions, all with same rotation."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    return positions[:, np.newaxis, :] + base[np.newaxis, :, :]


def _hex_support(angle_deg, theta):
    """Support function of unit hexagon at rotation angle_deg in direction theta."""
    a0 = math.radians(angle_deg)
    diff = (theta - a0) % (2 * math.pi)
    k = round(diff / (math.pi / 3.0))
    d1 = abs(diff - k * math.pi / 3.0)
    d2 = abs(diff - (k - 1) * math.pi / 3.0)
    d3 = abs(diff - (k + 1) * math.pi / 3.0)
    min_d = min(d1, d2, d3)
    return math.cos(min_d)


def _hex_distance_same_angle(d, angle_deg):
    """Minimum separation between two unit hexagons with same rotation angle."""
    a = math.radians(angle_deg)
    dx, dy = d
    min_sep = float('inf')
    s = math.sqrt(3) / 2.0
    for k in range(6):
        theta = a + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(theta), math.sin(theta)
        proj = abs(dx * nx + dy * ny)
        sep = proj - 2.0 * s
        if sep < min_sep:
            min_sep = sep
    return min_sep


def _hex_distance(d, angle1, angle2):
    """Minimum distance between two unit hexagons with different rotations."""
    a1 = math.radians(angle1)
    a2 = math.radians(angle2)
    dx, dy = d
    min_sep = float('inf')
    for k in range(6):
        theta1 = a1 + math.pi/6.0 + k * math.pi/3.0
        nx, ny = math.cos(theta1), math.sin(theta1)
        proj = abs(dx * nx + dy * ny)
        s1 = _hex_support(angle1, theta1)
        s2 = _hex_support(angle2, theta1)
        sep = proj - s1 - s2
        if sep < min_sep:
            min_sep = sep
        
        theta2 = a2 + math.pi/6.0 + k * math.pi/3.0
        nx, ny = math.cos(theta2), math.sin(theta2)
        proj = abs(dx * nx + dy * ny)
        s1 = _hex_support(angle1, theta2)
        s2 = _hex_support(angle2, theta2)
        sep = proj - s1 - s2
        if sep < min_sep:
            min_sep = sep
    return min_sep


def _enclosing_hex_side_lp(verts, outer_angle_deg):
    """Find minimum side length of regular hexagon containing all vertices via LP."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    dots = verts @ normals.T
    
    A = np.zeros((2 * M * 6, 3))
    b = np.zeros(2 * M * 6)
    
    idx = 0
    for k in range(6):
        nk = normals[k]
        A[idx:idx+M, 0] = nk[0]
        A[idx:idx+M, 1] = nk[1]
        A[idx:idx+M, 2] = -1.0
        b[idx:idx+M] = dots[:, k]
        idx += M
        A[idx:idx+M, 0] = -nk[0]
        A[idx:idx+M, 1] = -nk[1]
        A[idx:idx+M, 2] = -1.0
        b[idx:idx+M] = -dots[:, k]
        idx += M
    
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return side, (cx, cy)
    except Exception:
        pass
    
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return side, (cx, cy)


def _enclosing_hex_side_fast(verts, outer_angle_deg):
    """Fast approximation: centroid-based enclosing hex side."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    return r * 2.0 / _SQRT3


def _smooth_penalty(dist, scale=100.0):
    """Smooth penalty for overlap."""
    if dist >= 0:
        return 0.0
    d = -dist
    return scale * d * d


def _objective_smooth(params, inner_angle, outer_angle_grid):
    """Smooth objective for optimization."""
    n_pos = N * 2
    positions = np.array([(params[i], params[i+1]) for i in range(0, n_pos, 2)])
    outer_angle = params[-1]
    
    all_vertices = _hex_vertices_batch(positions, inner_angle).reshape(-1, 2)
    side = _enclosing_hex_side_fast(all_vertices, outer_angle)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            di = positions[j] - positions[i]
            dist = _hex_distance_same_angle(di, inner_angle)
            if dist < 0:
                penalty += _smooth_penalty(dist, scale=500.0)
    
    return side + penalty


def _objective_exact(params, inner_angle):
    """Exact objective with LP-based enclosing hex."""
    n_pos = N * 2
    positions = np.array([(params[i], params[i+1]) for i in range(0, n_pos, 2)])
    outer_angle = params[-1]
    
    all_vertices = _hex_vertices_batch(positions, inner_angle).reshape(-1, 2)
    side, _ = _enclosing_hex_side_lp(all_vertices, outer_angle)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            di = positions[j] - positions[i]
            dist = _hex_distance_same_angle(di, inner_angle)
            if dist < 0:
                penalty += _smooth_penalty(dist, 1000.0)
    
    return side + penalty


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        positions = []
        positions.append((0, 0))
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for ti, tj in ring1:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        ring2 = [(2,0),(1,1),(-1,2),(-2,0),(-1,-1),(0,-2)]
        for ti, tj in ring2[:4]:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        positions = []
        y1 = -0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        positions = []
        y1 = -0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 10:
        positions = []
        y0 = -3.0*scale
        positions.append((0, y0))
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        y4 = 3.0*scale
        positions.append((0, y4))
        return positions[:N]
    elif layout_idx == 11:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        coords = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]


def _optimize_layout(positions, inner_angle, outer_angles):
    """Optimize a layout with multi-stage refinement."""
    best_side = float('inf')
    best_pos = None
    best_outer = 0.0
    
    pos_arr = np.array(positions)
    
    # Stage 1: Fast screening over outer angles
    for outer_angle in outer_angles:
        all_vertices = _hex_vertices_batch(pos_arr, inner_angle).reshape(-1, 2)
        side = _enclosing_hex_side_fast(all_vertices, outer_angle)
        if side < best_side:
            best_side = side
            best_pos = positions
            best_outer = outer_angle
    
    # Stage 2: Smooth Nelder-Mead refinement
    if time.time() < _DEADLINE - 20.0:
        x0 = np.array([positions[i][0] for i in range(N)] + 
                      [positions[i][1] for i in range(N)] + [best_outer])
        try:
            res = minimize(_objective_smooth, x0, args=(inner_angle, outer_angles),
                          method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_pos = [(res.x[i], res.x[i+1]) for i in range(0, N*2, 2)]
                best_outer = res.x[-1]
        except Exception:
            pass
    
    # Stage 3: L-BFGS-B polish
    if time.time() < _DEADLINE - 15.0 and best_pos is not None:
        x0 = np.array([best_pos[i][0] for i in range(N)] + 
                      [best_pos[i][1] for i in range(N)] + [best_outer])
        try:
            res2 = minimize(_objective_smooth, x0, args=(inner_angle, outer_angles),
                           method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res2.fun < best_side:
                best_side = res2.fun
                best_pos = [(res2.x[i], res2.x[i+1]) for i in range(0, N*2, 2)]
                best_outer = res2.x[-1]
        except Exception:
            pass
    
    # Stage 4: High-precision exact polish with Nelder-Mead
    if time.time() < _DEADLINE - 8.0 and best_pos is not None:
        x0 = np.array([best_pos[i][0] for i in range(N)] + 
                      [best_pos[i][1] for i in range(N)] + [best_outer])
        try:
            res3 = minimize(_objective_exact, x0, args=(inner_angle,),
                           method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res3.fun < best_side:
                best_side = res3.fun
                best_pos = [(res3.x[i], res3.x[i+1]) for i in range(0, N*2, 2)]
                best_outer = res3.x[-1]
        except Exception:
            pass
    
    # Stage 5: Final exact LP polish - recompute with optimized positions
    if time.time() < _DEADLINE - 3.0 and best_pos is not None:
        pos_final = np.array([[p[0], p[1]] for p in best_pos])
        all_vertices = _hex_vertices_batch(pos_final, inner_angle).reshape(-1, 2)
        # Try several nearby outer angles for the final LP
        for da in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            test_angle = best_outer + da
            lp_side, _ = _enclosing_hex_side_lp(all_vertices, test_angle)
            if lp_side < best_side:
                best_side = lp_side
                best_outer = test_angle
    
    return best_side, best_pos, best_outer


def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    # Finer outer angle grid
    outer_angles = list(np.arange(0, 30.0, 0.25))
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner_angle = 30.0
        
        refined_side, refined_pos, refined_angle = _optimize_layout(
            positions, inner_angle, outer_angles)
        
        if refined_side < best_side:
            best_side = refined_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in refined_pos]
            best_center = (0.0, 0.0)
            best_outer_angle = refined_angle
    
    # Final: compute exact enclosing hex side via LP
    if best_inner is not None:
        pos_arr = np.array([[p[0], p[1]] for p in best_inner])
        all_vertices = _hex_vertices_batch(pos_arr, best_inner[0][2]).reshape(-1, 2)
        exact_side, center = _enclosing_hex_side_lp(all_vertices, best_outer_angle)
        if exact_side < best_side:
            best_side = exact_side
        best_center = center
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle

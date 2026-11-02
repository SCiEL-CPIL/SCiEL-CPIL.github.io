import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
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
    The hexagon has 6 sides. We want to minimize the side length such that
    all vertices are inside.
    
    A regular hexagon with center (cx, cy) and side length s has inradius r = s * sqrt(3)/2.
    The hexagon contains all points if for each of the 6 outward normals n_k,
    the projection of (point - center) onto n_k is <= r.
    
    We set up LP: minimize r
    subject to: n_k . (p_i - c) <= r for all i, k
    and -n_k . (p_i - c) <= r for all i, k  (equivalently n_{k+3} . (p_i - c) <= r)
    
    Variables: cx, cy, r
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    n_points = len(verts)
    
    # Variables: [cx, cy, r]
    # Objective: minimize r -> c = [0, 0, 1]
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Constraints: for each point i and each normal k:
    # n_k[0]*(p_i[0] - cx) + n_k[1]*(p_i[1] - cy) <= r
    # => -n_k[0]*cx - n_k[1]*cy - r <= -n_k[0]*p_i[0] - n_k[1]*p_i[1]
    # And for the opposite direction (normal k+3 = -normal k):
    # -n_k[0]*(p_i[0] - cx) - n_k[1]*(p_i[1] - cy) <= r
    # => n_k[0]*cx + n_k[1]*cy - r <= n_k[0]*p_i[0] + n_k[1]*p_i[1]
    
    A_ub = []
    b_ub = []
    
    for i in range(n_points):
        px, py = verts[i]
        for k in range(6):
            nx, ny = normals[k]
            # Constraint 1: n_k . (p - c) <= r
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * px - ny * py)
            # Constraint 2: -n_k . (p - c) <= r  (i.e., n_{k+3} . (p-c) <= r)
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * px + ny * py)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # r must be >= 0
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback to geometric method
    return _min_enclosing_hex_side_geometric(verts, outer_angle_deg)


def _min_enclosing_hex_side_geometric(verts, outer_angle_deg):
    """
    Find a valid enclosing regular hexagon side length using geometric projection.
    Uses centroid as initial center and refines with a small local search.
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Initial center: centroid
    cx, cy = np.mean(verts, axis=0)
    
    def get_side(cx, cy):
        rel = verts - np.array([cx, cy])
        projections = rel @ normals.T
        max_proj = np.max(projections, axis=0)
        min_proj = np.min(projections, axis=0)
        r = np.max(np.maximum(max_proj, -min_proj))
        if r < 0:
            r = 0
        side = r * 2.0 / _SQRT3
        return side, r

    best_side, best_r = get_side(cx, cy)
    best_cx, best_cy = cx, cy
    
    # Small local search around centroid to minimize r
    step = 0.1
    for dx in np.arange(-1.0, 1.1, step):
        for dy in np.arange(-1.0, 1.1, step):
            ncx = cx + dx
            ncy = cy + dy
            s, r = get_side(ncx, ncy)
            if s < best_side:
                best_side = s
                best_r = r
                best_cx = ncx
                best_cy = ncy
                
    return best_cx, best_cy, best_side


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
        # 2-4-2-3 pattern (more elongated)
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
        # Center + 6 ring + 4 more, but different selection
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
        # Alternative compact: center + 6 ring + 4 from second ring (different 4)
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        # Pick 4 from ring2 that are spaced apart
        ring2_pick = [(2, 0), (0, 2), (-2, 0), (0, -2)]
        target.extend(ring2_pick)
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 9:
        # 5-6 pattern (two rows of 5 and 6, offset)
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
    else:
        # Generic: use hex lattice coords sorted by distance
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y, i, j))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return [(x, y) for x, y, i, j in coords_sorted[:N]]


def _polish_positions(positions, outer_angle_deg, max_iter=50, tol=1e-6):
    """
    Polish positions by local optimization. For each position, try small perturbations
    to reduce the enclosing hexagon side length.
    """
    positions = [list(p) for p in positions]
    inner_angle = 30.0
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 15.0:
            break
        
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
        
        improved = False
        for idx in range(len(positions)):
            if time.time() > _DEADLINE - 15.0:
                break
            
            px, py = positions[idx]
            best_local_side = side
            best_local_pos = (px, py)
            
            # Try small perturbations
            for dx in np.arange(-0.05, 0.06, 0.02):
                for dy in np.arange(-0.05, 0.06, 0.02):
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = (px + dx, py + dy)
                    positions[idx] = new_pos
                    all_v = np.vstack([_hex_vertices(pp[0], pp[1], inner_angle) for pp in positions])
                    _, _, test_side = _min_enclosing_hex_side_lp(all_v, outer_angle_deg)
                    if test_side < best_local_side:
                        best_local_side = test_side
                        best_local_pos = new_pos
            positions[idx] = best_local_pos
            if best_local_pos != (px, py):
                improved = True
        
        if not improved:
            break
    
    return positions


def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    num_layouts = 10
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner_angle = 30.0
        
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        
        best_local_side = float('inf')
        best_local_center = (0.0, 0.0)
        best_local_angle = 0.0
        
        # Try several outer angles
        for outer_angle_deg in np.arange(0, 30.0, 1.0):
            if time.time() > _DEADLINE - 10.0:
                break
            
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = outer_angle_deg
        
        # Polish the positions
        if time.time() > _DEADLINE - 20.0:
            polished = positions
        else:
            polished = _polish_positions(positions, best_local_angle, max_iter=30)
            # Re-evaluate with polished positions
            all_vertices_polished = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in polished])
            for outer_angle_deg in np.arange(best_local_angle - 3, best_local_angle + 4, 1.0):
                if outer_angle_deg < 0 or outer_angle_deg >= 30:
                    continue
                res = _min_enclosing_hex_side_lp(all_vertices_polished, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_local_side:
                        best_local_side = side
                        best_local_center = (cx, cy)
                        best_local_angle = outer_angle_deg
            positions = polished
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_local_center
            best_outer_angle = best_local_angle
    
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle

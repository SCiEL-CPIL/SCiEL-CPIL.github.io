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

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    v_dot_n = verts @ normals.T
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
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

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    scale = 1.0000001
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 1:
        # Row-based: 3 rows
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
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 but shifted
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
        return positions[:N]
    elif layout_idx == 3:
        # 5-3-3 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 3):
            x = k * _SQRT3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 4:
        # 3-5-3 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y3))
        return positions[:N]
    else:
        # Fallback: sorted by distance
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]

def _refine_with_nlp(inner_positions, outer_angle_deg, center_guess, side_guess, time_budget):
    """
    Refine positions and outer hexagon parameters using NLP.
    Returns (inner_positions, center, side, outer_angle_deg) or None.
    """
    from scipy.optimize import minimize
    
    t0 = time.time()
    
    # Flatten all positions and outer parameters
    # x = [x1,y1,x2,y2,...,x11,y11, cx, cy, R, angle_deg]
    # R = side * sqrt(3)/2
    
    n_hex = len(inner_positions)
    n_vars = 2 * n_hex + 4  # 22 + 4 = 26
    
    def _objective(x):
        if time.time() - t0 > time_budget:
            return float('inf')
        
        # Extract positions
        pos = x[:2*n_hex].reshape(n_hex, 2)
        cx, cy, R, angle_deg = x[2*n_hex], x[2*n_hex+1], x[2*n_hex+2], x[2*n_hex+3]
        
        if R < 0:
            return float('inf')
        
        # Compute all vertices of inner hexagons (all at angle 0 for simplicity in NLP)
        # Actually, we should allow rotation too, but let's start with fixed inner angles
        # For now, keep inner hexagons at 0 degrees
        all_verts = []
        for i in range(n_hex):
            all_verts.append(_hex_vertices(pos[i, 0], pos[i, 1], 0.0))
        all_verts = np.vstack(all_verts)
        
        # Outer hexagon normals
        base_angle = math.radians(angle_deg + 30)
        angles = base_angle + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Check containment: for each vertex v and normal n:
        # (v - c).n <= R
        # This is a soft constraint: sum of max(0, (v-c).n - R)^2
        v_dot_n = all_verts @ normals.T  # (M, 6)
        c_dot_n = cx * normals[:, 0] + cy * normals[:, 1]  # (6,)
        
        # (v - c).n = v.n - c.n
        # For each (i, k): v_dot_n[i,k] - c_dot_n[k] <= R
        # Soft violation: max(0, v_dot_n[i,k] - c_dot_n[k] - R)
        
        # Vectorized: v_dot_n is (M,6), c_dot_n is (6,), broadcast
        violations = v_dot_n - c_dot_n[None, :] - R
        violations = np.maximum(0, violations)
        constraint_penalty = np.sum(violations ** 2)
        
        # Objective: R + penalty
        return R + 10.0 * constraint_penalty
    
    # Initial guess
    x0 = np.zeros(n_vars)
    for i, (px, py) in enumerate(inner_positions):
        x0[2*i] = px
        x0[2*i+1] = py
    x0[2*n_hex] = center_guess[0]
    x0[2*n_hex+1] = center_guess[1]
    x0[2*n_hex+2] = side_guess * _SQRT3 / 2
    x0[2*n_hex+3] = outer_angle_deg
    
    try:
        res = minimize(_objective, x0, method='L-BFGS-B',
                       options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if time.time() - t0 > time_budget:
            return None
        
        if res.success or res.fun < side_guess * _SQRT3 / 2:
            pos = res.x[:2*n_hex].reshape(n_hex, 2)
            cx, cy, R, angle_deg = res.x[2*n_hex], res.x[2*n_hex+1], res.x[2*n_hex+2], res.x[2*n_hex+3]
            
            if R > 0:
                side = R / (_SQRT3 / 2)
                new_positions = [(pos[i, 0], pos[i, 1]) for i in range(n_hex)]
                return new_positions, (cx, cy), side, angle_deg
    except Exception:
        pass
    
    return None

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    layouts = [_build_positions(i) for i in range(5)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 15.0:
                break
            
            all_vertices = []
            for px, py in positions:
                all_vertices.extend(_hex_vertices(px, py, inner_angle))
            all_vertices = np.array(all_vertices)
            
            # Coarse search
            best_coarse_side = float('inf')
            best_coarse_angle = 0.0
            best_coarse_center = (0.0, 0.0)
            
            for outer_angle_deg in np.arange(0, 30.1, 0.5):
                if time.time() > _DEADLINE - 15.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
            
            # Fine search
            if best_coarse_angle > 0:
                for outer_angle_deg in np.arange(max(0, best_coarse_angle - 0.5),
                                                 min(30.1, best_coarse_angle + 0.5), 0.01):
                    if time.time() > _DEADLINE - 15.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_coarse_side:
                            best_coarse_side = side
                            best_coarse_angle = outer_angle_deg
                            best_coarse_center = (cx, cy)
            
            # NLP refinement
            if best_coarse_side < best_side and time.time() < _DEADLINE - 30.0:
                time_budget = min(20.0, _DEADLINE - 30.0 - time.time())
                refined = _refine_with_nlp(positions, best_coarse_angle, 
                                            best_coarse_center, best_coarse_side, time_budget)
                if refined:
                    new_positions, new_center, new_side, new_angle = refined
                    # Verify the refined solution is valid
                    all_verts_new = []
                    for px, py in new_positions:
                        all_verts_new.extend(_hex_vertices(px, py, inner_angle))
                    all_verts_new = np.array(all_verts_new)
                    verify_res = _min_enclosing_hex_side(all_verts_new, new_angle)
                    if verify_res and verify_res[2] <= new_side + 0.001:
                        best_coarse_side = new_side
                        best_coarse_center = new_center
                        best_coarse_angle = new_angle
                        positions = new_positions
            
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle
                best_inner_angle = inner_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle

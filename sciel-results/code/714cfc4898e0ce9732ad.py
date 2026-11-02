import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    vdots = verts @ normals.T  # shape (M, 6)
    
    A_rows = []
    b_vals = []
    
    for j in range(6):
        nx, ny = normals[j]
        A_rows.append(np.tile([-nx, -ny, -1.0], (M, 1)))
        b_vals.append(-vdots[:, j])
        A_rows.append(np.tile([nx, ny, -1.0], (M, 1)))
        b_vals.append(vdots[:, j])
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    A = np.vstack([A, np.array([[0.0, 0.0, -1.0]])])
    b = np.concatenate([b, [0.0]])
    
    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs', options={'presolve': True,
                      'dual_feasibility_tolerance': 1e-12,
                      'primal_feasibility_tolerance': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None

def _compute_all_outer(params, positions, outer_angle_deg):
    """
    params: array of length 12*(N-1)+1 = [inner_angle, dx2,dy2,dr2, dx3,dy3,dr3, ...]
    positions[0] is fixed at (0,0). Returns (side, verts_all) or (inf, None).
    """
    inner_angle = params[0]
    pos = np.array([positions[0]])
    for k in range(1, N):
        base = 1 + 3*(k-1)
        dx = params[base]
        dy = params[base+1]
        dr = params[base+2]
        pos = np.vstack([pos, [positions[k][0]+dx, positions[k][1]+dy]])
    # All inner hexagons with same angle
    all_v = np.vstack([_hex_vertices(pos[i][0], pos[i][1], inner_angle) for i in range(N)])
    res = _min_enclosing_hex_side_lp(all_v, outer_angle_deg)
    if res is None:
        return float('inf'), None
    return res[2], all_v

def _objective(params, positions, outer_angle_deg):
    side, _ = _compute_all_outer(params, positions, outer_angle_deg)
    return side

def _build_initial_positions(mode):
    """Generate initial positions for 11 hexagons in a hexagonal lattice pattern."""
    sqrt3 = _SQRT3
    s = 1.000001
    u = np.array([sqrt3*s, 0.0])
    v = np.array([sqrt3/2*s, 1.5*s])
    
    if mode == 0:
        # Compact: center + ring1 + 4 from ring2
        target = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
                  (2,0),(0,2),(-2,0),(0,-2)]
    elif mode == 1:
        # Linear-ish arrangement
        target = [(i, 0) for i in range(-5, 6)]
    elif mode == 2:
        # Two rows
        target = [(i, 0) for i in range(-3, 4)] + [(i, 1) for i in range(-4, 3)]
    elif mode == 3:
        # Three rows staggered
        target = [(i, 0) for i in range(-4, 4)] + [(i, 1) for i in range(-3, 4)]
    elif mode == 4:
        # Cross pattern
        target = [(0,0), (1,0),(-1,0),(0,1),(0,-1),
                  (2,0),(-2,0),(0,2),(0,-2),(1,1),(-1,-1)]
    else:
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        target.extend(ring1)
        target.extend(sorted(ring2, key=lambda p: p[0]**2+p[1]**2)[:4])
    
    positions = []
    for ti, tj in target[:N]:
        p = ti*u + tj*v
        positions.append((p[0], p[1]))
    return positions

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
    
    num_modes = 5
    positions_list = [_build_initial_positions(i) for i in range(num_modes)]
    outer_angles = [0.0, 15.0, 30.0]
    inner_angles_init = [0.0, 30.0]
    
    # First pass: coarse search to find good starting points
    coarse_best = {}
    for mode_idx, positions in enumerate(positions_list):
        if time.time() > _DEADLINE - 60.0:
            break
        for outer_angle in outer_angles:
            for inner_angle in inner_angles_init:
                if time.time() > _DEADLINE - 60.0:
                    break
                pos = np.array(positions)
                all_v = np.vstack([_hex_vertices(pos[i][0], pos[i][1], inner_angle) for i in range(N)])
                res = _min_enclosing_hex_side_lp(all_v, outer_angle)
                if res:
                    key = (mode_idx, outer_angle, inner_angle)
                    coarse_best[key] = (res[2], res[0], res[1])
    
    # Second pass: local optimization from best starting points
    # Sort by side length
    sorted_keys = sorted(coarse_best.keys(), key=lambda k: coarse_best[k][0])
    
    for mode_idx, outer_angle, inner_angle in sorted_keys[:8]:
        if time.time() > _DEADLINE - 20.0:
            break
        
        positions = positions_list[mode_idx]
        side0, cx0, cy0 = coarse_best[(mode_idx, outer_angle, inner_angle)]
        
        # Initial parameters: [inner_angle, dx2,dy2,dr2, ...]
        x0 = np.zeros(1 + 3*(N-1))
        x0[0] = inner_angle
        # displacements are 0 initially
        
        bounds = [(-180, 180)] + [(-2.0, 2.0)] * (3*(N-1))
        
        try:
            result = minimize(
                lambda p: _objective(p, positions, outer_angle),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success and result.fun < best_side:
                params = result.x
                final_side, all_v = _compute_all_outer(params, positions, outer_angle)
                if final_side < best_side:
                    # Extract positions
                    inner_angle_opt = params[0]
                    pos_list = [positions[0]]
                    for k in range(1, N):
                        base = 1 + 3*(k-1)
                        pos_list.append((positions[k][0]+params[base],
                                        positions[k][1]+params[base+1]))
                    
                    # Compute outer hexagon center and side
                    res_final = _min_enclosing_hex_side_lp(all_v, outer_angle)
                    if res_final:
                        cx, cy, side = res_final
                        best_side = side
                        best_inner = [[p[0], p[1], inner_angle_opt] for p in pos_list]
                        best_center = (cx, cy)
                        best_outer_angle = outer_angle
                        best_inner_angle = inner_angle_opt
        except Exception:
            pass
    
    # If no improvement, use the best coarse result
    if best_inner is None and coarse_best:
        mode_idx, outer_angle, inner_angle = sorted_keys[0]
        side0, cx0, cy0 = coarse_best[(mode_idx, outer_angle, inner_angle)]
        positions = positions_list[mode_idx]
        best_side = side0
        best_inner = [[p[0], p[1], inner_angle] for p in positions]
        best_center = (cx0, cy0)
        best_outer_angle = outer_angle
        best_inner_angle = inner_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle

import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar, differential_evolution

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3
_SCALE = 1.0000001

def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    M = len(verts)
    vdots = verts @ normals.T
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
                      method='highs', options={'presolve': True})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None

def _build_base_positions():
    """Return (base_positions, rings) for the 11-hex lattice."""
    s = _SCALE
    u = (math.sqrt(3)*s, 0.0)
    v = (math.sqrt(3)/2*s, 1.5*s)
    # ring 0: center
    # ring 1: 6 neighbors
    # ring 2: 12 neighbors (we take 4)
    ring0 = [(0,0)]
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
    # Take 4 from ring 2: the ones at angles 0, 60, 180, 240 (i.e., (2,0),(1,1),(-2,0),(-1,-1))
    ring2_sel = [(2,0),(1,1),(-2,0),(-1,-1)]
    all_rings = [ring0, ring1, ring2_sel]
    positions = []
    for ring in all_rings:
        for (i,j) in ring:
            x = i*u[0] + j*v[0]
            y = i*u[1] + j*v[1]
            positions.append((x,y))
    return positions, all_rings

def _deform_positions(radii, rots):
    """Apply per-ring radial breathing and rotation to base positions."""
    base_pos, rings = _build_base_positions()
    out = []
    for ring_idx, ring in enumerate(rings):
        r = radii[ring_idx]
        theta = math.radians(rots[ring_idx])
        ct, st = math.cos(theta), math.sin(theta)
        for (x, y) in ring:
            # scale by (1+r) then rotate
            xs = x * (1.0 + r)
            ys = y * (1.0 + r)
            xr = ct*xs - st*ys
            yr = st*xs + ct*ys
            out.append((xr, yr))
    return out

def _overlap_penalty(positions, margin=0.0005):
    """Smooth penalty: for each pair, if distance < 2*sqrt(3)/2 - margin, add (deficit)^2."""
    # For two unit hexagons in same orientation, minimum non-overlap center distance = sqrt(3)
    # (edge-to-edge touching). With rotation, the min distance can be smaller; use conservative 1.73.
    min_dist = math.sqrt(3) - margin  # ~1.730
    n = len(positions)
    pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                deficit = min_dist - d
                pen += deficit * deficit
    return pen

def _objective(params):
    """params = [r0, r1, r2, rot0, rot1, rot2, outer_angle_deg].
    Returns side length + overlap penalty (smooth surrogate)."""
    r0, r1, r2, rot0, rot1, rot2, oa = params
    positions = _deform_positions([r0, r1, r2], [rot0, rot1, rot2])
    # overlap penalty (smooth, quadratic)
    pen = _overlap_penalty(positions)
    if pen > 1e-3:
        # hard penalty to discourage overlap
        return 100.0 + pen
    # compute all vertices
    all_v = np.vstack([_hex_vertices(px, py, 0.0) for px, py in positions])
    res = _min_enclosing_hex_side(all_v, oa)
    if res is None:
        return 1000.0 + pen
    return res[2] + 10.0 * pen

def construct():
    global _START_TIME, _DEADLINE
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    bounds = [(-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02),
              (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
              (0.0, 30.0)]
    
    # Multi-start differential evolution
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_params = None
    
    n_starts = 8
    for start_idx in range(n_starts):
        if time.time() > _DEADLINE - 15.0:
            break
        # Different initial populations via seed
        seed = 42 + start_idx * 1000
        try:
            res = differential_evolution(
                _objective, bounds,
                seed=seed,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                update_interval=1,
                callback=None,
            )
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Polish: refine outer angle with minimize_scalar given fixed positions
    r0, r1, r2, rot0, rot1, rot2, oa = best_params
    positions = _deform_positions([r0, r1, r2], [rot0, rot1, rot2])
    all_v = np.vstack([_hex_vertices(px, py, 0.0) for px, py in positions])
    
    def _obj_angle(a):
        res = _min_enclosing_hex_side(all_v, a)
        return res[2] if res else 1e6
    
    try:
        lo = max(0.0, oa - 3.0)
        hi = min(30.0, oa + 3.0)
        res_opt = minimize_scalar(_obj_angle, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-10, 'maxiter': 500})
        if res_opt.success and res_opt.fun < best_side:
            best_side = res_opt.fun
            best_params[6] = res_opt.x
            oa = res_opt.x
    except Exception:
        pass
    
    # Fine grid polish
    if time.time() < _DEADLINE - 10.0:
        for step in [0.01, 0.002, 0.0005]:
            lo = max(0.0, oa - step*2)
            hi = min(30.0, oa + step*2)
            for a in np.arange(lo, hi + step*0.1, step):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side(all_v, a)
                if res and res[2] < best_side:
                    best_side = res[2]
                    oa = a
                    best_params[6] = a
    
    # Get final center
    res_final = _min_enclosing_hex_side(all_v, oa)
    if res_final:
        cx, cy, side = res_final
        if side < best_side:
            best_side = side
        best_center = (cx, cy)
        best_outer_angle = oa
    
    best_inner = [[pos[0], pos[1], 0.0] for pos in positions]
    
    return best_inner, best_center, best_side, best_outer_angle

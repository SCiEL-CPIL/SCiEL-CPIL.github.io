import numpy as np
import math
import time
from scipy.optimize import minimize, least_squares

N = 12
DEADLINE = 138.0

_SQRT3 = np.sqrt(3.0)
_HEX_ANGLES = np.radians(np.array([0, 60, 120, 180, 240, 300]))
_OUTER_NORMAL_ANGLES = np.radians(np.array([30, 90, 150, 210, 270, 330]))


def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = _HEX_ANGLES + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))


def _hex_edge_normals(rot_deg):
    base = _OUTER_NORMAL_ANGLES + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))


def _sat_sep(v1, v2, n1, n2):
    min_overlap = float('inf')
    for nm in (n1, n2):
        for k in range(6):
            n = nm[k]
            p1 = v1 @ n
            p2 = v2 @ n
            mn1, mx1 = np.min(p1), np.max(p1)
            mn2, mx2 = np.min(p2), np.max(p2)
            if mx1 < mn2:
                return mn2 - mx1
            if mx2 < mn1:
                return mn1 - mx2
            ov = min(mx1, mx2) - max(mn1, mn2)
            if ov < min_overlap:
                min_overlap = ov
    return -min_overlap


def _outer_hex_side_fast(points, outer_rot_deg):
    angles = _OUTER_NORMAL_ANGLES + np.radians(outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / _SQRT3) * max_proj


def _check_disjoint(inner, tol=1e-7):
    n = len(inner)
    verts, norms = [], []
    for i in range(n):
        x, y, rot = inner[i]
        verts.append(_hex_vertices(x, y, 1.0, rot))
        norms.append(_hex_edge_normals(rot))
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_sep(verts[i], verts[j], norms[i], norms[j]) < -tol:
                return False
    return True


def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    spacing_x = _SQRT3 + gap
    spacing_y = 1.5 + 0.5 * gap
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts, dtype=float)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts


def _make_ring(gap=0.0):
    pts = []
    d1 = _SQRT3 + gap
    d2 = 2 * _SQRT3 + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
    pts = np.array(pts, dtype=float)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts


def _compute_penalty(overlaps, beta):
    if len(overlaps) == 0:
        return 0.0
    p_arr = np.array(overlaps)
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 500.0:
        return float(np.sum(p_arr ** 2))
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return float(beta * lse)


def _objective(p, t0, deadline, beta=50.0):
    if time.time() - t0 > deadline:
        return 1e6

    inner = p[:36].reshape(12, 3)
    ox, oy, oang = p[36], p[37], p[38]

    verts, norms = [], []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        verts.append(v)
        norms.append(_hex_edge_normals(rot))

    overlaps = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_sep(verts[i], verts[j], norms[i], norms[j])
            if d < 0:
                overlaps.append(-d)

    pen = _compute_penalty(overlaps, beta)

    all_pts = np.vstack(verts)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    L = _outer_hex_side_fast(all_pts, oang)
    return L + 100.0 * pen


def _contact_residuals(p, t0, deadline):
    """
    Residuals for least-squares formulation.
    1. Contact constraints: pairs of hexagons should be touching (distance ~ 0).
       We define 'touching' as separation < 0.1 (close enough to consider a contact candidate).
       Actually, for least squares, we want to minimize the square of the separation for pairs
       that are 'close'. But we don't know the graph.
       
       Alternative approach inspired by 'topological graph optimization':
       Assume a contact graph. If we assume the 'ring' topology or 'grid' topology,
       we can enforce that specific pairs are in contact.
       
       However, a more robust 'structural' change is to use a soft constraint for ALL pairs
       that are closer than a threshold, pushing them to exactly 0 separation, while
       pushing all other pairs to a large separation.
       
       Let's try:
       For each pair (i,j):
       sep = _sat_sep(...)
       if sep < 0.5: # close
           residual = sep # want to be 0 (touching)
       else:
           residual = 0 # or a large positive number?
       
       This is non-smooth.
       
       Better: Use the penalty method but with a different optimizer structure?
       The prompt suggests: "minimizes the sum of squared deviations from ideal edge-to-edge contact distances".
       
       Let's define a target separation for each pair.
       If we assume a specific contact graph (e.g., from the initial configuration),
       we can fix which pairs should touch.
       
       Let's implement a function that takes a contact graph (list of pairs) and
       minimizes (sep_ij - 0)^2 for (i,j) in graph, and (sep_ij - D_max)^2 for others?
       No, that's hard.
       
       Let's stick to the idea: 
       1. Identify 'close' pairs (sep < threshold).
       2. For these pairs, residual = sep (aim for 0).
       3. For 'far' pairs, residual = 0 (don't care, or aim for large).
       4. Also minimize outer side length.
       
       We can combine this with the outer side length minimization by adding
       residuals for the outer boundary constraints.
       
       Let's define residuals:
       - For each pair (i,j) with sep < 0.2: residual = sep
       - For outer boundary: We want to minimize L.
         L is determined by the max projection.
         We can add residuals for each vertex projection against the boundary.
         Let proj_k be the projection of vertex k on the 6 normal directions.
         We want |proj_k| <= R (where R is related to L).
         To minimize L, we can minimize the max |proj|.
         This is a minimax problem.
         
       Let's use scipy.optimize.least_squares.
       
       Residuals:
       1. Contact residuals: For all pairs, if sep < 0.5, add sep. 
          (This pushes close pairs to touch).
          To prevent overlap, we need a hard constraint or a very strong penalty.
          Least squares minimizes sum of squares. If sep is negative (overlap), 
          (sep)^2 is positive. So it will try to reduce overlap.
          But it might allow small overlaps if it reduces L significantly.
          We need to ensure disjointness.
          
       Let's try a hybrid:
       - Use least_squares to find a configuration where close pairs are in contact.
       - Then use the previous penalized objective to polish and ensure disjointness.
       
       Residuals for least_squares:
       - For each pair (i,j):
         sep = _sat_sep(...)
         if sep < 0.3:
             r = sep  # push to 0
         else:
             r = 0
       - Outer side length:
         We can add a residual for the outer side length itself?
         L is a scalar. We can't add a scalar directly as a residual vector element
         unless we treat it as a single residual.
         r_L = L
         
       So residual vector = [sep_1, sep_2, ..., sep_k, L]
       Where sep_i are the separations of close pairs.
       
       This minimizes sum(sep_i^2) + L^2.
       This encourages contacts and small L.
    """
    if time.time() - t0 > deadline:
        return np.array([1e6])

    inner = p[:36].reshape(12, 3)
    ox, oy, oang = p[36], p[37], p[38]

    verts, norms = [], []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        verts.append(v)
        norms.append(_hex_edge_normals(rot))

    residuals = []
    # Contact constraints
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_sep(verts[i], verts[j], norms[i], norms[j])
            if d < 0.5:  # Close enough to consider
                residuals.append(d)
            
    # Outer side length
    all_pts = np.vstack(verts)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    L = _outer_hex_side_fast(all_pts, oang)
    residuals.append(L)
    
    return np.array(residuals)


def _polish(params, t0, deadline):
    if time.time() - t0 > deadline - 15:
        return params

    # Stage 1: Least Squares for contact graph optimization
    if time.time() - t0 < deadline - 20:
        try:
            res = least_squares(_contact_residuals, params, args=(t0, deadline - 20),
                                method='lm',  # Levenberg-Marquardt
                                max_nfev=2000)
            # Check if result is better in terms of L (ignoring potential overlaps for now)
            # We will verify disjointness later.
            if res.x is not None:
                # Compare objective function values (penalized) to be safe
                obj_old = _objective(params, t0, deadline, 50.0)
                obj_new = _objective(res.x, t0, deadline, 50.0)
                if obj_new < obj_old:
                    params = res.x
        except Exception:
            pass

    # Stage 2: L-BFGS-B with moderate penalty
    if time.time() - t0 < deadline - 10:
        try:
            res = minimize(_objective, params, args=(t0, deadline - 10, 50.0),
                           method='L-BFGS-B',
                           options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 6000})
            if res.fun < _objective(params, t0, deadline, 50.0):
                params = res.x
        except Exception:
            pass

    # Stage 3: L-BFGS-B with higher penalty
    if time.time() - t0 < deadline - 5:
        try:
            res = minimize(_objective, params, args=(t0, deadline - 5, 200.0),
                           method='L-BFGS-B',
                           options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 6000})
            if res.fun < _objective(params, t0, deadline, 200.0):
                params = res.x
        except Exception:
            pass

    # Stage 4: Nelder-Mead final refinement
    if time.time() - t0 < deadline - 2:
        try:
            res = minimize(_objective, params, args=(t0, deadline - 2, 500.0),
                           method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < _objective(params, t0, deadline, 500.0):
                params = res.x
        except Exception:
            pass

    return params


def construct():
    t0 = time.time()

    def time_left():
        return DEADLINE - (time.time() - t0)

    configs = []
    gap = 0.05
    configs.append(_make_hex_grid(4, 3, gap=gap))
    configs.append(_make_hex_grid(3, 4, gap=gap))
    configs.append(_make_hex_grid(2, 6, gap=gap))
    configs.append(_make_hex_grid(6, 2, gap=gap))
    configs.append(_make_hex_grid(4, 3, rot_offset=30, gap=gap))
    configs.append(_make_hex_grid(3, 4, rot_offset=30, gap=gap))
    configs.append(_make_ring(gap=gap))

    # Perturbed ring
    np.random.seed(42)
    ring_p = _make_ring(gap=gap).copy()
    ring_p[:, 0] += np.random.randn(12) * 0.15
    ring_p[:, 1] += np.random.randn(12) * 0.15
    configs.append(ring_p)

    # More random starts with varied rotations
    for seed in [1, 7, 13]:
        np.random.seed(seed)
        pts = np.random.randn(12, 3) * 2.5
        pts[:, 2] = np.random.choice([0, 30, 15, 45], size=12)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)

    best_L = float('inf')
    best_params = None
    outer_rots = [0.0, 30.0, 15.0, 45.0]

    for cfg in configs:
        if time_left() < 15:
            break
        for orot in outer_rots:
            if time_left() < 8:
                break
            init = np.concatenate((cfg.flatten(), [0.0, 0.0, orot]))

            polished = _polish(init, t0, DEADLINE)

            inner = polished[:36].reshape(12, 3)
            ox, oy, oang = polished[36], polished[37], polished[38]
            verts = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(12)]
            norms = [_hex_edge_normals(inner[i, 2]) for i in range(12)]

            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    if _sat_sep(verts[i], verts[j], norms[i], norms[j]) < -1e-7:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                all_pts = np.vstack(verts)
                all_pts[:, 0] -= ox
                all_pts[:, 1] -= oy
                L = _outer_hex_side_fast(all_pts, oang)
                if L < best_L:
                    best_L = L
                    best_params = polished.copy()

    if best_params is None:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))

    inner = []
    for i in range(12):
        inner.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])

    if not _check_disjoint(inner):
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle

    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, outer_angle)
    outer_side *= 1.00001

    return inner, outer_center, outer_side, outer_angle

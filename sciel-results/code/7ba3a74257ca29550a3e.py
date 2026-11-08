"""Rewritten program: unit hexagons in hexagon n=12, minimize outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _polygon_overlap_depth(A, B):
    """SAT overlap depth between convex polygons A and B (arrays of 2D verts). 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            length = np.linalg.norm(edge)
            if length > 1e-12:
                axes.append(np.array([-edge[1], edge[0]]) / length)
    for ax in axes:
        pa = np.dot(A, ax)
        pb = np.dot(B, ax)
        if pa.max() <= pb.min() or pb.max() <= pa.min():
            return 0.0
        ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if ov < min_depth:
            min_depth = ov
    return min_depth

def _containment_violation(inner_verts, outer_verts):
    """Sum of distances by which inner verts fall outside outer (edge-wise)."""
    n_out = len(outer_verts)
    pen = 0.0
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            el = np.linalg.norm(edge)
            if el < 1e-12:
                continue
            nrm = np.array([-edge[1], edge[0]]) / el
            sd = np.dot(iv - v1, nrm)
            if sd < 0:
                pen += -sd
    return pen

def _smooth_max(vals, beta=50.0):
    """Smooth maximum via log-sum-exp."""
    vals = np.array(vals)
    if len(vals) == 0:
        return 0.0
    m = np.max(vals)
    if m < -1e10:
        return 0.0
    return m + np.log(np.sum(np.exp((vals - m) * beta)) / len(vals)) / beta

def _objective_smooth(params, S, center, outer_angle, beta=50.0):
    """Smooth penalty objective using log-sum-exp for non-smooth terms."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    # Containment penalties
    cont_pens = []
    for iv in inner_list:
        cont_pens.append(_containment_violation(iv, outer))
    
    # Overlap penalties
    overlap_pens = []
    for i in range(N):
        for j in range(i+1, N):
            d = _polygon_overlap_depth(inner_list[i], inner_list[j])
            if d > 0:
                overlap_pens.append(d)
    
    # Use smooth max for containment, sum for overlaps (with smooth handling)
    total_cont = _smooth_max(cont_pens, beta) if cont_pens else 0.0
    total_overlap = _smooth_max(overlap_pens, beta) if overlap_pens else 0.0
    
    return total_cont + total_overlap

def _make_grid(nrows, ncols):
    pos = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * np.sqrt(3.0)
            pos.append([x, y])
    arr = np.array(pos)
    arr -= arr.mean(axis=0)
    return arr

def _ring_init(radius=2.0):
    pts = []
    for i in range(6):
        a = np.radians(60*i)
        pts.append([radius*np.cos(a), radius*np.sin(a)])
    for i in range(6):
        a = np.radians(60*i + 30)
        pts.append([(radius*1.5)*np.cos(a), (radius*1.5)*np.sin(a)])
    return np.array(pts) - np.array(pts).mean(axis=0)

def _linear_init(S):
    xs = np.linspace(-S*0.7, S*0.7, N)
    return np.column_stack([xs, np.zeros(N)])

def _optimize_group(params_group, S, center, outer_angle, group_indices):
    """Optimize a subset of hexagons while others are fixed."""
    fixed_params = params_group.copy()
    
    def obj(p):
        p_full = fixed_params.copy()
        for k, idx in enumerate(group_indices):
            p_full[3*idx] = p[3*k]
            p_full[3*idx+1] = p[3*k+1]
            p_full[3*idx+2] = p[3*k+2]
        return _objective_smooth(p_full, S, center, outer_angle, beta=30.0)
    
    init_p = np.zeros(3 * len(group_indices))
    for k, idx in enumerate(group_indices):
        init_p[3*k] = fixed_params[3*idx]
        init_p[3*k+1] = fixed_params[3*idx+1]
        init_p[3*k+2] = fixed_params[3*idx+2]
    
    try:
        res = minimize(obj, init_p, method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 5000,
                                'ftol': 1e-12, 'gtol': 1e-8})
        for k, idx in enumerate(group_indices):
            fixed_params[3*idx] = res.x[3*k]
            fixed_params[3*idx+1] = res.x[3*k+1]
            fixed_params[3*idx+2] = res.x[3*k+2]
        return fixed_params, res.fun
    except Exception:
        return fixed_params, float('inf')

def _hierarchical_optimize(init_params, S, center, outer_angle, max_iter=3):
    """Hierarchical: optimize groups of 6, then all together."""
    params = init_params.copy()
    group1 = list(range(6))
    group2 = list(range(6, 12))
    
    for iteration in range(max_iter):
        # Optimize group 1
        params, _ = _optimize_group(params, S, center, outer_angle, group1)
        # Optimize group 2
        params, _ = _optimize_group(params, S, center, outer_angle, group2)
        # Joint optimization
        def obj(p):
            return _objective_smooth(p, S, center, outer_angle, beta=50.0)
        try:
            res = minimize(obj, params, method='L-BFGS-B',
                           options={'maxiter': 300, 'maxfun': 8000,
                                    'ftol': 1e-12, 'gtol': 1e-9})
            if res.fun < _objective_smooth(params, S, center, outer_angle, beta=50.0):
                params = res.x
        except Exception:
            pass
    
    return params

def _check_validity(params, S, center, outer_angle, tol=1e-5):
    """Check if packing is valid (no overlaps, all contained)."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    for iv in inner_list:
        if _containment_violation(iv, outer) > tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            if _polygon_overlap_depth(inner_list[i], inner_list[j]) > tol:
                return False
    return True

def _polish(params, S, center, outer_angle):
    """High-precision polish pass."""
    def obj(p):
        return _objective_smooth(p, S, center, outer_angle, beta=100.0)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': 500, 'maxfun': 15000,
                                'ftol': 1e-14, 'gtol': 1e-10})
        return res.x
    except Exception:
        return params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin

    best = None
    best_S = float('inf')

    centers = [(0.0, 0.0)]
    outer_angles = [0.0, 30.0]

    inits = []
    for nr, nc in [(2, 6), (3, 4), (4, 3), (6, 2)]:
        if nr * nc == N:
            inits.append(_make_grid(nr, nc))
    inits.append(_ring_init(2.0))
    inits.append(_ring_init(2.5))
    inits.append(_linear_init(4.0))
    inits.append(_linear_init(3.5))
    rng = np.random.default_rng(42)
    for _ in range(3):
        r = rng.uniform(0, 2.5, size=(N, 2))
        inits.append(r - r.mean(axis=0))

    # Finer S candidates
    S_candidates = [5.0, 4.5, 4.2, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0]

    for S in S_candidates:
        if time.time() > DEADLINE - 30:
            break
        for oa in outer_angles:
            if time.time() > DEADLINE - 15:
                break
            for ctr in centers:
                if time.time() > DEADLINE - 10:
                    break
                for init_pts in inits:
                    if time.time() > DEADLINE - 8:
                        break
                    init_params = np.zeros(3 * N)
                    for i in range(N):
                        init_params[3*i] = init_pts[i, 0]
                        init_params[3*i+1] = init_pts[i, 1]
                        init_params[3*i+2] = 0.0
                    
                    # Hierarchical optimization
                    opt_params = _hierarchical_optimize(init_params, S, ctr, oa, max_iter=2)
                    
                    # Check validity
                    if _check_validity(opt_params, S, ctr, oa, tol=1e-5):
                        # Polish
                        polished = _polish(opt_params, S, ctr, oa)
                        if _check_validity(polished, S, ctr, oa, tol=1e-6):
                            if S < best_S:
                                best_S = S
                                best = (S, polished, ctr, oa)
                        else:
                            if S < best_S:
                                best_S = S
                                best = (S, opt_params, ctr, oa)
                        break
                if time.time() > DEADLINE - 10:
                    break
            if time.time() > DEADLINE - 10:
                break

    if best is None:
        grid = _make_grid(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    S, params, ctr, oa = best
    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)

# EVOLVE-BLOCK-END

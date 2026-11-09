"""Unit regular hexagons inside a regular hexagon n=12, minimize outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def _total_penalty_xy(params, S, center, outer_angle):
    """Total penalty for fixed angle 0. params is [x0, y0, x1, y1, ...]"""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[2*i], params[2*i+1], 1.0, 0.0))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    total = 0.0
    for iv in inner_list:
        pen = _containment_violation(iv, outer)
        if pen > 0:
            total += pen * pen
    for i in range(N):
        for j in range(i+1, N):
            d = _polygon_overlap_depth(inner_list[i], inner_list[j])
            if d > 0:
                total += d * d
    return total

def _check_validity_xy(params, S, center, outer_angle, tol=1e-7):
    """Check if packing is valid (no overlaps, all contained) for fixed angle 0."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[2*i], params[2*i+1], 1.0, 0.0))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    for iv in inner_list:
        if _containment_violation(iv, outer) > tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            if _polygon_overlap_depth(inner_list[i], inner_list[j]) > tol:
                return False
    return True

def _refine_local_xy(params, S, center, outer_angle, deadline):
    """L-BFGS-B local refinement for fixed angle 0."""
    def obj(p):
        return _total_penalty_xy(p, S, center, outer_angle)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': 2000, 'maxfun': 50000,
                                'ftol': 1e-15, 'gtol': 1e-12})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _make_staggered_grid(rows, cols):
    """Generate a staggered hexagonal grid of centers."""
    pos = []
    for r in range(rows):
        for c in range(cols):
            # Flat-top hexagons: horizontal distance 2, vertical sqrt(3)
            # Staggered: odd rows shifted by 1
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * np.sqrt(3.0)
            pos.append([x, y])
    arr = np.array(pos)
    # Center the grid
    arr -= arr.mean(axis=0)
    return arr

def _construct_initial_configs():
    """Generate a variety of initial configurations."""
    configs = []
    
    # 3x4 Staggered Grid
    if 3 * 4 == N:
        configs.append(_make_staggered_grid(3, 4))
    # 4x3 Staggered Grid
    if 4 * 3 == N:
        configs.append(_make_staggered_grid(4, 3))
    
    # Ring: 1 center, 6 around, 5 more? 
    # Let's just use a compact cluster
    # A 2x6 grid
    if 2 * 6 == N:
        configs.append(_make_staggered_grid(2, 6))
        
    # Random clusters
    rng = np.random.default_rng(42)
    for _ in range(10):
        # Generate random points in a small area
        pts = rng.uniform(-2, 2, size=(N, 2))
        configs.append(pts)
        
    return configs

def _binary_search_min_S_xy(init_params_xy, center, outer_angle, deadline,
                             S_lo=0.99, S_hi=5.0, max_iters=15):
    """Binary search for minimum S for fixed angle 0 packing."""
    best_S = S_hi
    best_params = None
    
    # First check if S_hi works (it should, as it's large)
    p = init_params_xy.copy()
    p = _refine_local_xy(p, S_hi, center, outer_angle, deadline)
    if _check_validity_xy(p, S_hi, center, outer_angle):
        best_params = p
        best_S = S_hi
    else:
        # If even S_hi doesn't work, try to fix it
        # This might happen if the initial config is too spread out
        # But S_hi=5.0 is very large for 12 unit hexagons
        return None, None

    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break
        S_mid = (S_lo + S_hi) / 2.0
        # Start from the last valid params if available, else initial
        p = best_params.copy() if best_params is not None else init_params_xy.copy()
        
        # Refine at S_mid
        p = _refine_local_xy(p, S_mid, center, outer_angle, deadline)
        
        if _check_validity_xy(p, S_mid, center, outer_angle):
            best_S = S_mid
            best_params = p
            S_hi = S_mid
        else:
            S_lo = S_mid
            
    return best_S, best_params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin

    best_S = float('inf')
    best_result = None

    center = (0.0, 0.0)
    outer_angle = 0.0
    
    inits = _construct_initial_configs()
    
    # Try a few outer angles to see if rotation helps
    # Usually 0 or 30 is best for hexagons
    outer_angles = [0.0, 30.0]

    for oa in outer_angles:
        if time.time() > DEADLINE - 10:
            break
        for init_pts in inits:
            if time.time() > DEADLINE - 10:
                break
            
            # Prepare params for fixed angle 0 (relative to outer)
            # Note: _hex_verts uses angle_deg for the inner hexagon.
            # We fix inner angle to 0.
            init_params_xy = np.zeros(2 * N)
            for i in range(N):
                init_params_xy[2*i] = init_pts[i, 0]
                init_params_xy[2*i+1] = init_pts[i, 1]
                
            S_found, params_found = _binary_search_min_S_xy(
                init_params_xy, center, oa, DEADLINE,
                S_lo=0.99, S_hi=5.0, max_iters=12
            )
            
            if params_found is not None and S_found < best_S:
                best_S = S_found
                # Store result in the format expected by the interface
                # inner: (x, y, angle_degrees)
                inner = []
                for i in range(N):
                    inner.append([float(params_found[2*i]), float(params_found[2*i+1]), 0.0])
                best_result = (inner, center, best_S, oa)
                
                # Early exit if we hit the theoretical limit
                if best_S < 1.00001:
                    return inner, (float(center[0]), float(center[1])), float(best_S), float(oa)

    if best_result is None:
        # Fallback: return a valid but not optimal configuration
        grid = _make_staggered_grid(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    return best_result

# EVOLVE-BLOCK-END

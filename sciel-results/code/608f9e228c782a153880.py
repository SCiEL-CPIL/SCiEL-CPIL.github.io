"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
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

def _check_validity_xy(params, S, center, outer_angle, tol=1e-6):
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
                       options={'maxiter': 500, 'maxfun': 10000,
                                'ftol': 1e-14, 'gtol': 1e-10})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _make_grid(nrows, ncols):
    """Generate hexagonal lattice positions."""
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
    """Generate ring configuration (1 center + 6 ring + 5 more)."""
    pts = [[0.0, 0.0]]
    for i in range(6):
        a = np.radians(60*i)
        pts.append([radius*np.cos(a), radius*np.sin(a)])
    for i in range(6):
        a = np.radians(60*i + 30)
        pts.append([(radius*1.5)*np.cos(a), (radius*1.5)*np.sin(a)])
    return np.array(pts) - np.array(pts).mean(axis=0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial configurations (positions only, angles fixed to 0)
    configs = []
    
    # 3x4 grid
    pos_3x4 = _make_grid(3, 4)
    configs.append(pos_3x4.flatten().tolist())
    
    # 4x3 grid
    pos_4x3 = _make_grid(4, 3)
    configs.append(pos_4x3.flatten().tolist())
    
    # 2x6 grid
    pos_2x6 = _make_grid(2, 6)
    configs.append(pos_2x6.flatten().tolist())
    
    # 6x2 grid
    pos_6x2 = _make_grid(6, 2)
    configs.append(pos_6x2.flatten().tolist())
    
    # Ring
    pos_ring = _ring_init(2.0)
    configs.append(pos_ring.flatten().tolist())
    
    # Linear
    xs = np.linspace(-5, 5, N)
    configs.append(np.column_stack([xs, np.zeros(N)]).flatten().tolist())
    
    best_score = float('inf')
    best_result = None
    
    # Try different outer hexagon orientations
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        if time.time() > deadline - 20:
            break
            
        outer_center = (0.0, 0.0)
        
        # Binary search on outer side length S
        # Lower bound: theoretical min is roughly 2*sqrt(3) + something, but let's start safe
        # Upper bound: definitely fits in 6.0
        lo, hi = 3.0, 6.0
        
        # First, ensure hi is feasible
        found_hi = False
        for config in configs:
            if time.time() > deadline - 15:
                break
            p = np.array(config)
            p = _refine_local_xy(p, hi, outer_center, outer_angle, deadline)
            if _check_validity_xy(p, hi, outer_center, outer_angle):
                found_hi = True
                if hi < best_score:
                    best_score = hi
                    best_result = (p, outer_center, hi, outer_angle)
                break
        
        if not found_hi:
            continue
            
        # Binary search
        for _ in range(8):
            if time.time() > deadline - 10:
                break
                
            mid = (lo + hi) / 2.0
            found_mid = False
            
            # Try to fit at mid using best known params or initial configs
            # Use the best_params from previous successful step if available, else try configs
            # To be robust, we try a few configs
            
            candidates = []
            if best_result is not None and best_result[2] > mid:
                candidates.append(best_result[0])
            
            for config in configs[:3]: # Limit configs for speed in inner loop
                candidates.append(np.array(config))
                
            for p in candidates:
                if time.time() > deadline - 5:
                    break
                p = _refine_local_xy(p, mid, outer_center, outer_angle, deadline)
                if _check_validity_xy(p, mid, outer_center, outer_angle):
                    found_mid = True
                    if mid < best_score:
                        best_score = mid
                        best_result = (p, outer_center, mid, outer_angle)
                    break
                    
            if found_mid:
                hi = mid
            else:
                lo = mid
                
    # Final refinement if we have a result
    if best_result is not None and time.time() < deadline - 5:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Try to squeeze a bit more
        for delta in [0.05, 0.02, 0.01]:
            if time.time() > deadline - 2:
                break
            new_side = outer_side - delta
            if new_side < 3.0:
                break
            p = _refine_local_xy(opt_params, new_side, outer_center, outer_angle, deadline)
            if _check_validity_xy(p, new_side, outer_center, outer_angle):
                outer_side = new_side
                opt_params = p
        best_result = (opt_params, outer_center, outer_side, outer_angle)
        
    # Fallback
    if best_result is None:
        inner = []
        pos = _make_grid(3, 4)
        for i in range(N):
            inner.append([float(pos[i,0]), float(pos[i,1]), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[2*i])
        y = float(opt_params[2*i+1])
        inner.append([x, y, 0.0])
        
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END

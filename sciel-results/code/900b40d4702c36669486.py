"""Unit regular hexagons inside a regular hexagon n=12, minimize outer side."""
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

def _total_penalty(params, S, center, outer_angle):
    """Total penalty: sum of squares of containment + overlap. 0 means valid."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
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

def _check_validity(params, S, center, outer_angle, tol=1e-6):
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

def _hexagonal_lattice_init(S):
    """Generate points on a hexagonal lattice within radius S."""
    pts = []
    step = 1.9
    max_r = S * 0.8
    for r in range(-4, 5):
        for c in range(-4, 5):
            x = c * step + (step * 0.5 if r % 2 else 0)
            y = r * step * np.sqrt(3) / 2
            if np.sqrt(x*x + y*y) <= max_r:
                pts.append([x, y])
    if len(pts) < N:
        # Pad with random points
        while len(pts) < N:
            a = np.random.uniform(0, 2*np.pi)
            rad = np.random.uniform(0, max_r)
            pts.append([rad*np.cos(a), rad*np.sin(a)])
    pts = np.array(pts[:N])
    pts -= pts.mean(axis=0)
    return pts

def _refine_local(params, S, center, outer_angle, deadline):
    """L-BFGS-B local refinement with strict penalty."""
    def obj(p):
        return _total_penalty(p, S, center, outer_angle)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': 500, 'maxfun': 15000,
                                'ftol': 1e-16, 'gtol': 1e-12})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _polish(params, S, center, outer_angle, deadline):
    """High-precision polish: tighter tolerances, more iterations."""
    def obj(p):
        return _total_penalty(p, S, center, outer_angle)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': 2000, 'maxfun': 50000,
                                'ftol': 1e-20, 'gtol': 1e-16})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _squeeze(params, S, center, outer_angle, deadline):
    """Try to reduce S slightly by re-optimizing at progressively smaller S values."""
    best_p = params.copy()
    best_S = S
    # Try a few small reductions
    for frac in [0.005, 0.003, 0.002, 0.001, 0.0005]:
        if time.time() > deadline - 2:
            break
        S_try = best_S * (1.0 - frac)
        p_try = best_p.copy()
        p_try = _refine_local(p_try, S_try, center, outer_angle, deadline)
        if _check_validity(p_try, S_try, center, outer_angle):
            best_p = p_try
            best_S = S_try
    return best_S, best_p

def _binary_search_min_S(init_params, center, outer_angle, deadline,
                          S_lo=3.0, S_hi=5.0, max_iters=12):
    """Binary search for minimum S such that a valid packing exists."""
    best_S = S_hi
    best_params = None
    
    # First check if S_hi works
    p = init_params.copy()
    p = _refine_local(p, S_hi, center, outer_angle, deadline)
    if _check_validity(p, S_hi, center, outer_angle):
        best_params = p
        best_S = S_hi
    else:
        return None, None
    
    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break
        S_mid = (S_lo + S_hi) / 2.0
        p = best_params.copy() if best_params is not None else init_params.copy()
        p = _refine_local(p, S_mid, center, outer_angle, deadline)
        if _check_validity(p, S_mid, center, outer_angle):
            best_S = S_mid
            best_params = p
            S_hi = S_mid
        else:
            S_lo = S_mid
    
    return best_S, best_params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin before 150s limit

    best_S = float('inf')
    best_result = None

    centers = [(0.0, 0.0)]
    outer_angles = [0.0, 30.0, 15.0, 45.0, 7.5, 22.5]

    # Generate initial configurations
    inits = []
    for nr, nc in [(2, 6), (3, 4), (4, 3), (6, 2)]:
        if nr * nc == N:
            inits.append(_make_grid(nr, nc))
    inits.append(_ring_init(2.0))
    inits.append(_ring_init(2.5))
    inits.append(_ring_init(3.0))
    inits.append(_linear_init(4.0))
    inits.append(_linear_init(3.5))
    inits.append(_linear_init(3.0))
    inits.append(_hexagonal_lattice_init(4.0))
    inits.append(_hexagonal_lattice_init(3.5))
    
    rng = np.random.default_rng(42)
    for _ in range(10):
        r = rng.uniform(-2.5, 2.5, size=(N, 2))
        inits.append(r - r.mean(axis=0))

    # Phase 1: broad search with multiple outer angles and inits
    for oa in outer_angles:
        if time.time() > DEADLINE - 25:
            break
        for ctr in centers:
            if time.time() > DEADLINE - 20:
                break
            for init_pts in inits:
                if time.time() > DEADLINE - 15:
                    break
                
                init_params = np.zeros(3 * N)
                for i in range(N):
                    init_params[3*i] = init_pts[i, 0]
                    init_params[3*i+1] = init_pts[i, 1]
                    init_params[3*i+2] = 0.0
                
                # Binary search for minimum S
                S_found, params_found = _binary_search_min_S(
                    init_params, ctr, oa, DEADLINE,
                    S_lo=3.0, S_hi=5.0, max_iters=12
                )
                
                if params_found is not None and S_found < best_S:
                    # Final polish
                    polished = _polish(params_found, S_found, ctr, oa, DEADLINE)
                    if _check_validity(polished, S_found, ctr, oa):
                        # Try to squeeze further
                        S_squeezed, p_squeezed = _squeeze(polished, S_found, ctr, oa, DEADLINE)
                        if S_squeezed < best_S:
                            best_S = S_squeezed
                            best_result = (S_squeezed, p_squeezed, ctr, oa)
                        else:
                            best_S = S_found
                            best_result = (S_found, polished, ctr, oa)
                    else:
                        best_S = S_found
                        best_result = (S_found, params_found, ctr, oa)
                
                # Early exit if we found a very good solution
                if best_S < 3.30:
                    break
            if time.time() > DEADLINE - 15:
                break
        if time.time() > DEADLINE - 15:
            break

    # Phase 2: random restarts around best solution
    if best_result is not None and best_S > 3.30:
        S_cur, params_cur, ctr_cur, oa_cur = best_result
        rng2 = np.random.default_rng(123)
        for _ in range(20):
            if time.time() > DEADLINE - 10:
                break
            
            # Perturb the current best solution
            perturbation = rng2.normal(0, 0.08, size=(3*N,))
            perturbation[2::3] *= 3.0  # moderate perturbations for angles
            p_perturbed = params_cur + perturbation
            
            # Try slightly smaller S
            S_try = S_cur * (1.0 - rng2.uniform(0.0, 0.015))
            p_refined = _refine_local(p_perturbed, S_try, ctr_cur, oa_cur, DEADLINE)
            if _check_validity(p_refined, S_try, ctr_cur, oa_cur):
                polished = _polish(p_refined, S_try, ctr_cur, oa_cur, DEADLINE)
                if _check_validity(polished, S_try, ctr_cur, oa_cur):
                    S_sq, p_sq = _squeeze(polished, S_try, ctr_cur, oa_cur, DEADLINE)
                    if S_sq < best_S:
                        best_S = S_sq
                        best_result = (S_sq, p_sq, ctr_cur, oa_cur)
                    elif S_try < best_S:
                        best_S = S_try
                        best_result = (S_try, polished, ctr_cur, oa_cur)
                elif S_try < best_S:
                    best_S = S_try
                    best_result = (S_try, p_refined, ctr_cur, oa_cur)
            
            # Also try the same S with perturbation
            p_refined2 = _refine_local(p_perturbed, S_cur, ctr_cur, oa_cur, DEADLINE)
            if _check_validity(p_refined2, S_cur, ctr_cur, oa_cur):
                polished2 = _polish(p_refined2, S_cur, ctr_cur, oa_cur, DEADLINE)
                if _check_validity(polished2, S_cur, ctr_cur, oa_cur):
                    S_sq2, p_sq2 = _squeeze(polished2, S_cur, ctr_cur, oa_cur, DEADLINE)
                    if S_sq2 < best_S:
                        best_S = S_sq2
                        best_result = (S_sq2, p_sq2, ctr_cur, oa_cur)

    # Phase 3: Final high-precision polish on best result
    if best_result is not None:
        S_final, params_final, ctr_final, oa_final = best_result
        if time.time() < DEADLINE - 5:
            polished_final = _polish(params_final, S_final, ctr_final, oa_final, DEADLINE)
            if _check_validity(polished_final, S_final, ctr_final, oa_final):
                S_sq_final, p_sq_final = _squeeze(polished_final, S_final, ctr_final, oa_final, DEADLINE)
                if S_sq_final < S_final:
                    best_S = S_sq_final
                    best_result = (S_sq_final, p_sq_final, ctr_final, oa_final)
                else:
                    best_result = (S_final, polished_final, ctr_final, oa_final)

    if best_result is None:
        # Fallback: return a valid but not optimal configuration
        grid = _make_grid(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    S, params, ctr, oa = best_result
    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)

# EVOLVE-BLOCK-END

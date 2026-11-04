"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Structural change: LP-based feasibility search with sign-fixed linear constraints.
For a fixed orientation of each triangle, |det| >= t becomes a LINEAR constraint.
We solve LPs to maximize t, then re-derive signs from the solution and iterate.
Combined with diverse structured initializations and local polish."""
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
N_TRIPLES = N * (N - 1) * (N - 2) // 6  # 165

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute triple indices
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])

def _bary_to_cart(bary2):
    """bary2: (N,2) with columns (b1, b2). b0 = 1 - b1 - b2."""
    x = bary2[:, 0] + 0.5 * bary2[:, 1]
    y = (np.sqrt(3) / 2.0) * bary2[:, 1]
    return np.column_stack([x, y])

def _cart_to_bary2(pts):
    """pts: (N,2) -> (N,2) barycentric (b1, b2)."""
    y = pts[:, 1]
    x = pts[:, 0]
    b2 = 2.0 * y / np.sqrt(3)
    b1 = x - 0.5 * b2
    return np.column_stack([b1, b2])

def _clamp_bary2(bary2):
    """Clamp barycentric coords (N,2) to valid range."""
    b1, b2 = bary2[:, 0], bary2[:, 1]
    b2 = np.clip(b2, 0.0, 1.0)
    b1 = np.clip(b1, 0.0, 1.0 - b2)
    return np.column_stack([b1, b2])

def _all_areas(pts):
    """Compute all 165 triangle areas."""
    idx = _IDX_TRIPLES
    pi = pts[idx[:, 0]]
    pj = pts[idx[:, 1]]
    pk = pts[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)

def _min_area(pts):
    return _all_areas(pts).min()

def _signs_from_pts(pts):
    """Get sign of cross product for each triple (+1 or -1)."""
    idx = _IDX_TRIPLES
    pi = pts[idx[:, 0]]
    pj = pts[idx[:, 1]]
    pk = pts[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return np.sign(cross)

def _solve_lp(bary2, signs, t_min=0.0):
    """Solve LP: maximize t subject to sign_m * area_m >= t for all m, and barycentric bounds.
    Variables: b1_0..b1_10, b2_0..b2_10, t  (23 variables)
    Area of triangle (i,j,k) in barycentric coords:
      area = (sqrt(3)/4) * |b1_i(b2_j - b2_k) + b1_j(b2_k - b2_i) + b1_k(b2_i - b2_j)|
    """
    # Variables: x[0:11] = b1, x[11:22] = b2, x[22] = t
    n_vars = 2 * N + 1
    t_idx = 2 * N
    
    # Objective: minimize -t
    c = np.zeros(n_vars)
    c[t_idx] = -1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For each triple m with sign s_m: s_m * area_m >= t
    # => s_m * (sqrt(3)/4) * (b1_i*(b2_j - b2_k) + b1_j*(b2_k - b2_i) + b1_k*(b2_i - b2_j)) - t >= 0
    # => -s_m * (sqrt(3)/4) * (...) + t <= 0
    # This is LINEAR in the variables!
    
    coef = np.sqrt(3) / 4.0
    A_ub = []
    b_ub = []
    
    for m in range(N_TRIPLES):
        i, j, k = _IDX_TRIPLES[m]
        s = signs[m]
        row = np.zeros(n_vars)
        # Coefficient of b1_i: s * coef * (b2_j - b2_k)
        # But b2_j and b2_k are variables, so this is NOT linear!
        # Wait - area in barycentric: for points in the equilateral triangle,
        # the area formula involves products of barycentric coords, which is NOT linear.
        
        # Let me reconsider. The area of triangle formed by three points in barycentric
        # coordinates is: area = (sqrt(3)/4) * |det of the 3x3 matrix of barycentric coords|
        # where each row is (b0, b1, b2) for one point.
        # det = b0_i(b1_j*b2_k - b2_j*b1_k) - b1_i(b0_j*b2_k - b2_j*b0_k) + b2_i(b0_j*b1_k - b1_j*b0_k)
        # This is CUBIC in barycentric coords, not linear.
        
        # So LP in barycentric doesn't work directly. Let me use Cartesian instead.
        # In Cartesian: area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
        # This is also bilinear (products of x and y), not linear.
        
        # The LP approach needs a different formulation. Let me use the fact that
        # for FIXED signs, we need: sign_m * (cross product) >= 2t
        # The cross product is bilinear, so this is still not linear.
        
        # Alternative: use the constraint in terms of the cross product directly.
        # cross_ijk = x_i*y_j - x_i*y_k - x_j*y_i + x_j*y_k + x_k*y_i - x_k*y_j
        # This is bilinear. Not LP-solvable directly.
        
        pass
    
    # The LP approach as stated doesn't work because area is bilinear in Cartesian
    # or cubic in barycentric. Let me use a different structural approach:
    # Use scipy's SLSQP with the constraint that each signed area >= t, which is
    # smooth when signs are fixed, and iterate on sign updates.
    return None

def _solve_slsqp_fixed_signs(bary2_init, signs, t_init=0.0):
    """Solve: maximize t subject to sign_m * area_m >= t for all m, barycentric bounds.
    Using SLSQP with smooth constraints (no abs, no min)."""
    n_vars = 2 * N + 1  # b1_0..b1_10, b2_0..b2_10, t
    t_idx = 2 * N
    
    def _area_signed_vec(x):
        """Compute all signed areas given barycentric coords."""
        b1 = x[:N]
        b2 = x[N:2*N]
        bary = np.column_stack([1.0 - b1 - b2, b1, b2])
        # Area of triangle (i,j,k) = (sqrt(3)/4) * det of 3x3 barycentric matrix
        # det = b0_i*(b1_j*b2_k - b2_j*b1_k) - b1_i*(b0_j*b2_k - b2_j*b0_k) + b2_i*(b0_j*b1_k - b1_j*b0_k)
        idx = _IDX_TRIPLES
        areas = np.zeros(N_TRIPLES)
        for m in range(N_TRIPLES):
            i, j, k = idx[m]
            det = (bary[i, 0] * (bary[j, 1] * bary[k, 2] - bary[j, 2] * bary[k, 1])
                   - bary[i, 1] * (bary[j, 0] * bary[k, 2] - bary[j, 2] * bary[k, 0])
                   + bary[i, 2] * (bary[j, 0] * bary[k, 1] - bary[j, 1] * bary[k, 0]))
            areas[m] = (np.sqrt(3) / 4.0) * det
        return areas
    
    def _objective(x):
        return -x[t_idx]
    
    def _objective_grad(x):
        g = np.zeros(n_vars)
        g[t_idx] = -1.0
        return g
    
    # Constraints: sign_m * area_m - t >= 0
    def _constraints(x):
        b1 = x[:N]
        b2 = x[N:2*N]
        t = x[t_idx]
        bary = np.column_stack([1.0 - b1 - b2, b1, b2])
        idx = _IDX_TRIPLES
        vals = np.zeros(N_TRIPLES)
        for m in range(N_TRIPLES):
            i, j, k = idx[m]
            det = (bary[i, 0] * (bary[j, 1] * bary[k, 2] - bary[j, 2] * bary[k, 1])
                   - bary[i, 1] * (bary[j, 0] * bary[k, 2] - bary[j, 2] * bary[k, 0])
                   + bary[i, 2] * (bary[j, 0] * bary[k, 1] - bary[j, 1] * bary[k, 0]))
            areas[m] = (np.sqrt(3) / 4.0) * det
        return signs * areas - t
    
    # Vectorized constraint evaluation for speed
    def _constraints_vec(x):
        b1 = x[:N]
        b2 = x[N:2*N]
        t = x[t_idx]
        bary = np.column_stack([1.0 - b1 - b2, b1, b2])
        idx = _IDX_TRIPLES
        bi = bary[idx[:, 0]]
        bj = bary[idx[:, 1]]
        bk = bary[idx[:, 2]]
        dets = (bi[:, 0] * (bj[:, 1] * bk[:, 2] - bj[:, 2] * bk[:, 1])
                - bi[:, 1] * (bj[:, 0] * bk[:, 2] - bj[:, 2] * bk[:, 0])
                + bi[:, 2] * (bj[:, 0] * bk[:, 1] - bj[:, 1] * bk[:, 0]))
        areas = (np.sqrt(3) / 4.0) * dets
        return signs * areas - t
    
    # Bounds: b1 >= 0, b2 >= 0, b1 + b2 <= 1, t >= 0
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # b1_i
    for i in range(N):
        bounds.append((0.0, 1.0))  # b2_i
    bounds.append((0.0, 1.0))  # t
    
    # Inequality constraints for b1_i + b2_i <= 1: b1_i + b2_i - 1 <= 0
    # In SLSQP, inequality constraints are of the form: fun(x) >= 0
    # So we need: 1 - b1_i - b2_i >= 0
    def _sum_constraints(x):
        b1 = x[:N]
        b2 = x[N:2*N]
        return 1.0 - b1 - b2
    
    cons = [
        {'type': 'ineq', 'fun': _constraints_vec},
    ]
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[i] - x[N + i]})
    
    x0 = np.concatenate([bary2_init[:, 0], bary2_init[:, 1], [t_init]])
    
    try:
        res = minimize(_objective, x0, method='SLSQP', jac=_objective_grad,
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-12})
        if res.success:
            bary2_opt = np.column_stack([res.x[:N], res.x[N:2*N]])
            t_opt = res.x[t_idx]
            return bary2_opt, t_opt
        else:
            return bary2_init, t_init
    except Exception:
        return bary2_init, t_init

def _sign_refine_loop(bary2_init, max_iters=50, time_limit=10.0):
    """Iterate: solve with fixed signs, update signs from solution, repeat."""
    start_time = time.time()
    bary2 = bary2_init.copy()
    t_best = 0.0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        pts = _bary_to_cart(bary2)
        signs = _signs_from_pts(pts)
        
        # Compute current min area as t_init
        areas = _all_areas(pts)
        t_init = max(areas.min() * 0.95, 0.0001)
        
        bary2_new, t_val = _solve_slsqp_fixed_signs(bary2, signs, t_init)
        
        # Check if signs are consistent
        pts_new = _bary_to_cart(bary2_new)
        signs_new = _signs_from_pts(pts_new)
        
        # Score the new solution
        min_a = _min_area(pts_new)
        if min_a > t_best:
            t_best = min_a
            bary2 = bary2_new.copy()
        
        # Check convergence: signs stable
        if np.all(signs_new == signs):
            # Signs are stable, one more iteration to confirm
            bary2_new2, t_val2 = _solve_slsqp_fixed_signs(bary2, signs_new, t_val)
            pts_check = _bary_to_cart(bary2_new2)
            signs_check = _signs_from_pts(pts_check)
            min_a2 = _min_area(pts_check)
            if min_a2 > t_best:
                t_best = min_a2
                bary2 = bary2_new2.copy()
            if np.all(signs_check == signs_new):
                break
    
    return bary2, t_best

def _structured_init_1(rng):
    """Center + two pentagonal rings."""
    pts = []
    cx, cy = 0.5, np.sqrt(3) / 6.0
    pts.append([cx, cy])
    for i in range(5):
        a = 2 * np.pi * i / 5 + rng.uniform(-0.05, 0.05)
        r = 0.12 + rng.uniform(-0.01, 0.01)
        pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    for i in range(5):
        a = 2 * np.pi * i / 5 + np.pi / 5 + rng.uniform(-0.05, 0.05)
        r = 0.25 + rng.uniform(-0.01, 0.01)
        pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    pts = np.array(pts)
    bary2 = _cart_to_bary2(pts)
    return _clamp_bary2(bary2)

def _structured_init_2(rng):
    """Hexagonal lattice subset."""
    pts = []
    cx, cy = 0.5, np.sqrt(3) / 6.0
    # 1 center
    pts.append([cx, cy])
    # 6 around center
    for i in range(6):
        a = 2 * np.pi * i / 6 + np.pi / 6
        r = 0.15 + rng.uniform(-0.01, 0.01)
        pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    # 4 more at corners-ish
    for i in range(4):
        a = 2 * np.pi * i / 4 + np.pi / 4
        r = 0.28 + rng.uniform(-0.01, 0.01)
        pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    pts = np.array(pts)
    bary2 = _cart_to_bary2(pts)
    return _clamp_bary2(bary2)

def _structured_init_3(rng):
    """3x3 grid + 2 extra, clipped to triangle."""
    pts = []
    for i in range(3):
        for j in range(3):
            x = 0.15 + 0.35 * i / 2.0 + rng.uniform(-0.01, 0.01)
            y = 0.08 + 0.35 * j / 2.0 + rng.uniform(-0.01, 0.01)
            pts.append([x, y])
    # 2 more
    pts.append([0.5 + rng.uniform(-0.02, 0.02), 0.2 + rng.uniform(-0.02, 0.02)])
    pts.append([0.3 + rng.uniform(-0.02, 0.02), 0.4 + rng.uniform(-0.02, 0.02)])
    pts = np.array(pts)
    # Clip to triangle
    bary2 = _cart_to_bary2(pts)
    return _clamp_bary2(bary2)

def _random_init(rng):
    """Random barycentric init."""
    b2 = rng.uniform(0.05, 0.9, size=N)
    b1 = rng.uniform(0.0, 1.0 - b2, size=N)
    bary2 = np.column_stack([b1, b2])
    return _clamp_bary2(bary2)

def _local_polish(bary2, time_limit=5.0):
    """Polish with Nelder-Mead on the hard objective."""
    start_time = time.time()
    
    def _obj(flat):
        b1, b2 = flat[:N], flat[N:2*N]
        bary2 = np.column_stack([b1, b2])
        bary2 = _clamp_bary2(bary2)
        pts = _bary_to_cart(bary2)
        return -_min_area(pts)
    
    x0 = bary2.flatten()
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    
    try:
        res = minimize(_obj, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        bary2_opt = _clamp_bary2(np.column_stack([res.x[:N], res.x[N:2*N]]))
        if time.time() - start_time < time_limit:
            return bary2_opt
    except Exception:
        pass
    return bary2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    rng = np.random.RandomState(42)
    
    best_bary2 = None
    best_score = -np.inf
    
    # Generate diverse initializations
    inits = []
    for i in range(4):
        if time_left() < 40.0:
            break
        inits.append(_structured_init_1(rng))
    for i in range(3):
        if time_left() < 35.0:
            break
        inits.append(_structured_init_2(rng))
    for i in range(3):
        if time_left() < 30.0:
            break
        inits.append(_structured_init_3(rng))
    for i in range(5):
        if time_left() < 25.0:
            break
        inits.append(_random_init(rng))
    
    # Run sign-refinement loop on each initialization
    for init_bary2 in inits:
        if time_left() < 20.0:
            break
        try:
            bary2_opt, t_val = _sign_refine_loop(init_bary2, max_iters=30, time_limit=8.0)
            pts = _bary_to_cart(bary2_opt)
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_bary2 = bary2_opt.copy()
        except Exception:
            pass
    
    # Local polish on best
    if best_bary2 is not None and time_left() > 15.0:
        try:
            polished = _local_polish(best_bary2, time_limit=min(10.0, time_left() - 10.0))
            pts = _bary_to_cart(polished)
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_bary2 = polished.copy()
        except Exception:
            pass
    
    # Perturbation-based restarts on best
    if best_bary2 is not None and time_left() > 15.0:
        for i in range(8):
            if time_left() < 8.0:
                break
            try:
                noise = rng.normal(0, 0.02, size=2 * N)
                perturbed = _clamp_bary2(best_bary2 + noise)
                bary2_opt, t_val = _sign_refine_loop(perturbed, max_iters=15, time_limit=4.0)
                pts = _bary_to_cart(bary2_opt)
                score = _min_area(pts)
                if score > best_score:
                    best_score = score
                    best_bary2 = bary2_opt.copy()
            except Exception:
                pass
    
    # Final polish
    if best_bary2 is not None and time_left() > 8.0:
        try:
            polished = _local_polish(best_bary2, time_limit=min(5.0, time_left() - 5.0))
            pts = _bary_to_cart(polished)
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_bary2 = polished.copy()
        except Exception:
            pass
    
    # Fallback
    if best_bary2 is None:
        best_bary2 = _structured_init_1(rng)
    
    pts = _bary_to_cart(best_bary2)
    return np.array(pts)

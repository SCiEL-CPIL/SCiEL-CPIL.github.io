import numpy as np
from scipy.optimize import minimize, linprog
import time
import itertools
from scipy.spatial import ConvexHull, Delaunay

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
n_triples = len(_IDX_TRIPLES)

def _bary_to_cart(u, v):
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _cart_to_bary(x, y):
    w = 2 * y / np.sqrt(3)
    v = x - w * 0.5
    u = 1 - v - w
    return u, v

def _project_to_bary_simplex(bary):
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _all_dets_from_bary(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _score_bary(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets)) / (2 * TRI_AREA)

def _solve_lp_for_point(i, bary_others, signs, t_target, time_budget=2.0):
    """
    Solve LP to optimize point i, given other points and fixed signs.
    Variables: u_i, v_i, t
    Maximize t subject to:
      For each triple involving i: sign * det >= t
      u_i >= 0, v_i >= 0, u_i + v_i <= 1
    """
    start_time = time.time()
    if time.time() - start_time > time_budget:
        return None
    
    bary_others = bary_others.copy()
    u_o = bary_others[:, 0]
    v_o = bary_others[:, 1]
    w_o = 1.0 - u_o - v_o
    
    # Find all triples involving point i
    triples_with_i = []
    for idx, (a, b, c) in enumerate(_IDX_TRIPLES):
        if a == i or b == i or c == i:
            triples_with_i.append((idx, a, b, c))
    
    n_vars = 3  # u_i, v_i, t
    c = np.array([0.0, 0.0, -1.0])  # maximize t => minimize -t
    
    A_ub = []
    b_ub = []
    
    for _, a, b, c_idx in triples_with_i:
        # det(a,b,c) where one is i
        # We need the coefficient of u_i and v_i in the determinant
        # det = u_a*(v_b*w_c - w_b*v_c) - v_a*(u_b*w_c - w_b*u_c) + w_a*(u_b*v_c - v_b*u_c)
        
        # Get the other two points
        others = [x for x in (a, b, c_idx) if x != i]
        p1, p2 = others[0], others[1]
        
        u1, v1, w1 = u_o[p1], v_o[p1], w_o[p1]
        u2, v2, w2 = u_o[p2], v_o[p2], w_o[p2]
        
        # det as function of (u_i, v_i):
        # det = u_i * (v1*w2 - w1*v2) 
        #      - v_i * (u1*w2 - w1*u2)
        #      + (1 - u_i - v_i) * (u1*v2 - v1*u2)
        # = u_i * (v1*w2 - w1*v2 - u1*v2 + v1*u2)
        # + v_i * (-u1*w2 + w1*u2 - u1*v2 + v1*u2)
        # + (u1*v2 - v1*u2)
        
        A_coeff = (v1 * w2 - w1 * v2) - (u1 * v2 - v1 * u2)
        B_coeff = -(u1 * w2 - w1 * u2) - (u1 * v2 - v1 * u2)
        C_const = (u1 * v2 - v1 * u2)
        
        s = signs[_IDX_TRIPLES[_, 0], _IDX_TRIPLES[_, 1], _IDX_TRIPLES[_, 2]]
        
        # Constraint: s * (A*u_i + B*v_i + C) >= t
        # => -s*A*u_i - s*B*v_i + t <= s*C
        A_ub.append([-s * A_coeff, -s * B_coeff, 1.0])
        b_ub.append(s * C_const)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, 1), (0, 1), (0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.array([res.x[0], res.x[1]])
    except Exception:
        pass
    return None

def _lp_alternating_optimize(bary_init, n_sweeps=20, time_budget=30.0):
    """
    Alternating LP optimization.
    """
    start_time = time.time()
    bary = bary_init.copy()
    bary = _project_to_bary_simplex(bary)
    
    for sweep in range(n_sweeps):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current signs
        dets = _all_dets_from_bary(bary)
        signs = np.sign(dets)
        # For zero dets, keep previous or default to 1
        signs[signs == 0] = 1.0
        
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            
            bary_others = np.delete(bary, i, axis=0)
            new_point = _solve_lp_for_point(i, bary_others, signs, 0.0, time_budget=1.0)
            
            if new_point is not None:
                # Check if it's an improvement
                bary_test = bary.copy()
                bary_test[i] = new_point
                bary_test = _project_to_bary_simplex(bary_test)
                
                score_old = _score_bary(bary)
                score_new = _score_bary(bary_test)
                
                if score_new > score_old - 1e-10:
                    bary = bary_test
                    improved = True
                elif score_new > score_old:
                    bary = bary_test
                    improved = True
        
        if not improved:
            break
    
    return bary

def _delauay_init():
    """
    Generate points based on Delaunay triangulation of a grid.
    """
    # Create a fine grid
    n_grid = 10
    grid_u = []
    grid_v = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            grid_u.append(u)
            grid_v.append(v)
    
    grid = np.column_stack((grid_u, grid_v))
    
    # Use Delaunay to find well-distributed points
    try:
        # Add a dummy point far away to ensure triangulation works
        pts_3d = np.column_stack((grid[:, 0], grid[:, 1], np.zeros(len(grid))))
        hull = ConvexHull(pts_3d)
        # Just return a subset of the grid
        indices = np.linspace(0, len(grid) - 1, N, dtype=int)
        return grid[indices]
    except Exception:
        # Fallback: uniform grid
        return np.array([[i / N, j / N] for i in range(N) for j in range(N) if i + j < N][:N])

def _structured_inits():
    """Generate multiple structured initial configurations."""
    inits = []
    
    # 1. Vertices + center + edge midpoints
    bary1 = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ])
    inits.append(bary1)
    
    # 2. Grid-like
    bary2 = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.33, 0.0], [0.67, 0.0],
        [0.0, 0.33], [0.0, 0.67],
        [0.33, 0.33], [0.67, 0.33], [0.33, 0.67], [0.5, 0.5],
    ])
    inits.append(bary2)
    
    # 3. Concentric
    bary3 = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [1/3, 1/3],
        [0.2, 0.2], [0.6, 0.2], [0.2, 0.6],
        [0.4, 0.4], [0.8, 0.1], [0.1, 0.8], [0.5, 0.3],
    ])
    inits.append(bary3)
    
    # 4. Random with fixed seed
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = []
        for _ in range(N):
            u, v = rng.random(2)
            while u + v > 1:
                u, v = rng.random(2)
            pts.append([u, v])
        inits.append(np.array(pts))
    
    return inits

def _objective_smooth(x, beta=50.0, penalty_lambda=100.0):
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    excess_u = np.maximum(0, -u)
    penalty += penalty_lambda * np.sum(excess_u**2)
    excess_v = np.maximum(0, -v)
    penalty += penalty_lambda * np.sum(excess_v**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _local_optimize_lbfgs(x0, beta=50.0, maxiter=300, gtol=1e-10, time_budget=5.0):
    start_time = time.time()
    try:
        res = minimize(
            _objective_smooth, x0, method='L-BFGS-B',
            args=(beta,),
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        if time.time() - start_time < time_budget:
            return res.x
        return x0
    except Exception:
        return x0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
        
    best_bary = None
    best_score = -float('inf')
    
    inits = _structured_inits()
    
    # Optimize each initial point
    for init in inits:
        if time_left() < 10.0:
            break
        
        bary0 = _project_to_bary_simplex(init)
        
        # Phase 1: LP alternating optimization (structural change)
        if time_left() > 40.0:
            bary_lp = _lp_alternating_optimize(bary0, n_sweeps=15, time_budget=20.0)
            score_lp = _score_bary(bary_lp)
            if score_lp > best_score:
                best_score = score_lp
                best_bary = bary_lp.copy()
            bary0 = bary_lp
        else:
            bary0 = bary0.copy()
        
        # Phase 2: L-BFGS-B polishing
        if time_left() > 20.0:
            for beta in [30.0, 100.0, 300.0]:
                if time_left() < 5.0:
                    break
                x0 = bary0.flatten()
                x_opt = _local_optimize_lbfgs(x0, beta=beta, maxiter=200, gtol=1e-10, time_budget=3.0)
                bary0 = _project_to_bary_simplex(x_opt.reshape(N, 2))
                s = _score_bary(bary0)
                if s > best_score:
                    best_score = s
                    best_bary = bary0.copy()
        
        # Phase 3: Another LP round if time permits
        if time_left() > 15.0:
            bary_lp2 = _lp_alternating_optimize(bary0, n_sweeps=10, time_budget=8.0)
            score_lp2 = _score_bary(bary_lp2)
            if score_lp2 > best_score:
                best_score = score_lp2
                best_bary = bary_lp2.copy()
        
        # Phase 4: Final L-BFGS polish
        if time_left() > 10.0:
            x0 = bary0.flatten()
            x_opt = _local_optimize_lbfgs(x0, beta=500.0, maxiter=100, gtol=1e-12, time_budget=3.0)
            bary0 = _project_to_bary_simplex(x_opt.reshape(N, 2))
            s = _score_bary(bary0)
            if s > best_score:
                best_score = s
                best_bary = bary0.copy()
    
    # If we found a good solution, convert to Cartesian
    if best_bary is not None:
        pts_best = _bary_to_cart(best_bary[:, 0], best_bary[:, 1])
        return pts_best
    
    # Fallback: return random valid points
    pts = []
    for _ in range(N):
        u, v = np.random.random(2)
        while u + v > 1:
            u, v = np.random.random(2)
        pts.append([u + 0.5*v, np.sqrt(3)/2*v])
    return np.array(pts)

def score(points):
    """Compute the raw score: min triangle area / TRI_AREA."""
    bary = np.zeros((N, 2))
    for i in range(N):
        p = points[i]
        w = 2 * p[1] / np.sqrt(3)
        v = p[0] - w * 0.5
        u = 1 - v - w
        bary[i] = [u, v]
    
    dets = _all_dets_from_bary(bary)
    min_area = np.min(np.abs(dets))
    return min_area / (2 * TRI_AREA)

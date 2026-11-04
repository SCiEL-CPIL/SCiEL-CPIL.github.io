import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of equilateral triangle with side 1

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
n_triples = len(_IDX_TRIPLES)  # 165

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
    """
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

def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _objective_smooth(x, beta, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B using epigraph formulation.
    x: (2N+1,) array of u, v coords plus t.
    Minimize: t + penalty, subject to |det_i| >= t for all i (relaxed smoothly).
    """
    n_vars = 2 * N
    bary = x[:n_vars].reshape(N, 2)
    t = x[n_vars]
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords (u,v,w < 0)
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
        return t + penalty
        
    # Smooth constraint: -log(sum(exp(-beta*(areas - t)))) should be >= 0
    # i.e., t should be <= min(areas)
    # Penalty: max(0, t - smooth_min) where smooth_min = min - (1/beta)*log(sum(exp(-beta*(areas-min))))
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    # We want t <= smooth_min, so penalty if t > smooth_min
    constraint_violation = np.maximum(0.0, t - smooth_min)
    penalty += penalty_lambda * constraint_violation**2
    
    return t + penalty

def _objective_grad(x, beta, penalty_lambda=100.0):
    """
    Smooth objective with gradient for L-BFGS-B.
    x: (2N+1,) array of u, v coords plus t.
    """
    n_vars = 2 * N
    bary = x[:n_vars].reshape(N, 2)
    t = x[n_vars]
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    penalty = 0.0
    grad = np.zeros(2 * N + 1)
    
    # Penalty for u+v > 1
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    mask = (u + v - 1.0) > 0
    if np.any(mask):
        grad[mask, 0] += 2 * penalty_lambda * excess[mask]
        grad[mask, 1] += 2 * penalty_lambda * excess[mask]
    
    # Penalty for u < 0
    excess_u = np.maximum(0, -u)
    penalty += penalty_lambda * np.sum(excess_u**2)
    mask_u = u < 0
    if np.any(mask_u):
        grad[mask_u, 0] += 2 * penalty_lambda * excess_u[mask_u]
    
    # Penalty for v < 0
    excess_v = np.maximum(0, -v)
    penalty += penalty_lambda * np.sum(excess_v**2)
    mask_v = v < 0
    if np.any(mask_v):
        grad[mask_v, 1] += 2 * penalty_lambda * excess_v[mask_v]
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        grad[n_vars] = 1.0  # dt of t
        return t + penalty, grad
        
    m = np.min(areas)
    shifted = areas - m
    exps = np.exp(-beta * shifted)
    sum_exps = np.sum(exps)
    log_sum_exp = np.log(sum_exps)
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    constraint_violation = np.maximum(0.0, t - smooth_min)
    penalty += penalty_lambda * constraint_violation**2
    
    # Gradient w.r.t. t
    grad[n_vars] = 1.0  # from t
    if constraint_violation > 0:
        grad[n_vars] += 2 * penalty_lambda * constraint_violation
    
    # Gradient w.r.t. barycentric coords
    # d(smooth_min)/dbary = d(m)/dbary - (1/beta) * d(log_sum_exp)/dbary
    # d(log_sum_exp)/dbary = sum(-beta * d(areas_i)/dbary * exps_i / sum_exps)
    # = -beta * sum(d(areas_i)/dbary * w_i) where w_i = exps_i / sum_exps
    # So d(smooth_min)/dbary = d(m)/dbary + sum(d(areas_i)/dbary * w_i)
    # d(constraint_violation^2)/dbary = 2*cv * d(cv)/dbary = 2*cv * (-d(smooth_min)/dbary) if cv > 0
    # d(penalty)/dbary = 2*penalty_lambda*cv * (-d(smooth_min)/dbary)
    # Total: d(t + penalty)/dbary = -d(smooth_min)/dbary * (2*penalty_lambda*cv) [when cv > 0]
    
    if constraint_violation > 0:
        w_i = exps / sum_exps
        # Compute gradient of areas w.r.t. barycentric coords
        i_idx = _IDX_TRIPLES[:, 0]
        j_idx = _IDX_TRIPLES[:, 1]
        k_idx = _IDX_TRIPLES[:, 2]
        
        # For each triple, compute d(det)/dbary for the three involved points
        # det = u_i*(v_j*w_k - w_j*v_k) - v_i*(u_j*w_k - w_j*u_k) + w_i*(u_j*v_k - v_j*u_k)
        # w = 1-u-v, so dw/du = -1, dw/dv = -1
        
        # This is complex, use finite difference approximation for gradient
        # But for now, approximate: the gradient points in direction that increases min area
        # Use the weighted average of area gradients as a proxy
        
        # Simple approximation: compute gradient via finite differences
        eps = 1e-6
        grad_bary = np.zeros(2 * N)
        
        for dim in range(2 * N):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[dim] += eps
            x_minus[dim] -= eps
            
            bary_plus = _project_to_bary_simplex(x_plus[:n_vars].reshape(N, 2))
            bary_minus = _project_to_bary_simplex(x_minus[:n_vars].reshape(N, 2))
            
            dets_plus = _all_dets_from_bary(bary_plus)
            dets_minus = _all_dets_from_bary(bary_minus)
            
            areas_plus = np.abs(dets_plus)
            areas_minus = np.abs(dets_minus)
            
            m_plus = np.min(areas_plus)
            m_minus = np.min(areas_minus)
            
            shifted_plus = areas_plus - m_plus
            lse_plus = np.log(np.sum(np.exp(-beta * shifted_plus)))
            smooth_min_plus = m_plus - (1.0 / beta) * lse_plus
            
            shifted_minus = areas_minus - m_minus
            lse_minus = np.log(np.sum(np.exp(-beta * shifted_minus)))
            smooth_min_minus = m_minus - (1.0 / beta) * lse_minus
            
            grad_bary[dim] = (smooth_min_plus - smooth_min_minus) / (2 * eps)
        
        grad[:n_vars] = -2 * penalty_lambda * constraint_violation * grad_bary
    
    return t + penalty, grad

def _objective_slsqp(x):
    """
    Objective for SLSQP: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 20
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - np.array(selected), axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _local_optimize_nm(x0, maxiter=300, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, maxiter=200, gtol=1e-9):
    """Run L-BFGS-B local optimization from x0 with epigraph variable."""
    try:
        # Add t variable
        t_init = 0.001
        x0_ext = np.concatenate([x0, [t_init]])
        
        res = minimize(
            _objective_smooth, x0_ext, method='L-BFGS-B',
            args=(50.0,),
            bounds=[(0, 1)] * (2 * N) + [(0, 0.1)],
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x[:2*N]
    except Exception:
        return x0

def _local_optimize_slsqp(x0, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _simulated_annealing(x0, time_budget=10.0, t0=0.01, t_final=0.0001, cooling_rate=0.95):
    """
    Simulated annealing to escape local minima.
    x0: (2N,) barycentric coords.
    """
    start_time = time.time()
    x = x0.copy()
    bary = _project_to_bary_simplex(x.reshape(N, 2))
    current_score = _min_area_ratio_from_bary(bary)
    
    if current_score < 0:
        current_score = -1e-10
    
    best_x = x.copy()
    best_score = current_score
    
    t = t0
    rng = np.random.RandomState(42)
    
    while t > t_final and (time.time() - start_time) < time_budget:
        # Generate neighbor: perturb random point
        idx = rng.randint(N)
        delta = rng.normal(0, 0.05, size=2)
        x_new = x.copy()
        x_new[idx*2] += delta[0]
        x_new[idx*2+1] += delta[1]
        
        bary_new = _project_to_bary_simplex(x_new.reshape(N, 2))
        new_score = _min_area_ratio_from_bary(bary_new)
        
        if new_score < 0:
            new_score = -1e-10
        
        delta_score = new_score - current_score
        
        if delta_score > 0 or rng.random() < np.exp(delta_score / t):
            x = x_new
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
        
        t *= cooling_rate
    
    return best_x

def _final_polish(x0, maxiter=500):
    """
    High-precision final polish using Nelder-Mead with very tight tolerances.
    """
    try:
        res = minimize(
            _objective_slsqp, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def score(points):
    """Compute the raw score: min triangle area / TRI_AREA."""
    bary = np.zeros((N, 2))
    for i in range(N):
        p = points[i]
        # Convert Cartesian to barycentric
        # p = u*V0 + v*V1 + w*V2, w = 1-u-v
        # x = v + w*0.5, y = w*sqrt(3)/2
        # w = y / (sqrt(3)/2) = 2y/sqrt(3)
        # v = x - w*0.5 = x - y/sqrt(3)
        # u = 1 - v - w
        w = 2 * p[1] / np.sqrt(3)
        v = p[0] - w * 0.5
        u = 1 - v - w
        bary[i] = [u, v]
    
    dets = _all_dets_from_bary(bary)
    min_area = np.min(np.abs(dets))
    return min_area / (2 * TRI_AREA)  # det is 2*area in barycentric, so area = det/2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(15):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Magic Init
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # 4. Structured Starts
    # Start 1: Hexagonal lattice pattern
    bary1 = np.array([
        [0.0, 0.0],    # V0
        [1.0, 0.0],    # V1
        [0.0, 1.0],    # V2
        [0.5, 0.0],    # mid V0-V1
        [0.5, 0.5],    # mid V1-V2
        [0.0, 0.5],    # mid V2-V0
        [1/3, 1/3],    # centroid
        [0.25, 0.0],   # 1/4 V0-V1
        [0.75, 0.0],   # 3/4 V0-V1
        [0.25, 0.25],  # interior
        [0.5, 0.75],   # interior
    ])
    inits.append(_project_to_bary_simplex(bary1).flatten())
    
    # Start 2: Concentric hexagons
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary2).flatten())
    
    # Start 3: Symmetric grid with jitter
    rng3 = np.random.RandomState(123)
    bary3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    bary3 += rng3.normal(0, 0.02, size=bary3.shape)
    inits.append(_project_to_bary_simplex(bary3).flatten())
    
    # Start 4: Random with structured bias
    rng4 = np.random.RandomState(456)
    bary4 = []
    bary4.append([0.0, 0.0])
    bary4.append([1.0, 0.0])
    bary4.append([0.0, 1.0])
    bary4.append([1/3, 1/3])
    for _ in range(7):
        u = rng4.uniform(0.1, 0.9)
        v = rng4.uniform(0.1, 0.9)
        if u + v < 0.85:
            bary4.append([u, v])
        else:
            bary4.append([u * 0.8, v * 0.8])
    inits.append(_project_to_bary_simplex(np.array(bary4[:N])).flatten())
    
    # Phase 1: Local optimization from each init
    for init in inits:
        if time_left() < 10.0:
            break
        
        x0 = _project_to_bary_simplex(init.reshape(N, 2)).flatten()
        
        # Try multiple optimization methods
        for opt_func in [_local_optimize_nm, _local_optimize_lbfgs, _local_optimize_slsqp]:
            if time_left() < 10.0:
                break
            
            try:
                x_opt = opt_func(x0)
                x_opt = _project_to_bary_simplex(x_opt.reshape(N, 2)).flatten()
                
                # Evaluate
                bary_opt = x_opt.reshape(N, 2)
                s = _min_area_ratio_from_bary(bary_opt) / (2 * TRI_AREA)
                
                if s > best_score:
                    best_score = s
                    best_x = x_opt.copy()
                    
                # Warm start for next method
                x0 = x_opt
            except Exception:
                pass
    
    # Phase 2: Simulated annealing from best solution
    if best_x is not None and time_left() > 15.0:
        sa_time = min(15.0, time_left() - 10.0)
        x_sa = _simulated_annealing(best_x, time_budget=sa_time)
        x_sa = _project_to_bary_simplex(x_sa.reshape(N, 2)).flatten()
        
        bary_sa = x_sa.reshape(N, 2)
        s_sa = _min_area_ratio_from_bary(bary_sa) / (2 * TRI_AREA)
        
        if s_sa > best_score:
            best_score = s_sa
            best_x = x_sa.copy()
    
    # Phase 3: Final high-precision polish
    if best_x is not None and time_left() > 5.0:
        x_polished = _final_polish(best_x)
        x_polished = _project_to_bary_simplex(x_polished.reshape(N, 2)).flatten()
        
        bary_polished = x_polished.reshape(N, 2)
        s_polished = _min_area_ratio_from_bary(bary_polished) / (2 * TRI_AREA)
        
        if s_polished > best_score:
            best_score = s_polished
            best_x = x_polished.copy()
    
    # Convert to Cartesian
    if best_x is not None:
        bary_best = best_x.reshape(N, 2)
        pts_best = _bary_to_cart(bary_best[:, 0], bary_best[:, 1])
        return pts_best
    
    # Fallback: return random valid points
    pts = []
    for _ in range(N):
        u, v = np.random.random(2)
        while u + v > 1:
            u, v = np.random.random(2)
        pts.append([u + 0.5*v, np.sqrt(3)/2*v])
    return np.array(pts)

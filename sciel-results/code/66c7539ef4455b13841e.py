"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the unit-base equilateral triangle

# Precompute indices for all triples to speed up area calculation
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _all_dets_from_bary(bary):
    """Compute all 165 determinants (2*area) from barycentric coords.
    det(i,j,k) = (u_j - u_i)*(v_k - v_i) - (v_j - v_i)*(u_k - u_i)
    This is the cross product in barycentric space, proportional to actual area."""
    idx = _get_indices()
    ui = bary[idx[:, 0], 0]
    uj = bary[idx[:, 1], 0]
    uk = bary[idx[:, 2], 0]
    vi = bary[idx[:, 0], 1]
    vj = bary[idx[:, 1], 1]
    vk = bary[idx[:, 2], 1]
    dets = (uj - ui) * (vk - vi) - (vj - vi) * (uk - ui)
    return dets

def _min_area_ratio(bary):
    """Return min |det| / (2 * TRI_AREA) which equals min triangle area / TRI_AREA."""
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets)) / (2.0 * TRI_AREA)

def _score_bary(bary):
    """Score: min triangle area (actual, not ratio)."""
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets)) / 2.0

def _fix_signs(bary):
    """Determine the sign of each determinant based on current point ordering.
    Returns array of +1 or -1 for each triple."""
    dets = _all_dets_from_bary(bary)
    # If det is zero, default to +1
    signs = np.where(dets >= 0, 1.0, -1.0)
    return signs

def _softmin_objective(x, beta=50.0):
    """Soft-min objective: -logsumexp(beta * dets) / beta approximates -min(dets).
    x = [bary(22), t] where t is the lower bound variable.
    We minimize -t + penalty for violating t <= signed_dets."""
    bary = x[:2*N].reshape(N, 2)
    bary = _clamp_bary(bary)
    t = x[2*N]
    
    dets = _all_dets_from_bary(bary)
    signs = _fix_signs(bary)
    signed_dets = signs * dets
    
    # Soft constraint: all signed_dets should be >= t
    # Penalty: sum of softplus(t - signed_dets) * beta
    violations = t - signed_dets
    # Use log(1+exp(beta*v)) as smooth max(0, v)
    # For numerical stability:
    max_v = np.max(violations)
    if max_v > 100 / beta:
        penalty = np.sum(violations) * beta
    else:
        log_terms = np.log1p(np.exp(beta * violations - beta * max_v)) + beta * max_v
        penalty = np.sum(log_terms) / beta
    
    # Objective: maximize t, so minimize -t + penalty
    obj = -t + penalty
    return obj

def _softmin_grad(x, beta=50.0):
    """Gradient of softmin objective."""
    bary = x[:2*N].reshape(N, 2)
    bary = _clamp_bary(bary)
    t = x[2*N]
    
    dets = _all_dets_from_bary(bary)
    signs = _fix_signs(bary)
    signed_dets = signs * dets
    
    violations = t - signed_dets
    
    # Gradient of penalty term
    # d/dsigned_det_j of log(1+exp(beta*(t - signed_det_j))) = -beta*exp(beta*v_j)/(1+exp(beta*v_j)) = -beta*sigmoid(-beta*v_j)
    exp_vals = np.exp(np.clip(beta * violations, -500, 500))
    grad_penalty = -beta * exp_vals / (1 + exp_vals)
    
    # Gradient w.r.t. t: 1 + sum of grad_penalty contributions
    grad_t = 1.0 + np.sum(-grad_penalty)  # d/dt of -t is -1, d/dt of penalty is sum(-grad_penalty_j * (-1)) ... 
    # Actually: d/dt [ -t + sum log(1+exp(beta*(t-s_j))) ]
    # = -1 + sum beta*exp(beta*(t-s_j))/(1+exp(beta*(t-s_j)))
    # = -1 + sum beta*sigmoid(beta*(t-s_j))
    sig_vals = 1.0 / (1.0 + np.exp(-np.clip(beta * violations, -500, 500)))
    grad_t = -1.0 + np.sum(beta * sig_vals)
    
    # Gradient w.r.t. bary coordinates
    # This is complex; use numerical differentiation for bary part
    grad_bary = np.zeros(2 * N)
    for i in range(2 * N):
        x_plus = x.copy()
        x_plus[i] += 1e-7
        f_plus = _softmin_objective(x_plus, beta)
        x_minus = x.copy()
        x_minus[i] -= 1e-7
        f_minus = _softmin_objective(x_minus, beta)
        grad_bary[i] = (f_plus - f_minus) / (2e-7)
    
    grad = np.concatenate([grad_bary, [grad_t]])
    return grad

def _hard_objective(x, signs):
    """Hard constraint objective: minimize -t subject to signed_dets >= t for all triples.
    x = [bary(22), t]"""
    bary = x[:2*N].reshape(N, 2)
    bary = _clamp_bary(bary)
    t = x[2*N]
    
    dets = _all_dets_from_bary(bary)
    signed_dets = signs * dets
    
    # Penalty for violations
    violations = np.maximum(0, t - signed_dets)
    penalty = 100.0 * np.sum(violations**2)
    
    return -t + penalty

def _structured_init_bary(seed=42):
    """Generate diverse structured initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    patterns = []
    
    # Pattern A: Symmetric grid
    pA = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
    ])
    patterns.append(pA)
    
    # Pattern B: Hexagonal-ish
    pB = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [0.33, 0.33],
        [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
    ])
    patterns.append(pB)
    
    # Pattern C: More spread out
    pC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
        [0.33, 0.33],
        [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
    ])
    patterns.append(pC)
    
    # Pattern D: Random-ish structured
    pD = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [rng.uniform(0.3, 0.7), 0.0],
        [rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)],
        [0.0, rng.uniform(0.3, 0.7)],
        [1/3, 1/3],
        [rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)],
        [rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.3)],
        [rng.uniform(0.1, 0.3), rng.uniform(0.7, 0.9)],
        [rng.uniform(0.3, 0.7), rng.uniform(0.2, 0.5)]
    ])
    patterns.append(pD)
    
    # Pattern E: Based on known good configurations
    pE = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [1/3, 0.0], [2/3, 0.0],
        [0.0, 1/3], [0.0, 2/3],
        [1/3, 1/3], [2/3, 1/3],
        [1/6, 1/2], [1/2, 1/2],
        [5/6, 1/2]
    ])
    patterns.append(pE)
    
    # Pattern F: Dense interior with symmetric placement
    pF = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [1/3, 1/3], [2/3, 1/3], [1/3, 2/3],
        [0.5, 0.25], [0.25, 0.5], [0.75, 0.25],
        [0.5, 0.5]
    ])
    patterns.append(pF)
    
    # Pattern G: Random with constraint satisfaction
    pG = np.zeros((N, 2))
    pG[0] = [0.0, 0.0]
    pG[1] = [1.0, 0.0]
    pG[2] = [0.0, 1.0]
    for i in range(3, N):
        u = rng.uniform(0.05, 0.95)
        v = rng.uniform(0.05, 0.95 - u)
        pG[i] = [u, v]
    patterns.append(pG)
    
    idx = seed % len(patterns)
    bary = patterns[idx].copy()
    
    jitter = rng.normal(0, 0.02, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    
    return bary

def _fps_init(seed=42):
    """Farthest Point Sampling initialization for diverse spread."""
    rng = np.random.RandomState(seed)
    
    # Start with vertices
    bary = np.zeros((N, 2))
    bary[0] = [0.0, 0.0]
    bary[1] = [1.0, 0.0]
    bary[2] = [0.0, 1.0]
    
    # FPS: pick point farthest from existing set
    for i in range(3, N):
        # Generate many candidates
        candidates = rng.uniform(0, 1, size=(500, 2))
        candidates = _clamp_bary(candidates)
        
        # Compute min distance to existing points (in barycentric space)
        best_candidate = None
        best_min_dist = -1
        for c in candidates:
            dists = np.sqrt(np.sum((bary[:i] - c)**2, axis=1))
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = c
        
        bary[i] = best_candidate
    
    return bary

def _random_init_bary(seed=42):
    """Random initialization with good spread."""
    rng = np.random.RandomState(seed)
    bary = np.zeros((N, 2))
    bary[0] = [0.0, 0.0]
    bary[1] = [1.0, 0.0]
    bary[2] = [0.0, 1.0]
    for i in range(3, N):
        u = rng.uniform(0.02, 0.98)
        v = rng.uniform(0.02, 0.98 - u)
        bary[i] = [u, v]
    return bary

def _l_bfgs_softmin(bary0, t0, maxiter=300, beta=100.0):
    """L-BFGS-B optimization with soft-min epigraph formulation."""
    try:
        x0 = np.concatenate([bary0.flatten(), [t0]])
        bounds = [(0, 1)] * (2 * N) + [(0, TRI_AREA)]
        
        res = minimize(
            _softmin_objective, x0, args=(beta,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        
        bary_opt = _clamp_bary(res.x[:2*N].reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _l_bfgs_hard(bary0, t0, maxiter=200):
    """L-BFGS-B with hard sign-fixed constraints."""
    try:
        signs = _fix_signs(bary0)
        x0 = np.concatenate([bary0.flatten(), [t0]])
        bounds = [(0, 1)] * (2 * N) + [(0, TRI_AREA)]
        
        res = minimize(
            _hard_objective, x0, args=(signs,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-9}
        )
        
        bary_opt = _clamp_bary(res.x[:2*N].reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _slsqp_polish(bary0, maxiter=200):
    """SLSQP polish with explicit constraints."""
    try:
        signs = _fix_signs(bary0)
        x0 = np.concatenate([bary0.flatten(), [_score_bary(bary0)]])
        bounds = [(0, 1)] * (2 * N) + [(0, TRI_AREA)]
        
        # Inequality constraints: signed_det >= t
        cons = []
        idx = _get_indices()
        for m in range(len(idx)):
            i, j, k = idx[m]
            s = signs[m]
            def constraint(x, i=i, j=j, k=k, s=s, m=m):
                b = x[:2*N].reshape(N, 2)
                b = _clamp_bary(b)
                t = x[2*N]
                ui, uj, uk = b[i,0], b[j,0], b[k,0]
                vi, vj, vk = b[i,1], b[j,1], b[k,1]
                det = (uj-ui)*(vk-vi) - (vj-vi)*(uk-ui)
                return s*det - t
            cons.append({'type': 'ineq', 'fun': constraint})
        
        res = minimize(
            lambda x: -x[2*N], x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        
        if res.success or res.fun < 0:
            bary_opt = _clamp_bary(res.x[:2*N].reshape(N, 2))
            return bary_opt
        return bary0.copy()
    except Exception:
        return bary0.copy()

def _anneal_softmin(bary0, t0, betas=[50, 100, 200, 400], maxiter_per_stage=150):
    """Multi-stage beta annealing with soft-min objective."""
    bary = bary0.copy()
    t = t0
    
    for beta in betas:
        x0 = np.concatenate([bary.flatten(), [t]])
        bounds = [(0, 1)] * (2 * N) + [(0, TRI_AREA)]
        
        try:
            res = minimize(
                _softmin_objective, x0, args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter_per_stage, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            bary = _clamp_bary(res.x[:2*N].reshape(N, 2))
            t = res.x[2*N]
        except Exception:
            pass
    
    return bary

def _hill_climb_optimize(bary0, max_iter=300, initial_step=0.03):
    """Hill climbing with adaptive step size."""
    bary = bary0.copy()
    best_val = _score_bary(bary)
    step = initial_step
    
    for it in range(max_iter):
        improved = False
        
        for _ in range(3):
            i = np.random.randint(N)
            d = np.random.randint(2)
            sign = 1 if np.random.rand() > 0.5 else -1
            
            trial = bary.copy()
            trial[i, d] += sign * step
            trial = _clamp_bary(trial)
            
            val = _score_bary(trial)
            if val > best_val:
                bary = trial
                best_val = val
                improved = True
        
        if not improved:
            step *= 0.8
            if step < 1e-7:
                break
    
    return bary, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    def time_left():
        return deadline - time.time()
    
    def score_bary(bary):
        return _score_bary(bary)
    
    # Phase 1: Try multiple initializations with annealed soft-min optimization
    initial_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in initial_seeds:
        if time_left() < 10.0:
            break
        
        # Alternate between structured, FPS, and random inits
        if seed % 3 == 0:
            bary_init = _structured_init_bary(seed=seed)
        elif seed % 3 == 1:
            bary_init = _fps_init(seed=seed)
        else:
            bary_init = _random_init_bary(seed=seed)
        
        score_init = score_bary(bary_init)
        
        # Annealed soft-min optimization
        if time_left() > 15.0:
            bary_annealed = _anneal_softmin(bary_init, score_init, 
                                            betas=[50, 100, 200], 
                                            maxiter_per_stage=100)
            score_annealed = score_bary(bary_annealed)
            if score_annealed > score_init:
                score_init = score_annealed
                bary_init = bary_annealed
        
        # Hill climb refinement
        if time_left() > 8.0:
            bary_opt, score_opt = _hill_climb_optimize(bary_init, max_iter=150, initial_step=0.03)
            if score_opt > score_init:
                score_init = score_opt
                bary_init = bary_opt
        
        # Hard sign-fixed L-BFGS polish
        if time_left() > 5.0:
            bary_hard = _l_bfgs_hard(bary_init, score_init, maxiter=150)
            score_hard = score_bary(bary_hard)
            if score_hard > score_init:
                score_init = score_hard
                bary_init = bary_hard
        
        if score_init > best_score:
            best_score = score_init
            best_bary = bary_init.copy()
    
    # Phase 2: SLSQP polish on best solution (if time permits)
    if best_bary is not None and time_left() > 20.0:
        bary_slsqp = _slsqp_polish(best_bary, maxiter=150)
        score_slsqp = score_bary(bary_slsqp)
        if score_slsqp > best_score:
            best_score = score_slsqp
            best_bary = bary_slsqp.copy()
    
    # Phase 3: Multi-start perturbations
    rng = np.random.RandomState(123)
    restart_count = 0
    max_restarts = 50
    
    while restart_count < max_restarts and time_left() > 8.0:
        restart_count += 1
        
        pert = rng.normal(0, 0.04, size=(N, 2))
        bary_start = _clamp_bary(best_bary + pert)
        
        # Quick anneal
        if time_left() > 12.0:
            bary_annealed = _anneal_softmin(bary_start, score_bary(bary_start),
                                            betas=[100, 200], maxiter_per_stage=80)
            score_annealed = score_bary(bary_annealed)
            if score_annealed > score_bary(bary_start):
                bary_start = bary_annealed
        
        # Hill climb
        bary_opt, score_opt = _hill_climb_optimize(bary_start, max_iter=100, initial_step=0.02)
        
        if score_opt > best_score:
            best_score = score_opt
            best_bary = bary_opt.copy()
    
    # Phase 4: Final fine-tuning
    if time_left() > 5.0:
        for ft in range(10):
            if time_left() < 2.0:
                break
            pert = rng.normal(0, 0.005, size=(N, 2))
            bary_ft = _clamp_bary(best_bary + pert)
            bary_opt, score_opt = _hill_climb_optimize(bary_ft, max_iter=150, initial_step=0.01)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
    
    # Final polish
    if time_left() > 3.0 and best_bary is not None:
        bary_final = _l_bfgs_hard(best_bary, best_score, maxiter=200)
        score_final = score_bary(bary_final)
        if score_final > best_score:
            best_score = score_final
            best_bary = bary_final.copy()
    
    # Final fallback
    if best_bary is None:
        best_bary = _structured_init_bary(seed=42)
    
    return _bary_to_cart(best_bary)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END

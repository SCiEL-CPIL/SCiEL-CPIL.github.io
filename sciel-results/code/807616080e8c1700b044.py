import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute all triple indices
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

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

def _project_to_bary_simplex(bary):
    """Project onto the barycentric simplex: u>=0, v>=0, u+v<=1."""
    bary = np.array(bary, dtype=float)
    bary = np.maximum(bary, 0.0)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _all_dets_from_bary(bary):
    """Compute all triple determinants (2*area) from barycentric coords."""
    idx = _IDX_TRIPLES
    ui = bary[idx[:, 0], 0]
    vi = bary[idx[:, 0], 1]
    uj = bary[idx[:, 1], 0]
    vj = bary[idx[:, 1], 1]
    uk = bary[idx[:, 2], 0]
    vk = bary[idx[:, 2], 1]
    
    du_ji = uj - ui
    dv_ji = vj - vi
    du_ki = uk - ui
    dv_ki = vk - vi
    
    dets = np.sqrt(3) / 2.0 * (du_ji * dv_ki - dv_ji * du_ki)
    return dets

def _min_area_ratio_from_bary(bary):
    """Compute the minimum area (divided by triangle area) from barycentric coords."""
    dets = _all_dets_from_bary(bary)
    min_det = np.min(np.abs(dets))
    return 0.5 * min_det / TRI_AREA

def _objective_smooth(x, beta=50.0):
    """Smooth objective using log-sum-exp approximation of -min area."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    abs_dets = np.abs(dets)
    
    max_det = np.max(abs_dets)
    if max_det < 1e-15:
        return 0.0
    
    neg_beta_dets = -beta * abs_dets
    max_nb = np.max(neg_beta_dets)
    lse = max_nb + np.log(np.sum(np.exp(neg_beta_dets - max_nb)))
    return lse / beta

def _objective_grad(x, beta=50.0):
    """Objective with gradient for L-BFGS-B."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    abs_dets = np.abs(dets)
    
    max_det = np.max(abs_dets)
    if max_det < 1e-15:
        return 0.0, np.zeros_like(x)
    
    neg_beta_dets = -beta * abs_dets
    max_nb = np.max(neg_beta_dets)
    exp_vals = np.exp(neg_beta_dets - max_nb)
    sum_exp = np.sum(exp_vals)
    lse = max_nb + np.log(sum_exp)
    val = lse / beta
    
    idx = _IDX_TRIPLES
    signs = np.sign(dets)
    
    ui = bary[idx[:, 0], 0]
    vi = bary[idx[:, 0], 1]
    uj = bary[idx[:, 1], 0]
    vj = bary[idx[:, 1], 1]
    uk = bary[idx[:, 2], 0]
    vk = bary[idx[:, 2], 1]
    
    sqrt3_2 = np.sqrt(3) / 2.0
    
    grad_det = np.zeros((N_TRIPLES, N, 2))
    grad_det[:, idx[:, 0], 0] = sqrt3_2 * (vj - vk)
    grad_det[:, idx[:, 0], 1] = sqrt3_2 * (uj - uk)
    grad_det[:, idx[:, 1], 0] = sqrt3_2 * (vk - vi)
    grad_det[:, idx[:, 1], 1] = -sqrt3_2 * (uk - ui)
    grad_det[:, idx[:, 2], 0] = -sqrt3_2 * (vj - vi)
    grad_det[:, idx[:, 2], 1] = sqrt3_2 * (uj - ui)
    
    sign_det = signs[:, None, None]
    grad_abs_det = sign_det * grad_det
    
    weights = exp_vals / sum_exp
    grad_bary = -np.sum(weights[:, None, None] * grad_abs_det, axis=0)
    
    grad_x = np.zeros_like(x)
    grad_x[:N*2] = grad_bary.flatten()
    
    return val, grad_x

def _objective_slsqp(x):
    """Objective for SLSQP (uses exact min area)."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    min_abs_det = np.min(np.abs(dets))
    return -min_abs_det

def _structured_init_fps():
    """Farthest-point-sampling structured initialization in barycentric coords."""
    points = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    points.append(np.array([1.0/3, 1.0/3]))
    
    rng = np.random.RandomState(789)
    
    while len(points) < N:
        candidates = []
        for _ in range(2000):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v <= 1.0:
                candidates.append(np.array([u, v]))
        candidates = np.array(candidates)
        
        best_candidate = None
        best_min_dist = -1
        
        for cand in candidates:
            dists = [np.linalg.norm(cand - p) for p in points]
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_candidate = cand
        
        if best_candidate is not None:
            points.append(best_candidate)
    
    return np.array(points)

def _local_optimize_nm(x0, maxiter=500, xatol=1e-9, fatol=1e-9):
    """Nelder-Mead local optimization."""
    try:
        res = minimize(
            _objective_smooth, x0, method='Nelder-Mead',
            args=(30.0,),
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, maxiter=300, gtol=1e-8):
    """L-BFGS-B with gradient."""
    try:
        res = minimize(
            _objective_grad, x0, method='L-BFGS-B',
            jac=True,
            args=(30.0,),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, maxiter=200):
    """SLSQP with simplex constraints."""
    n = len(x0) // 2
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    bounds = [(0.0, 1.0)] * (2 * n)
    
    try:
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _simulated_annealing(bary_init, max_iter=5000, t0=0.05, tf=0.001, cooling=0.9997):
    """Simulated annealing with jitter moves."""
    rng = np.random.RandomState(12345)
    current = _project_to_bary_simplex(bary_init.copy())
    current_score = _min_area_ratio_from_bary(current)
    best = current.copy()
    best_score = current_score
    temp = t0
    
    for it in range(max_iter):
        # Jitter move: perturb 1-3 random points
        n_pert = rng.randint(1, 4)
        idxs = rng.choice(N, size=n_pert, replace=False)
        scale = temp * 2.0
        new_bary = current.copy()
        for idx in idxs:
            new_bary[idx] += rng.normal(0, scale, size=2)
        new_bary = _project_to_bary_simplex(new_bary)
        
        new_score = _min_area_ratio_from_bary(new_bary)
        
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / (temp * 0.1)):
            current = new_bary
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        
        temp *= cooling
    
    return best, best_score

def _ga_population_init(pop_size=20):
    """Initialize GA population with diverse starts."""
    rng = np.random.RandomState(42)
    pop = []
    
    # Include structured starts
    pop.append(_structured_init_fps())
    pop.append(_project_to_bary_simplex(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3], [0.25, 0.0], [0.75, 0.0], [0.25, 0.25], [0.5, 0.75],
    ])))
    
    # Random starts
    for _ in range(pop_size - 2):
        bary = []
        for _ in range(N):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v > 1.0:
                u, v = u * 0.8, v * 0.8
            bary.append([u, v])
        pop.append(_project_to_bary_simplex(np.array(bary)))
    
    return pop

def _ga_evolve(pop, max_gen=30, elite_size=5, mutation_rate=0.3, crossover_rate=0.7):
    """Genetic algorithm evolution."""
    rng = np.random.RandomState(999)
    
    def fitness(bary):
        return _min_area_ratio_from_bary(bary)
    
    scores = [fitness(b) for b in pop]
    best_idx = np.argmax(scores)
    best_bary = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    for gen in range(max_gen):
        # Sort by fitness
        order = np.argsort(scores)[::-1]
        pop = [pop[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Elitism
        new_pop = pop[:elite_size]
        
        # Crossover and mutation
        while len(new_pop) < len(pop):
            if rng.random() < crossover_rate and len(pop) >= 2:
                # Tournament selection
                i1 = rng.choice(len(pop), size=2, replace=False)
                i1 = i1[np.argmax([scores[i1[0]], scores[i1[1]]])]
                i2 = rng.choice(len(pop), size=2, replace=False)
                i2 = i2[np.argmax([scores[i2[0]], scores[i2[1]]])]
                
                # Blend crossover
                alpha = rng.uniform(0.3, 0.7)
                child = alpha * pop[i1] + (1 - alpha) * pop[i2]
            else:
                i1 = rng.choice(len(pop), size=2, replace=False)
                i1 = i1[np.argmax([scores[i1[0]], scores[i1[1]]])]
                child = pop[i1].copy()
            
            # Mutation
            if rng.random() < mutation_rate:
                n_mut = rng.randint(1, 4)
                idxs = rng.choice(N, size=n_mut, replace=False)
                for idx in idxs:
                    child[idx] += rng.normal(0, 0.05, size=2)
            
            child = _project_to_bary_simplex(child)
            new_pop.append(child)
        
        pop = new_pop
        scores = [fitness(b) for b in pop]
        
        if scores[0] > best_score:
            best_score = scores[0]
            best_bary = pop[0].copy()
    
    return best_bary, best_score

def _final_polish(bary, max_iter=50):
    """Final high-precision polish using alternating SLSQP and NM."""
    rng = np.random.RandomState(777)
    current = bary.copy()
    current_score = _min_area_ratio_from_bary(current)
    
    for it in range(max_iter):
        # Small random perturbation
        bary_pert = current + rng.normal(0, 0.002, size=current.shape)
        bary_pert = _project_to_bary_simplex(bary_pert)
        
        # SLSQP refinement
        x_sl = _local_optimize_slsqp(bary_pert.flatten(), maxiter=80)
        bary_sl = _project_to_bary_simplex(x_sl.reshape(-1, 2))
        score_sl = _min_area_ratio_from_bary(bary_sl)
        
        if score_sl > current_score:
            current = bary_sl.copy()
            current_score = score_sl
        
        # NM refinement
        x_nm = _local_optimize_nm(bary_sl.flatten(), maxiter=150, xatol=1e-10, fatol=1e-10)
        bary_nm = _project_to_bary_simplex(x_nm.reshape(-1, 2))
        score_nm = _min_area_ratio_from_bary(bary_nm)
        
        if score_nm > current_score:
            current = bary_nm.copy()
            current_score = score_nm
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
    
    best_points = None
    best_score = -np.inf
    best_bary = None
    
    # Generate multiple structured starts
    starts = []
    starts.append(_structured_init_fps())
    starts.append(_project_to_bary_simplex(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3], [0.25, 0.0], [0.75, 0.0], [0.25, 0.25], [0.5, 0.75],
    ])))
    
    rng = np.random.RandomState(42)
    bary_rand = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1/3, 1/3]]
    for _ in range(7):
        u = rng.uniform(0.1, 0.8)
        v = rng.uniform(0.1, 0.8)
        if u + v < 0.85:
            bary_rand.append([u, v])
        else:
            bary_rand.append([u * 0.7, v * 0.7])
    starts.append(_project_to_bary_simplex(np.array(bary_rand[:N])))
    
    # Evaluate initial starts
    for bary_start in starts:
        if time_left() < 5:
            break
        bary_clamped = _project_to_bary_simplex(bary_start)
        points = _bary_to_cart(bary_clamped)
        score = _min_area_ratio_from_bary(bary_clamped)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_bary = bary_clamped.copy()
    
    # Phase 1: Multi-start local optimization
    max_restarts = 40
    restart = 0
    
    while restart < max_restarts and time_left() > 30.0:
        restart += 1
        
        if restart % 4 == 0:
            bary_start = []
            for _ in range(N):
                u = rng.uniform(0.05, 0.9)
                v = rng.uniform(0.05, 0.9)
                if u + v < 0.9:
                    bary_start.append([u, v])
                else:
                    bary_start.append([u * 0.8, v * 0.8])
            bary_start = _project_to_bary_simplex(np.array(bary_start[:N]))
        elif restart % 2 == 0:
            scale = 0.04 + 0.02 * (1.0 - restart / max_restarts)
            bary_pert = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_start = _project_to_bary_simplex(bary_pert)
        else:
            scale = 0.08 + 0.04 * (1.0 - restart / max_restarts)
            bary_pert = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_start = _project_to_bary_simplex(bary_pert)
        
        bary_opt = bary_start.copy()
        
        if time_left() > 25:
            x_sl = _local_optimize_slsqp(bary_opt.flatten(), maxiter=120)
            bary_sl = _project_to_bary_simplex(x_sl.reshape(-1, 2))
            score_sl = _min_area_ratio_from_bary(bary_sl)
            if score_sl > best_score:
                best_score = score_sl
                best_points = _bary_to_cart(bary_sl).copy()
                best_bary = bary_sl.copy()
                bary_opt = bary_sl.copy()
        
        if time_left() > 20:
            x_lb = _local_optimize_lbfgs(bary_opt.flatten(), maxiter=150)
            bary_lb = _project_to_bary_simplex(x_lb.reshape(-1, 2))
            score_lb = _min_area_ratio_from_bary(bary_lb)
            if score_lb > best_score:
                best_score = score_lb
                best_points = _bary_to_cart(bary_lb).copy()
                best_bary = bary_lb.copy()
                bary_opt = bary_lb.copy()
        
        if time_left() > 15:
            x_nm = _local_optimize_nm(bary_opt.flatten(), maxiter=200)
            bary_nm = _project_to_bary_simplex(x_nm.reshape(-1, 2))
            score_nm = _min_area_ratio_from_bary(bary_nm)
            if score_nm > best_score:
                best_score = score_nm
                best_points = _bary_to_cart(bary_nm).copy()
                best_bary = bary_nm.copy()
    
    # Phase 2: Simulated annealing for global exploration
    if time_left() > 25.0:
        sa_bary, sa_score = _simulated_annealing(best_bary, max_iter=3000, t0=0.04, tf=0.001, cooling=0.9998)
        if sa_score > best_score:
            best_score = sa_score
            best_points = _bary_to_cart(sa_bary).copy()
            best_bary = sa_bary.copy()
    
    # Phase 3: Genetic algorithm for population-based search
    if time_left() > 20.0:
        pop = _ga_population_init(pop_size=15)
        ga_bary, ga_score = _ga_evolve(pop, max_gen=20, elite_size=4, mutation_rate=0.4, crossover_rate=0.6)
        if ga_score > best_score:
            best_score = ga_score
            best_points = _bary_to_cart(ga_bary).copy()
            best_bary = ga_bary.copy()
    
    # Phase 4: Final high-precision polish
    if time_left() > 10.0 and best_bary is not None:
        polished_bary, polished_score = _final_polish(best_bary, max_iter=30)
        if polished_score > best_score:
            best_score = polished_score
            best_points = _bary_to_cart(polished_bary).copy()
            best_bary = polished_bary.copy()
    
    # Fallback
    if best_points is None:
        bary_fb = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3], [0.25, 0.0], [0.75, 0.0],
            [0.25, 0.25], [0.5, 0.75],
        ])
        best_points = _bary_to_cart(_project_to_bary_simplex(bary_fb))
    
    return best_points

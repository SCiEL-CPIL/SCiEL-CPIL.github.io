# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

# Precompute indices for all C(11,3) = 165 triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

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
    bary: (N, 2) array of (u, v). w is derived.
    Returns the minimum area ratio of any triangle formed by 3 points.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
    
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _all_areas_from_bary(bary):
    """
    Compute all 165 triangle areas.
    """
    dets = _all_dets_from_bary(bary)
    return np.abs(dets)

def _objective_epigraph(x, penalty_lambda=1000.0):
    """
    Epigraph formulation: x = [u_1, v_1, ..., u_N, v_N, t]
    Minimize -t subject to all triangle areas >= t.
    Smoothed via log-sum-exp for the constraint violations.
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    
    areas = _all_areas_from_bary(bary)
    
    # For each triangle, we want area >= t
    # Violation: max(0, t - area)
    # Smooth approximation: softplus(t - area) = log(1 + exp(t - area))
    # To avoid overflow, use stable softplus
    diff = t - areas
    # softplus(x) = log(1 + exp(x)) = max(x,0) + log(1 + exp(-|x|))
    softplus_vals = np.maximum(diff, 0) + np.log1p(np.exp(-np.abs(diff)))
    
    # Sum of violations
    total_violation = np.sum(softplus_vals)
    
    return -t + penalty + total_violation

def _objective_smooth(x, beta, penalty_lambda=50.0):
    """
    Smooth objective for L-BFGS-B.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords (u,v,w < 0)
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    
    areas = _all_areas_from_bary(bary)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

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
            min_d = np.min(np.linalg.norm(c - selected, axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

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

def _simulated_annealing(x0, deadline, max_steps=10000, t0=0.03, t_end=1e-8):
    """
    Simulated annealing with adaptive step sizes to explore the landscape.
    """
    rng = np.random.RandomState(789)
    current = x0.copy()
    current_score = _min_area_ratio_from_bary(current)
    best = current.copy()
    best_score = current_score
    
    # Cooling schedule
    alpha = (t_end / t0) ** (1.0 / max_steps)
    temperature = t0
    
    # Adaptive step size
    step_scale = 0.05
    recent_improvements = 0
    recent_steps = 0
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        neighbor = current.copy()
        move_type = rng.random()
        
        if move_type < 0.15:
            # Topological swap: swap barycentric coords of two random points
            i, j = rng.choice(N, size=2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j].copy(), neighbor[i].copy()
        elif move_type < 0.55:
            # Local jitter: Gaussian perturbation of one point
            i = rng.randint(N)
            noise = rng.normal(0, step_scale, size=2)
            neighbor[i] += noise
            neighbor[i] = np.clip(neighbor[i], 0, None)
            s = neighbor[i, 0] + neighbor[i, 1]
            if s > 1.0:
                neighbor[i] /= s
        elif move_type < 0.85:
            # Global jitter: perturb multiple points slightly
            n_pts = rng.randint(2, 5)
            idxs = rng.choice(N, size=n_pts, replace=False)
            for i in idxs:
                noise = rng.normal(0, step_scale * 0.7, size=2)
                neighbor[i] += noise
                neighbor[i] = np.clip(neighbor[i], 0, None)
                s = neighbor[i, 0] + neighbor[i, 1]
                if s > 1.0:
                    neighbor[i] /= s
        else:
            # Larger exploration move
            i = rng.randint(N)
            noise = rng.normal(0, step_scale * 2.0, size=2)
            neighbor[i] += noise
            neighbor[i] = np.clip(neighbor[i], 0, None)
            s = neighbor[i, 0] + neighbor[i, 1]
            if s > 1.0:
                neighbor[i] /= s
        
        neighbor = _project_to_bary_simplex(neighbor)
        neighbor_score = _min_area_ratio_from_bary(neighbor)
        
        if neighbor_score < 0:
            continue
            
        # Acceptance criterion
        accepted = False
        if neighbor_score >= current_score:
            current = neighbor
            current_score = neighbor_score
            accepted = True
        else:
            # Metropolis criterion
            delta = current_score - neighbor_score
            if temperature > 1e-12:
                prob = np.exp(-delta / temperature)
                if rng.random() < prob:
                    current = neighbor
                    current_score = neighbor_score
                    accepted = True
        
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
            recent_improvements += 1
        
        recent_steps += 1
        if recent_steps % 100 == 0:
            # Adapt step size based on recent performance
            if recent_improvements > 10:
                step_scale *= 0.8  # Shrinking: too many improvements, be more precise
            elif recent_improvements < 2:
                step_scale *= 1.3  # Growing: stuck, explore more
            step_scale = np.clip(step_scale, 0.005, 0.15)
            recent_improvements = 0
            recent_steps = 0
        
        temperature *= alpha
        
    return best, best_score

def _local_optimize_epigraph(x_bary, deadline, max_iter=2000):
    """
    Local optimization using epigraph formulation.
    x_bary: (N, 2) array of barycentric coords.
    Returns optimized (N, 2) barycentric coords.
    """
    # Initial t estimate
    areas = _all_areas_from_bary(x_bary)
    t0 = np.min(areas) * 0.9
    
    x_init = np.concatenate([x_bary.flatten(), [t0]])
    
    # Bounds: u,v in [0,1], t in [0, 0.5]
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)]
    
    res = minimize(
        _objective_epigraph,
        x_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    x_opt = res.x[:2*N].reshape(N, 2)
    x_opt = _project_to_bary_simplex(x_opt)
    
    return x_opt

def _multi_start_optimize(init_bary, deadline):
    """
    Multi-start local optimization from a given initial configuration.
    Uses both smooth min and epigraph formulations.
    """
    best_x = init_bary.copy()
    best_score = _min_area_ratio_from_bary(init_bary)
    
    if best_score < 0:
        return best_x, best_score
    
    bounds = [(0, 1)] * (2 * N)
    current_x = init_bary.flatten()
    
    # Phase 1: Smooth min with increasing beta
    betas = [5.0, 20.0, 50.0, 200.0, 1000.0]
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                _objective_smooth,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            x_opt = res.x.reshape(N, 2)
            x_opt = _project_to_bary_simplex(x_opt)
            
            score = _min_area_ratio_from_bary(x_opt)
            if score > best_score:
                best_score = score
                best_x = x_opt
                current_x = x_opt.flatten()
        except Exception:
            pass
    
    # Phase 2: Epigraph formulation
    if time.time() < deadline - 10:
        try:
            x_epigraph = _local_optimize_epigraph(best_x, deadline - 5)
            score = _min_area_ratio_from_bary(x_epigraph)
            if score > best_score:
                best_score = score
                best_x = x_epigraph
        except Exception:
            pass
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. More diverse random inits
    rng = np.random.RandomState(42)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333]:
        rng_s = np.random.RandomState(seed)
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng_s.random()
            v = rng_s.random()
            while u + v > 1.0:
                u = rng_s.random()
                v = rng_s.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Structured inits based on symmetry
    # Regular 11-point pattern
    magic_pts_1 = [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [1/3, 1/3],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts_1).flatten())
    
    # Another structured pattern
    magic_pts_2 = [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.33, 0.0], [0.66, 0.0], [0.0, 0.33], [0.0, 0.66],
        [0.33, 0.33], [0.66, 0.33], [0.33, 0.66], [0.5, 0.5],
    ]
    inits.append(np.array(magic_pts_2).flatten())
    
    # Grid-based init
    grid_pts = []
    for i in range(4):
        for j in range(4 - i):
            grid_pts.append([i/3.0, j/3.0])
    if len(grid_pts) >= N:
        inits.append(np.array(grid_pts[:N]).flatten())
    
    # Process each init
    for idx, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        try:
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            x_opt, score = _multi_start_optimize(init_bary, deadline - 10)
            
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
                
        except Exception:
            pass
    
    if best_x is None:
        best_x = _structured_init_fps()
        best_score = _min_area_ratio_from_bary(best_x)
    
    # Simulated Annealing Phase
    if time.time() < deadline - 25:
        try:
            sa_x, sa_score = _simulated_annealing(best_x, deadline - 15)
            if sa_score > best_score:
                best_score = sa_score
                best_x = sa_x
        except Exception:
            pass
    
    # Final high-precision polish pass
    if time.time() < deadline - 10:
        try:
            # Polish with epigraph
            polish_x = _local_optimize_epigraph(best_x, deadline - 5)
            score = _min_area_ratio_from_bary(polish_x)
            if score > best_score:
                best_score = score
                best_x = polish_x
            
            # Additional smooth min polish
            bounds = [(0, 1)] * (2 * N)
            current_x = best_x.flatten()
            for beta in [5000.0, 20000.0]:
                if time.time() > deadline - 3:
                    break
                res = minimize(
                    _objective_smooth,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                polish_x = res.x.reshape(N, 2)
                polish_x = _project_to_bary_simplex(polish_x)
                score = _min_area_ratio_from_bary(polish_x)
                if score > best_score:
                    best_score = score
                    best_x = polish_x
                    current_x = polish_x.flatten()
        except Exception:
            pass
    
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END

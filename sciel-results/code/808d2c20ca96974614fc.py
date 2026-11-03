"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices in Cartesian (for final output only)
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (C(11,3) = 165)
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    # P = u*(0,0) + v*(1,0) + w*(0.5, sqrt(3)/2)
    x = v * 1.0 + w * 0.5
    y = w * (np.sqrt(3) / 2)
    return np.column_stack((x, y))

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid simplex."""
    u = np.clip(bary[:, 0], 0.0, 1.0)
    v = np.clip(bary[:, 1], 0.0, 1.0)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        # Project back to the line u+v=1 while keeping ratio
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _get_dets(bary):
    """
    Compute determinants of all triples of barycentric coordinates.
    The area of the triangle formed by points i,j,k is A_ref * |det(P_i, P_j, P_k)|.
    Since A_ref is constant, maximizing min area is equivalent to maximizing min |det|.
    """
    # P is (N, 3) with columns u, v, w
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
    
    # Determinant of 3x3 matrix with rows Pi, Pj, Pk
    # det = Pi . (Pj x Pk)
    # Using vectorized operations
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_det(bary):
    """Return the minimum absolute determinant."""
    dets = _get_dets(bary)
    return np.min(np.abs(dets))

def _objective_hard(x):
    """
    Objective for SLSQP/Nelder-Mead: Minimize -min_det.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    min_d = _min_det(bary)
    return -min_d

def _objective_smooth(x, beta=50.0):
    """
    Objective for L-BFGS-B: Minimize negative soft-min of determinants.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    
    dets = _get_dets(bary)
    areas = np.abs(dets)
    
    min_a = np.min(areas)
    # Shift to avoid overflow in exp
    shifted = areas - min_a
    # soft_min = min_a - (1/beta) * log(sum(exp(-beta * shifted)))
    # Note: as beta -> inf, soft_min -> min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_a - log_sum / beta
    
    return -soft_min

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 30
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    # Start with centroid for symmetry
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

def _generate_initial_bary(seed, mode=0):
    """Generate diverse initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    if mode == 0:
        # FPS
        return _structured_init_fps()
    elif mode == 1:
        # Symmetric pattern: vertices, midpoints, centroid + some interior
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3],
            [0.25, 0.25], [0.5, 0.25], [0.25, 0.5], [0.5, 0.5] # Overlap at 0.5,0.5, will jitter
        ])
        # Add jitter to break symmetry slightly
        jitter = rng.normal(0, 0.05, size=bary.shape)
        bary = bary + jitter
    elif mode == 2:
        # Hexagonal-ish
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.33, 0.0], [0.66, 0.0], [0.0, 0.33], [0.0, 0.66],
            [0.33, 0.33], [0.66, 0.33], [0.33, 0.66], [0.5, 0.5]
        ])
        jitter = rng.normal(0, 0.02, size=bary.shape)
        bary = bary + jitter
    elif mode == 3:
        # Random
        bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0.01, 0.99)
                v = rng.uniform(0.01, 0.99)
                if u + v < 0.98:
                    bary[i] = [u, v]
                    break
    else:
        # Pure random
        bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0.01, 0.99)
                v = rng.uniform(0.01, 0.99)
                if u + v < 0.98:
                    bary[i] = [u, v]
                    break
                    
    return _clamp_bary(bary)

def _optimize_slsqp(bary0, maxiter=200):
    """SLSQP optimization on hard objective."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        # SLSQP constraints: u+v <= 1
        constraints = []
        for i in range(N):
            def cons(x, i=i):
                return 1.0 - (x[2*i] + x[2*i+1])
            constraints.append({'type': 'ineq', 'fun': cons})
            
        res = minimize(
            _objective_hard, x0, method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _optimize_lbfgs_softmin(bary0, maxiter=300, beta=50.0):
    """L-BFGS-B optimization on soft-min objective."""
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        def obj(x):
            return _objective_smooth(x, beta=beta)
        
        res = minimize(
            obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _optimize_nelder_mead(bary0, maxiter=300):
    """Nelder-Mead optimization on hard objective."""
    try:
        x0 = bary0.flatten()
        res = minimize(
            _objective_hard, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-14, 'xatol': 1e-14, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
        
    best_bary = None
    best_score = -np.inf
    
    # Phase 1: Multi-start with diverse initializations
    # Modes: 0=FPS, 1=Symmetric, 2=Hex, 3=Random, 4=Random
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333]
    
    for idx, seed in enumerate(seeds):
        if time_left() < 5.0:
            break
            
        mode = idx % 5
        bary_init = _generate_initial_bary(seed=seed, mode=mode)
        score_init = _min_det(bary_init)
        
        # 1. Smooth L-BFGS-B to find a good basin
        if time_left() > 8.0:
            bary_opt = _optimize_lbfgs_softmin(bary_init, maxiter=200, beta=50.0)
            score_opt = _min_det(bary_opt)
            if score_opt > score_init:
                score_init = score_opt
                bary_init = bary_opt
        
        # 2. SLSQP on hard objective to refine the min
        if time_left() > 5.0:
            bary_opt = _optimize_slsqp(bary_init, maxiter=200)
            score_opt = _min_det(bary_opt)
            if score_opt > score_init:
                score_init = score_opt
                bary_init = bary_opt
        
        # 3. Nelder-Mead for final polish
        if time_left() > 3.0:
            bary_opt = _optimize_nelder_mead(bary_init, maxiter=150)
            score_opt = _min_det(bary_opt)
            if score_opt > score_init:
                score_init = score_opt
                bary_init = bary_opt
        
        if score_init > best_score:
            best_score = score_init
            best_bary = bary_init.copy()

    # Phase 2: Perturbation around best solution
    rng = np.random.RandomState(999)
    for _ in range(30):
        if time_left() < 5.0:
            break
            
        # Adaptive scale
        scale = 0.05 * (1.0 + np.random.random())
        pert = rng.normal(0, scale, size=(N, 2))
        bary_start = _clamp_bary(best_bary + pert)
        
        # Quick optimization
        if time_left() > 4.0:
            bary_opt = _optimize_lbfgs_softmin(bary_start, maxiter=100, beta=50.0)
            score_opt = _min_det(bary_opt)
            
            if score_opt > best_score and time_left() > 3.0:
                bary_opt = _optimize_slsqp(bary_opt, maxiter=100)
                score_opt = _min_det(bary_opt)
                
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()

    # Phase 3: Final high-precision refinement
    if time_left() > 10.0:
        # Beta annealing
        for beta in [20, 50, 100, 200]:
            if time_left() < 5.0: break
            bary_opt = _optimize_lbfgs_softmin(best_bary, maxiter=200, beta=beta)
            score_opt = _min_det(bary_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
        
        # Final SLSQP
        if time_left() > 5.0:
            bary_opt = _optimize_slsqp(best_bary, maxiter=300)
            score_opt = _min_det(bary_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
                
        # Final NM
        if time_left() > 3.0:
            bary_opt = _optimize_nelder_mead(best_bary, maxiter=200)
            score_opt = _min_det(bary_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()

    # Fallback
    if best_bary is None:
        best_bary = _generate_initial_bary(seed=42, mode=0)
    
    return _bary_to_cart(best_bary)

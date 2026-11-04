"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Combines 2D barycentric representation (inspiration 1) with soft-min annealing,
differential evolution, and structured initializations (current program)."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _bary_to_cart(bary2d):
    """Convert 2D barycentric (u, v) to Cartesian. w = 1-u-v. P = u*V0 + v*V1 + w*V2."""
    u = bary2d[:, 0]
    v = bary2d[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack([x, y])

def _all_dets_from_bary(bary2d):
    """Compute all determinants (twice the area in barycentric coords, scaled by TRI_AREA)."""
    u = bary2d[:, 0]
    v = bary2d[:, 1]
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

def _min_area_ratio(bary2d):
    """Returns the minimum |det| (proportional to area / TRI_AREA)."""
    dets = _all_dets_from_bary(bary2d)
    return np.min(np.abs(dets))

def _clamp_bary(bary2d):
    """Project to valid barycentric simplex: u>=0, v>=0, u+v<=1."""
    u = np.maximum(0.0, bary2d[:, 0])
    v = np.maximum(0.0, bary2d[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack([u, v])

def _score_bary(bary2d):
    """Score: minimum triangle area (in actual units)."""
    bary2d = _clamp_bary(bary2d)
    return _min_area_ratio(bary2d) * TRI_AREA

def _softmin_objective(x, beta=120.0):
    """Smooth surrogate: negative soft-min of |dets| using log-sum-exp, with floor penalty."""
    bary2d = x.reshape(N, 2)
    bary2d = _clamp_bary(bary2d)
    dets = _all_dets_from_bary(bary2d)
    areas = np.abs(dets)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    # Numerically stable log-sum-exp
    max_shift = np.max(shifted)
    log_sum = max_shift + np.log(np.sum(np.exp(-beta * (shifted - max_shift))))
    soft_min = min_a - log_sum / beta
    
    # Penalize any area that is extremely small (near-degenerate)
    floor_penalty = 1000.0 * np.sum(np.maximum(0.0, 1e-6 - areas) ** 2)
    
    return -soft_min + floor_penalty

def _hard_objective(x):
    """Hard objective: negative min |det|."""
    bary2d = x.reshape(N, 2)
    bary2d = _clamp_bary(bary2d)
    dets = _all_dets_from_bary(bary2d)
    return -np.min(np.abs(dets))

def _structured_init_bary(seed=42):
    """Generate structured initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    patterns = []
    
    # Pattern A: Symmetric with vertices, edge midpoints, centroid
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
    
    # Pattern C: More spread
    pC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
        [0.33, 0.33],
        [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
    ])
    patterns.append(pC)
    
    # Pattern D: Concentric hexagons
    pD = np.array([
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
    patterns.append(pD)
    
    # Pattern E: Magic init
    pE = np.array([
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ])
    patterns.append(pE)
    
    # Pattern F: Hexagonal lattice
    pF = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    patterns.append(pF)
    
    idx = seed % len(patterns)
    bary2d = patterns[idx].copy()
    jitter = rng.normal(0, 0.015, size=bary2d.shape)
    bary2d = bary2d + jitter
    bary2d = _clamp_bary(bary2d)
    return bary2d

def _fps_init():
    """Farthest Point Sampling on a grid of barycentric coordinates."""
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
        dists = np.min(np.linalg.norm(candidates[:, None, :] - np.array(selected)[None, :, :], axis=2), axis=1)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _random_init_bary(seed):
    """Random initialization in barycentric coords."""
    rng = np.random.RandomState(seed)
    bary2d = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0.01, 0.99)
            v = rng.uniform(0.01, 0.99)
            if u + v < 0.98:
                bary2d[i] = [u, v]
                break
    return bary2d

def _l_bfgs_softmin(bary2d_0, maxiter=400, beta=120.0):
    """L-BFGS-B optimization using soft-min surrogate for smooth gradients."""
    try:
        x0 = bary2d_0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        def obj(x):
            return _softmin_objective(x, beta=beta)
        
        res = minimize(
            obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 80, 'maxfun': 100000}
        )
        bary2d_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary2d_opt
    except Exception:
        return bary2d_0.copy()

def _l_bfgs_hard(bary2d_0, maxiter=200):
    """L-BFGS-B optimization using hard min."""
    try:
        x0 = bary2d_0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        res = minimize(
            _hard_objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 80}
        )
        bary2d_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary2d_opt
    except Exception:
        return bary2d_0.copy()

def _slsqp_refine(bary2d_0, maxiter=200):
    """SLSQP refinement with explicit constraint u+v<=1."""
    try:
        x0 = bary2d_0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        # Constraints: for each point, u_i + v_i <= 1
        cons = []
        for i in range(N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1],
                'jac': lambda x, i=i: np.concatenate([np.zeros(2*i), [-1.0, -1.0], np.zeros(2*(N-i-1))])
            })
        
        res = minimize(
            _hard_objective, x0, method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 100}
        )
        bary2d_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary2d_opt
    except Exception:
        return bary2d_0.copy()

def _nelder_mead_refine(bary2d_0, maxiter=500):
    """Nelder-Mead refinement."""
    try:
        x0 = bary2d_0.flatten()
        
        res = minimize(
            _hard_objective, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        bary2d_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary2d_opt
    except Exception:
        return bary2d_0.copy()

def _beta_anneal_refine(bary2d_0, betas=(20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0)):
    """Refine with increasing beta values and tighter convergence."""
    current = bary2d_0.copy()
    for beta in betas:
        current = _l_bfgs_softmin(current, beta=beta, maxiter=300)
    # Final hard polish with more iterations
    current = _l_bfgs_hard(current, maxiter=500)
    return current

def _differential_evolution_global(rng, maxiter=300, popsize=15):
    """Global search using differential evolution."""
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def de_objective(x):
        return _hard_objective(x)
    
    try:
        result = differential_evolution(
            de_objective, bounds,
            maxiter=maxiter, popsize=popsize,
            seed=int(rng.randint(0, 2**31)),
            tol=1e-12, atol=1e-14,
            mutation=(0.5, 1.5), recombination=0.9,
            init='latinhypercube',
            polish=True,
            updating='deferred'
        )
        return _clamp_bary(result.x.reshape(N, 2))
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    best_bary2d = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    initial_bary_list = []
    
    # Structured initializations
    n_structured = 6
    for i in range(n_structured):
        if time_left() < 30.0:
            break
        bary2d = _structured_init_bary(seed=42 + i)
        initial_bary_list.append(bary2d)
    
    # FPS initialization
    if time_left() > 30.0:
        bary2d_fps = _fps_init()
        initial_bary_list.append(bary2d_fps)
    
    # Random initializations
    n_random = 8
    for i in range(n_random):
        if time_left() < 25.0:
            break
        bary2d = _random_init_bary(seed=100 + i)
        initial_bary_list.append(bary2d)
    
    # Optimize each initialization with soft-min + annealing
    for start_bary in initial_bary_list:
        if time_left() < 15.0:
            break
        
        try:
            # Soft-min optimization with moderate beta, then anneal
            refined = _l_bfgs_softmin(start_bary, beta=60.0, maxiter=200)
            score = _score_bary(refined)
            if score > best_score:
                best_score = score
                best_bary2d = refined.copy()
            
            # Quick anneal on the best result from this init
            if time_left() > 20.0:
                refined2 = _beta_anneal_refine(refined)
                score2 = _score_bary(refined2)
                if score2 > best_score:
                    best_score = score2
                    best_bary2d = refined2.copy()
        except Exception:
            pass
    
    # Polish best solution with beta annealing
    if best_bary2d is not None and time_left() > 30.0:
        try:
            refined = _beta_anneal_refine(best_bary2d)
            score = _score_bary(refined)
            if score > best_score:
                best_score = score
                best_bary2d = refined.copy()
        except Exception:
            pass
    
    # Differential evolution global search
    if time_left() > 40.0:
        try:
            de_result = _differential_evolution_global(rng, maxiter=200, popsize=12)
            if de_result is not None:
                score = _score_bary(de_result)
                if score > best_score:
                    best_score = score
                    best_bary2d = de_result.copy()
        except Exception:
            pass
    
    # Additional refinement from perturbed best solution with adaptive noise
    if best_bary2d is not None and time_left() > 25.0:
        noise_scale = 0.01
        for i in range(12):
            if time_left() < 10.0:
                break
            try:
                noise = rng.normal(0, noise_scale, size=2 * N)
                perturbed = best_bary2d + noise.reshape(N, 2)
                perturbed = _clamp_bary(perturbed)
                
                refined = _l_bfgs_softmin(perturbed, beta=160.0, maxiter=200)
                score = _score_bary(refined)
                if score > best_score:
                    best_score = score
                    best_bary2d = refined.copy()
                    noise_scale *= 0.7  # Reduce noise on improvement
                else:
                    noise_scale *= 1.2  # Increase noise on failure
                    noise_scale = min(noise_scale, 0.05)
            except Exception:
                pass
    
    # SLSQP refinement
    if best_bary2d is not None and time_left() > 20.0:
        try:
            refined = _slsqp_refine(best_bary2d, maxiter=400)
            score = _score_bary(refined)
            if score > best_score:
                best_score = score
                best_bary2d = refined.copy()
        except Exception:
            pass
    
    # Nelder-Mead refinement
    if best_bary2d is not None and time_left() > 15.0:
        try:
            refined = _nelder_mead_refine(best_bary2d, maxiter=800)
            score = _score_bary(refined)
            if score > best_score:
                best_score = score
                best_bary2d = refined.copy()
        except Exception:
            pass
    
    # Final high-precision polish: multiple passes with increasing precision
    if best_bary2d is not None and time_left() > 10.0:
        # Pass 1: L-BFGS hard with tight tolerance
        try:
            refined = _l_bfgs_hard(best_bary2d, maxiter=500)
            score = _score_bary(refined)
            if score > best_score:
                best_score = score
                best_bary2d = refined.copy()
        except Exception:
            pass
        
        # Pass 2: SLSQP with very tight tolerance
        if best_bary2d is not None and time_left() > 5.0:
            try:
                refined = _slsqp_refine(best_bary2d, maxiter=500)
                score = _score_bary(refined)
                if score > best_score:
                    best_score = score
                    best_bary2d = refined.copy()
            except Exception:
                pass
        
        # Pass 3: Another L-BFGS hard pass
        if best_bary2d is not None and time_left() > 3.0:
            try:
                refined = _l_bfgs_hard(best_bary2d, maxiter=300)
                score = _score_bary(refined)
                if score > best_score:
                    best_score = score
                    best_bary2d = refined.copy()
            except Exception:
                pass
    
    # Convert best barycentric to Cartesian
    if best_bary2d is None:
        # Fallback: use structured init
        bary2d = _structured_init_bary(seed=42)
    else:
        bary2d = _clamp_bary(best_bary2d)
    
    pts = _bary_to_cart(bary2d)
    return np.array(pts)

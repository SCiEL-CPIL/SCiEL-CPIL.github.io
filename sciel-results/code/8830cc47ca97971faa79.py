import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 16
D = 2
NP = N * (N - 1) // 2  # number of pairwise distances

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _soft_objective(params, beta=50.0):
    """
    Smooth surrogate objective using log-sum-exp with beta parameter.
    beta controls sharpness: larger beta -> closer to exact min/max.
    params: flattened array of shape (2*(N-2),) representing points 2..N-1.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max: max + (1/beta) * log(sum(exp(beta * (x - max))))
    max_d_sq = np.max(d_sq)
    soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
    
    # Soft min: min - (1/beta) * log(sum(exp(-beta * (x - min))))
    min_d_sq = np.min(d_sq)
    soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
    
    if soft_min_d_sq < 1e-8:
        return 1e10
    
    return soft_max_d_sq / soft_min_d_sq

def _soft_gradient(params, beta=50.0):
    """
    Compute gradient of the soft objective w.r.t. params.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max weights
    max_d_sq = np.max(d_sq)
    exp_max = np.exp(beta * (d_sq - max_d_sq))
    w_max = exp_max / np.sum(exp_max)
    soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(exp_max))
    
    # Soft min weights
    min_d_sq = np.min(d_sq)
    exp_min = np.exp(-beta * (d_sq - min_d_sq))
    w_min = exp_min / np.sum(exp_min)
    soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(exp_min))
    
    if soft_min_d_sq < 1e-8:
        return np.zeros_like(params)
    
    # Objective: f = soft_max / soft_min
    # df/d(soft_max) = 1/soft_min
    # df/d(soft_min) = -soft_max / soft_min^2
    
    # Gradient of soft_max w.r.t. d_sq: beta * w_max
    # Gradient of soft_min w.r.t. d_sq: -beta * w_min
    
    grad_d_sq = (1.0 / soft_min_d_sq) * beta * w_max + (soft_max_d_sq / (soft_min_d_sq ** 2)) * beta * w_min
    
    # Now compute gradient of d_sq w.r.t. params
    grad_P = np.zeros((N, 2))
    iu = np.triu_indices(N, 1)
    for idx in range(NP):
        i, j = iu[0][idx], iu[1][idx]
        g = 2.0 * grad_d_sq[idx]
        grad_P[i] += g * (P[i] - P[j])
        grad_P[j] += g * (P[j] - P[i])
    
    # Only points 2..N-1 are free
    grad_params = grad_P[2:].flatten()
    
    return grad_params

def _exact_objective(params):
    """
    Exact objective for local refinement.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _score(P)

def _get_initial_candidates():
    """
    Generate diverse initial configurations.
    Returns list of (N, 2) arrays.
    """
    candidates = []
    
    # 1. Hexagonal Lattice Subset (primary seed from inspiration)
    R = 3
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    
    # Select 16 points closest to the origin
    dists = np.sum(hex_pts**2, axis=1)
    idx = np.argsort(dists)[:16]
    hex_seed = hex_pts[idx]
    candidates.append(hex_seed)
    
    # Perturbed hexagonal seeds
    rng = np.random.default_rng(123)
    for sigma in [0.05, 0.1, 0.15]:
        perturbed = hex_seed + rng.normal(0, sigma, size=(16, 2))
        candidates.append(perturbed)
    
    # 2. 4x4 Grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    candidates.append(grid)
    
    # 3. Two Concentric Circles (8 inner, 8 outer)
    pts_circ = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_circ.append([2 * np.cos(ang), 2 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_circ.append([1 * np.cos(ang), 1 * np.sin(ang)])
    candidates.append(np.array(pts_circ))
    
    # 4. Random Perturbed Grid
    rng = np.random.default_rng(42)
    perturbed_grid = grid + rng.normal(0, 0.1, grid.shape)
    candidates.append(perturbed_grid)
    
    # 5. Random Gaussian Cluster
    rng = np.random.default_rng(123)
    gauss = rng.normal(0, 1, (N, 2))
    candidates.append(gauss)
    
    # 6. Ring of 16
    pts_ring = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts_ring.append([2 * np.cos(ang), 2 * np.sin(ang)])
    candidates.append(np.array(pts_ring))
    
    # 7. Two rings: 8 and 8, different radii
    pts_2rings = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_2rings.append([1.5 * np.cos(ang), 1.5 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_2rings.append([0.5 * np.cos(ang), 0.5 * np.sin(ang)])
    candidates.append(np.array(pts_2rings))
    
    # 8. Three rings: 6, 5, 5
    pts_3rings = []
    for k in range(6):
        ang = 2 * np.pi * k / 6
        pts_3rings.append([1.8 * np.cos(ang), 1.8 * np.sin(ang)])
    for k in range(5):
        ang = 2 * np.pi * (k + 0.5) / 5
        pts_3rings.append([1.0 * np.cos(ang), 1.0 * np.sin(ang)])
    for k in range(5):
        ang = 2 * np.pi * k / 5
        pts_3rings.append([0.4 * np.cos(ang), 0.4 * np.sin(ang)])
    candidates.append(np.array(pts_3rings))
    
    # 9. Square spiral pattern
    pts_spiral = []
    angles = np.linspace(0, 4 * np.pi, 16, endpoint=False)
    radii = np.linspace(0.3, 2.0, 16)
    for a, r in zip(angles, radii):
        pts_spiral.append([r * np.cos(a), r * np.sin(a)])
    candidates.append(np.array(pts_spiral))
    
    # 10. Random uniform in square
    rng = np.random.default_rng(777)
    rand_pts = rng.uniform(-1.5, 1.5, size=(16, 2))
    candidates.append(rand_pts)
    
    return candidates

def _prepare_for_optimization(P):
    """
    Fix P[0]=(0,0) and P[1]=(1,0) by finding the min-distance pair in P,
    rotating/ translating/scaling the whole set so that pair becomes (0,0) and (1,0).
    """
    d_sq = _pairwise_dists_sq(P)
    min_idx = np.argmin(d_sq)
    iu = np.triu_indices(N, 1)
    p, q = iu[0][min_idx], iu[1][min_idx]
    
    p0 = P[p]
    p1 = P[q]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return None, None
        
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (P - p0) @ R_mat.T / norm_v
    
    perm = list(range(N))
    perm[0] = p
    perm[1] = q
    remaining = [idx for idx in range(N) if idx != p and idx != q]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    init_params = transformed[2:].flatten()
    
    return init_params, perm

def _anneal_soft_beta(params, start_beta=20.0, end_beta=200.0, n_steps=4):
    """
    Refine params using annealed soft objective with increasing beta (sharper approximation).
    Uses L-BFGS-B with gradient.
    """
    current_params = params.copy()
    
    for step in range(n_steps):
        beta = start_beta * (end_beta / start_beta) ** (step / max(n_steps - 1, 1))
        
        def obj(p):
            return _soft_objective(p, beta=beta)
        
        def grad(p):
            return _soft_gradient(p, beta=beta)
        
        try:
            res = minimize(
                obj,
                current_params,
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            current_params = res.x
        except Exception:
            break
    
    return current_params

def _powell_refine(params, maxiter=100):
    """
    Powell method refinement (derivative-free) to escape local minima.
    """
    try:
        res = minimize(
            _exact_objective,
            params,
            method='Powell',
            options={'maxiter': maxiter, 'maxfev': maxiter * 10, 'disp': False}
        )
        return res.x
    except Exception:
        return params

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (150s limit, leave 40s buffer)
    
    candidates = _get_initial_candidates()
    
    best_score = float('inf')
    best_params = None
    
    results = []
    
    # Phase 1: Multi-start optimization with Powell bridge + L-BFGS-B annealing
    for i, P0 in enumerate(candidates):
        if time.time() > deadline - 50:  # Leave 50s for final polish
            break
            
        init_params, perm = _prepare_for_optimization(P0)
        if init_params is None:
            continue
        
        # Step 1a: Initial exact score
        P_init = np.zeros((N, 2))
        P_init[0] = [0.0, 0.0]
        P_init[1] = [1.0, 0.0]
        P_init[2:] = init_params.reshape(-1, 2)
        init_score = _score(P_init)
        
        # Step 1b: Powell refinement (derivative-free, good for non-smooth landscape)
        if time.time() < deadline - 30:
            try:
                powell_params = _powell_refine(init_params, maxiter=80)
                P_powell = np.zeros((N, 2))
                P_powell[0] = [0.0, 0.0]
                P_powell[1] = [1.0, 0.0]
                P_powell[2:] = powell_params.reshape(-1, 2)
                powell_score = _score(P_powell)
                
                if powell_score < best_score:
                    best_score = powell_score
                    best_params = powell_params.copy()
                
                # Use Powell result as starting point for gradient refinement
                current_params = powell_params
            except Exception:
                current_params = init_params
        else:
            current_params = init_params
        
        # Step 1c: Annealed L-BFGS-B with soft objective (increasing beta)
        if time.time() < deadline - 20:
            try:
                refined_params = _anneal_soft_beta(current_params, start_beta=20.0, end_beta=150.0, n_steps=4)
                
                P_refined = np.zeros((N, 2))
                P_refined[0] = [0.0, 0.0]
                P_refined[1] = [1.0, 0.0]
                P_refined[2:] = refined_params.reshape(-1, 2)
                
                refined_score = _score(P_refined)
                
                if refined_score < best_score:
                    best_score = refined_score
                    best_params = refined_params.copy()
                
                results.append((refined_score, refined_params, P0))
                
            except Exception:
                results.append((init_score, init_params, P0))
    
    # Phase 2: Local refinement on top K candidates with alternating Powell/L-BFGS-B
    results.sort(key=lambda x: x[0])
    
    top_k = min(6, len(results))
    if time.time() < deadline - 15:
        for score, params, P0 in results[:top_k]:
            if time.time() > deadline - 8:
                break
            
            current = params.copy()
            current_score = score
            
            # Alternating refinement: Powell -> L-BFGS-B -> Powell -> L-BFGS-B
            for round_idx in range(3):
                if time.time() > deadline - 5:
                    break
                
                # Powell step
                if time.time() < deadline - 10:
                    try:
                        p_powell = _powell_refine(current, maxiter=60)
                        P_p = np.zeros((N, 2))
                        P_p[0] = [0.0, 0.0]
                        P_p[1] = [1.0, 0.0]
                        P_p[2:] = p_powell.reshape(-1, 2)
                        s_p = _score(P_p)
                        
                        if s_p < current_score:
                            current = p_powell
                            current_score = s_p
                    except Exception:
                        pass
                
                # L-BFGS-B with high beta
                if time.time() < deadline - 5:
                    try:
                        def tight_obj(p):
                            return _soft_objective(p, beta=200.0)
                        
                        def tight_grad(p):
                            return _soft_gradient(p, beta=200.0)
                        
                        res_tight = minimize(
                            tight_obj,
                            current,
                            jac=tight_grad,
                            method='L-BFGS-B',
                            options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12}
                        )
                        
                        P_t = np.zeros((N, 2))
                        P_t[0] = [0.0, 0.0]
                        P_t[1] = [1.0, 0.0]
                        P_t[2:] = res_tight.x.reshape(-1, 2)
                        s_t = _score(P_t)
                        
                        if s_t < current_score:
                            current = res_tight.x
                            current_score = s_t
                    except Exception:
                        pass
            
            if current_score < best_score:
                best_score = current_score
                best_params = current.copy()
    
    # Phase 3: Final high-precision polish on the best result
    if best_params is not None and time.time() < deadline - 3:
        try:
            # Very tight Nelder-Mead on exact objective
            res_polish = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            P_polish = np.zeros((N, 2))
            P_polish[0] = [0.0, 0.0]
            P_polish[1] = [1.0, 0.0]
            P_polish[2:] = res_polish.x.reshape(-1, 2)
            
            polish_score = _score(P_polish)
            
            if polish_score < best_score:
                best_score = polish_score
                best_params = res_polish.x.copy()
                
        except Exception:
            pass
    
    # Construct final points
    if best_params is not None:
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        # Check validity
        if _score(final_pts) > 1e6:
            return candidates[0]
            
        return final_pts
    else:
        return candidates[0]

# EVOLVE-BLOCK-END

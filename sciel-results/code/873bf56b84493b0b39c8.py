"""Improved program: optimize 16 points via smooth surrogate + L-BFGS-B with annealing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 16
D = 2
DEADLINE = 110  # safety margin before 120s kill

def _pairwise_dists_sq(pts):
    """Compute all pairwise squared distances (N x N matrix, zero diagonal)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(d2, np.inf)  # exclude self-pairs
    return d2

def _score(pts):
    """Compute (max_dist / min_dist)^2."""
    d2 = _pairwise_dists_sq(pts)
    d2_min = np.min(d2)
    d2_max = np.max(d2)
    if d2_min < 1e-15:
        return 1e10
    return (d2_max / d2_min)

def _soft_objective(pts, beta, t):
    """
    Smooth surrogate: minimize log-sum-exp(beta * log(d2)) - log-sum-exp(-beta * log(d2))
    which approximates log(d2_max) - log(d2_min) = log(ratio).
    We minimize the ratio itself via: exp(soft_max - soft_min)
    Actually, let's use: minimize (soft_max_d2 / soft_min_d2) where
    soft_max_d2 = (1/beta)*log(sum(exp(beta*d2)))
    soft_min_d2 = -(1/beta)*log(sum(exp(-beta*d2)))
    This is smooth and differentiable.
    """
    d2 = _pairwise_dists_sq(pts)
    # For numerical stability
    d2_clipped = np.clip(d2, 1e-10, None)
    
    # Soft max of d2
    m = np.max(d2_clipped)
    soft_max = m + np.log(np.sum(np.exp(beta * (d2_clipped - m)))) / beta
    
    # Soft min of d2
    m2 = np.min(d2_clipped)
    soft_min = m2 - np.log(np.sum(np.exp(-beta * (d2_clipped - m2)))) / beta
    
    if soft_min < 1e-15:
        return 1e10
    return soft_max / soft_min

def _soft_objective_grad(pts, beta):
    """Gradient of the soft objective w.r.t. points."""
    d2 = _pairwise_dists_sq(pts)
    d2_clipped = np.clip(d2, 1e-10, None)
    
    # Soft max
    m = np.max(d2_clipped)
    exp_terms = np.exp(beta * (d2_clipped - m))
    sum_exp = np.sum(exp_terms)
    soft_max = m + np.log(sum_exp) / beta
    # d(soft_max)/d(d2_ij) = exp_terms_ij / sum_exp
    w_max = exp_terms / sum_exp
    
    # Soft min
    m2 = np.min(d2_clipped)
    exp_terms_min = np.exp(-beta * (d2_clipped - m2))
    sum_exp_min = np.sum(exp_terms_min)
    soft_min = m2 - np.log(sum_exp_min) / beta
    # d(soft_min)/d(d2_ij) = -exp_terms_min_ij / sum_exp_min
    w_min = -exp_terms_min / sum_exp_min
    
    if soft_min < 1e-15:
        return 1e10, np.zeros_like(pts)
    
    ratio = soft_max / soft_min
    # d(ratio)/d(d2_ij) = (w_max_ij * soft_min - soft_max * w_min_ij) / soft_min^2
    # = (w_max_ij - ratio * w_min_ij) / soft_min
    grad_d2 = (w_max - ratio * w_min) / soft_min
    
    # grad_d2 is symmetric, diagonal is inf but we set it to 0 effectively
    np.fill_diagonal(grad_d2, 0.0)
    
    # Gradient w.r.t. points: d(d2_ij)/d(p_k) = 2*(p_i - p_j) * delta_{k,i} - 2*(p_i - p_j) * delta_{k,j}
    # So grad_pts[k] = sum_{j != k} 2 * grad_d2[k,j] * (p_k - p_j)
    grad_pts = np.zeros_like(pts)
    for k in range(N):
        for j in range(N):
            if k == j:
                continue
            grad_pts[k] += 2.0 * grad_d2[k, j] * (pts[k] - pts[j])
    
    return ratio, grad_pts

def _regular_hex_init():
    """Regular hexagonal-like arrangement for 16 points."""
    # Use a 4x4 grid perturbed slightly
    pts = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    # Center it
    pts -= pts.mean(axis=0)
    return pts

def _golden_spiral_init():
    """Golden angle spiral initial configuration."""
    pts = np.zeros((N, 2))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        r = np.sqrt(i + 1)
        theta = i * golden_angle
        pts[i, 0] = r * np.cos(theta)
        pts[i, 1] = r * np.sin(theta)
    return pts

def _random_init(seed):
    """Random initialization in unit square, centered."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 2)
    pts -= pts.mean(axis=0)
    # Scale to reasonable size
    pts *= 2.0
    return pts

def _fourier_circle_init():
    """Points on a circle with small perturbations."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def construct():
    start_time = time.time()
    
    # Try multiple initializations with annealed soft objective
    best_pts = None
    best_score = 1e10
    
    inits = [
        _regular_hex_init,
        _golden_spiral_init,
        _fourier_circle_init,
    ]
    
    # Add a few random inits
    for seed in [42, 123, 456]:
        inits.append(lambda s=seed: _random_init(s))
    
    beta_schedule = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    for init_fn in inits:
        if time.time() - start_time > DEADLINE:
            break
        
        pts0 = init_fn()
        
        # Annealing schedule
        for beta in beta_schedule:
            if time.time() - start_time > DEADLINE:
                break
            
            def obj_wrapper(x):
                val, _ = _soft_objective_grad(x.reshape(N, 2), beta)
                return val
            
            def grad_wrapper(x):
                _, g = _soft_objective_grad(x.reshape(N, 2), beta)
                return g.flatten()
            
            try:
                res = minimize(
                    obj_wrapper,
                    pts0.flatten(),
                    jac=grad_wrapper,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts0 = res.x.reshape(N, 2)
            except Exception:
                break
        
        # Final polish with exact objective
        if time.time() - start_time > DEADLINE:
            break
        
        # Compute actual score
        d2 = _pairwise_dists_sq(pts0)
        d2_min = np.min(d2)
        d2_max = np.max(d2)
        if d2_min < 1e-15:
            continue
        actual_score = d2_max / d2_min
        
        if actual_score < best_score:
            best_score = actual_score
            best_pts = pts0.copy()
    
    # Final high-precision polish on best solution
    if best_pts is not None and time.time() - start_time < DEADLINE:
        def exact_obj(x):
            pts = x.reshape(N, 2)
            d2 = _pairwise_dists_sq(pts)
            d2_min = np.min(d2)
            d2_max = np.max(d2)
            if d2_min < 1e-15:
                return 1e10
            return d2_max / d2_min
        
        # Use Nelder-Mead for final polish (no gradient needed, robust)
        try:
            res = minimize(
                exact_obj,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate = res.x.reshape(N, 2)
            d2 = _pairwise_dists_sq(candidate)
            d2_min = np.min(d2)
            d2_max = np.max(d2)
            if d2_min > 1e-15:
                cand_score = d2_max / d2_min
                if cand_score < best_score:
                    best_pts = candidate
                    best_score = cand_score
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
    
    return best_pts

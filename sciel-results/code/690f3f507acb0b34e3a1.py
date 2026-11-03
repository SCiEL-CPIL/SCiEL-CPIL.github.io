"""Rewritten packing solver: multi-start gradient ascent on the minimum-pair-distance objective."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
HALF = 0.5
EPS = 1e-12


def _score(c):
    """Compute the packing score (minimum radius) for centre array c of shape (N,3)."""
    # Boundary constraint: distance to nearest face
    bound = HALF - np.max(np.abs(c), axis=1)
    r_bound = np.min(bound)
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * np.min(d)
    return min(r_bound, r_pair)


def _neg_score(c):
    return -_score(c)


def _neg_score_grad(c):
    """
    Gradient of the negative score.
    Score = min( min_i (0.5 - |c_i|_inf), 0.5 * min_{i<j} |c_i - c_j| )
    We approximate the subgradient of the min by using the active constraint(s).
    For simplicity and robustness, we use a smooth approximation:
    f(c) = -soft_min of the candidate radii.
    """
    # Compute all candidate radius values
    # Boundary radii
    bound = HALF - np.max(np.abs(c), axis=1)  # shape (N,)
    # Pairwise radii (only i<j)
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    pair_r = 0.5 * d  # shape (N, N)

    # We want to maximize r = min(all candidate radii)
    # Use a log-sum-exp style soft min for smoothness
    # Actually, let's just do a hard min with subgradient for the active constraint.
    # Simpler: use the fact that scipy's L-BFGS-B can handle non-smooth functions
    # via finite differences, but analytical grad is faster.
    
    # Let's use a differentiable proxy: -softmin
    # softmin(x, t) = -t * log(sum(exp(-x_i / t)))
    # We'll use a small temperature
    t = 0.01
    all_r = np.concatenate([bound, pair_r[np.triu_indices(N, 1)]])
    
    # Gradient of softmin:
    # d/dx_j [-t * log(sum_i exp(-x_i/t))] = (1/sum_i exp(-x_i/t)) * exp(-x_j/t)
    # This gives a probability distribution over the constraints.
    
    # But we need gradient w.r.t. c, not w.r.t. x.
    # This is complex. Let's just use finite difference for the gradient,
    # but that's slow. Alternatively, use the subgradient of the hard min.
    
    # For the hard min, the subgradient is a convex combination of the gradients
    # of the active constraints. Let's find which constraint is active.
    
    r_val = min(np.min(bound), np.min(pair_r))
    
    grad = np.zeros_like(c)
    
    # Active boundary constraints
    active_bound = np.where(np.abs(bound - r_val) < 1e-12)[0]
    for i in active_bound:
        # grad of (0.5 - |c_i|_inf) w.r.t. c_i
        # |c_i|_inf = max(|c_i0|, |c_i1|, |c_i2|)
        abs_c = np.abs(c[i])
        max_abs = np.max(abs_c)
        active_dim = np.where(np.abs(abs_c - max_abs) < 1e-12)[0]
        for k in active_dim:
            grad[i, k] += -np.sign(c[i, k])
    
    # Active pairwise constraints
    i_idx, j_idx = np.where(np.abs(pair_r - r_val) < 1e-12)
    for i, j in zip(i_idx, j_idx):
        if i >= j:
            continue
        diff_ij = c[i] - c[j]
        d_ij = np.sqrt(np.sum(diff_ij * diff_ij))
        if d_ij < EPS:
            # Coincident points; use random direction
            grad[i] += np.random.randn(3)
            grad[j] -= np.random.randn(3)
        else:
            # grad of 0.5 * |c_i - c_j| w.r.t. c_i is 0.5 * diff_ij / |diff_ij|
            # grad w.r.t. c_j is -0.5 * diff_ij / |diff_ij|
            unit = diff_ij / d_ij
            grad[i] += 0.5 * unit
            grad[j] -= 0.5 * unit
    
    # Normalize the subgradient (it's a convex combination)
    # The subgradient of min is a convex combo of the active gradients.
    # We just sum them and normalize.
    norm = np.sqrt(np.sum(grad * grad))
    if norm > EPS:
        grad /= norm
    
    return -grad  # Negative because we're minimizing -score


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_score = -np.inf
    best_c = None
    
    # Initial guess: simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_c = grid[:N].copy()
    
    # Try multiple random initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            c = init_c.copy()
        else:
            # Random perturbation of the grid
            c = init_c + 0.05 * np.random.randn(N, 3)
            # Or fully random
            if restart % 2 == 0:
                c = np.random.uniform(-0.4, 0.4, (N, 3))
        
        # Ensure points are inside the cube
        c = np.clip(c, -HALF + 0.01, HALF - 0.01)
        
        # Optimize
        try:
            result = minimize(
                _neg_score,
                c,
                method='L-BFGS-B',
                jac=_neg_score_grad,
                bounds=[(-HALF + 1e-6, HALF - 1e-6)] * N * 3,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            c_opt = result.x
            score_opt = _score(c_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_c = c_opt.copy()
        except Exception:
            continue
    
    # Final polish: try a few more iterations from the best solution
    if best_c is not None and time.time() < deadline:
        try:
            result = minimize(
                _neg_score,
                best_c,
                method='L-BFGS-B',
                jac=_neg_score_grad,
                bounds=[(-HALF + 1e-6, HALF - 1e-6)] * N * 3,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_opt = result.x
            score_opt = _score(c_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # Ensure validity: clip to cube
    if best_c is None:
        return init_c[:N]
    
    best_c = np.clip(best_c, -HALF, HALF)
    return best_c

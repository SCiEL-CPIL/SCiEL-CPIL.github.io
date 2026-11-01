"""Packing of 50 equal circles in a unit square via Log-Sum-Exp optimization + SA + L-BFGS polish."""
import numpy as np
import time
import sys

N = 50
HALF = 0.5


def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r


def _log_sum_exp_obj(pts, beta):
    """
    Objective: -log(sum(exp(-beta * distances)))
    Minimizing this is equivalent to maximizing the soft-min of distances.
    We include pairwise distances and wall distances.
    """
    # Pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    
    # Wall
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    # Combine distances
    # To avoid overflow in exp, we use the log-sum-exp trick:
    # log(sum(exp(-beta * d))) = -beta * max_d + log(sum(exp(-beta * (d - max_d))))
    # We want to minimize -log(sum(exp(-beta * d))) = beta * max_d - log(sum(exp(-beta * (d - max_d))))
    
    all_dists = np.concatenate([pair_dist, wall_dist])
    max_d = np.max(all_dists)
    # exp(-beta * (d - max_d)) is safe because d <= max_d, so exponent >= 0? 
    # Wait, d - max_d <= 0, so -beta * (d - max_d) >= 0. 
    # If beta is large and d is small, this can overflow.
    # Let's check: d is min distance. max_d is max distance.
    # If d is very small, d - max_d is large negative. -beta * (d - max_d) is large positive.
    # So we need to be careful.
    
    # Actually, standard log-sum-exp for log(sum(exp(x_i))) uses max_x.
    # Here x_i = -beta * d_i.
    # max_x = -beta * min_d.
    # log(sum(exp(-beta * d_i))) = -beta * min_d + log(sum(exp(-beta * (d_i - min_d))))
    # Since d_i >= min_d, d_i - min_d >= 0, so -beta * (d_i - min_d) <= 0. No overflow.
    
    min_d = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - min_d))
    sum_exp = np.sum(exp_terms)
    
    log_sum_exp_val = -beta * min_d + np.log(sum_exp)
    
    return -log_sum_exp_val # We minimize this, so we return the negative


def _log_sum_exp_grad(pts, beta):
    """
    Gradient of -log(sum(exp(-beta * distances)))
    = - (1/sum(exp(-beta * d))) * sum(exp(-beta * d) * (-beta) * grad_d)
    = beta * (sum(w_i * grad_d_i)) / sum(w_i)
    where w_i = exp(-beta * d_i)
    
    Note: The objective is J = -log(S). Grad J = - (1/S) * Grad S.
    S = sum(exp(-beta * d_k)).
    Grad S = sum(exp(-beta * d_k) * (-beta) * grad d_k).
    Grad J = - (1/S) * (-beta) * sum(w_k * grad d_k) = beta * (sum(w_k * grad d_k)) / S.
    
    We want to minimize J. So we move in direction -Grad J.
    """
    # Pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    dist_safe = np.where(dist > 1e-12, dist, 1e-12)
    
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    
    # Wall
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    all_dists = np.concatenate([pair_dist, wall_dist])
    min_d = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - min_d))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp # Normalized weights w_k / S
    
    grad = np.zeros_like(pts)
    
    # Pairwise contribution
    # For pair (i,j), d = ||x_i - x_j||. grad_{x_i} d = (x_i - x_j)/d.
    # Contribution to sum(w_k * grad d_k) from pair (i,j):
    # w_pair * [ (x_i - x_j)/d for x_i, (x_j - x_i)/d for x_j ]
    
    n_pairs = len(iu[0])
    for idx in range(n_pairs):
        i, j = iu[0][idx], iu[1][idx]
        d = dist[i, j]
        if d < 1e-12:
            continue
        w = weights[idx] # weight for this pair
        diff_ij = (pts[i] - pts[j]) / d
        # Grad J = beta * ( ... ). We are computing the term inside the sum for the gradient direction.
        # Actually, we just need the gradient of the objective.
        # Grad J = beta * ( sum w_k grad d_k ) / S = beta * sum (w_k/S) grad d_k.
        # So we accumulate w_k * grad d_k.
        
        grad[i] += w * diff_ij
        grad[j] -= w * diff_ij
        
    # Wall contribution
    # For circle i, wall_dist = min(d_left, d_right, d_bottom, d_top).
    # The gradient is non-smooth at ties, but we can pick the active constraint or average.
    # For a smooth approximation, we can use the same weighting logic, but wall distances are separate terms in the sum.
    # Wait, in the objective, we treated wall distances as individual terms in the sum.
    # So for circle i, we have 4 terms: d_left_i, d_right_i, etc.
    # Their weights are in the second half of the 'weights' array.
    
    w_start = n_pairs
    for i in range(N):
        # Weights for the 4 walls of circle i
        w_l = weights[w_start + 4*i]
        w_r = weights[w_start + 4*i + 1]
        w_b = weights[w_start + 4*i + 2]
        w_t = weights[w_start + 4*i + 3]
        
        # Gradients of distances w.r.t. pts[i]
        # d_left = x_i + 0.5 -> grad_x = 1, grad_y = 0
        # d_right = 0.5 - x_i -> grad_x = -1, grad_y = 0
        # d_bottom = y_i + 0.5 -> grad_x = 0, grad_y = 1
        # d_top = 0.5 - y_i -> grad_x = 0, grad_y = -1
        
        grad[i, 0] += w_l * 1.0 - w_r * 1.0
        grad[i, 1] += w_b * 1.0 - w_t * 1.0
        
    return beta * grad


def _gradient_descent(pts0, beta, deadline, lr=0.001, decay=0.999, max_steps=5000):
    """
    Perform gradient descent on the log-sum-exp objective.
    """
    pts = pts0.copy()
    current_obj = _log_sum_exp_obj(pts, beta)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        grad = _log_sum_exp_grad(pts, beta)
        
        # Adaptive step size: simple decay
        # We can also use a line search or barzilai-borwein, but let's keep it simple first.
        # To ensure stability, we might need to normalize the gradient or use a smaller lr.
        # The gradient magnitude scales with beta.
        
        pts -= lr * grad
        
        # Project back to square
        pts = np.clip(pts, -HALF, HALF)
        
        new_obj = _log_sum_exp_obj(pts, beta)
        
        # If objective increased (which it shouldn't for a descent direction, but numerics can cause issues)
        # or if we want to be safe, we can revert.
        # However, with a fixed lr, it might oscillate.
        # Let's add a simple check: if new_obj > current_obj, shrink lr and retry? 
        # For now, let's just proceed.
        
        current_obj = new_obj
        lr *= decay
        
    return pts


def _simulated_annealing(pts0, deadline, rng, n_steps=2000, beta=50.0):
    """
    Simulated annealing using the log-sum-exp objective.
    """
    pts = pts0.copy()
    current_obj = _log_sum_exp_obj(pts, beta)
    best_pts = pts.copy()
    best_obj = current_obj
    
    t_start = 0.01
    t_end = 0.0001
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        # Random move
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_obj = _log_sum_exp_obj(new_pts, beta)
        
        delta = new_obj - current_obj
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
    
    return best_pts


def _lbfgs_polish(pts0, beta, deadline, maxiter=500):
    """
    Use L-BFGS-B to minimize the log-sum-exp objective.
    """
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _log_sum_exp_obj(p, beta)
        
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _log_sum_exp_grad(p, beta).ravel()
        
    x0 = pts0.ravel()
    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-10})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)


def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)


def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Multi-start initialization and coarse optimization
    inits = [
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.05),
        _grid_init(rng, jitter=0.02),
        _random_init(rng),
    ]
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        
        # Coarse optimization with low beta
        pts_coarse = _gradient_descent(init, beta=20.0, deadline=deadline - 25.0, lr=0.005, max_steps=1000)
        
        # SA to escape local minima
        pts_sa = _simulated_annealing(pts_coarse, deadline - 20.0, rng, n_steps=1000, beta=50.0)
        
        # Fine optimization with higher beta
        pts_fine = _gradient_descent(pts_sa, beta=100.0, deadline=deadline - 15.0, lr=0.001, max_steps=1000)
        
        # L-BFGS polish
        try:
            pts_final = _lbfgs_polish(pts_fine, beta=200.0, deadline=deadline - 10.0, maxiter=300)
        except Exception:
            pts_final = pts_fine
            
        r = _compute_r(pts_final)
        if r > best_r:
            best_r = r
            best_pts = pts_final.copy()
            
    if best_pts is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # Phase 2: Intensive polishing on the best result found
    # Use a very high beta to sharpen the minimum
    if time.time() < deadline - 10.0:
        # SA with high beta to find a better basin
        pts_sa_high = _simulated_annealing(best_pts, deadline - 10.0, rng, n_steps=2000, beta=500.0)
        
        # Gradient descent
        pts_gd_high = _gradient_descent(pts_sa_high, beta=1000.0, deadline=deadline - 5.0, lr=0.0005, max_steps=2000)
        
        # L-BFGS
        try:
            pts_final_high = _lbfgs_polish(pts_gd_high, beta=2000.0, deadline=deadline - 2.0, maxiter=500)
        except Exception:
            pts_final_high = pts_gd_high
            
        r_high = _compute_r(pts_final_high)
        if r_high > best_r:
            best_r = r_high
            best_pts = pts_final_high.copy()
            
    return best_pts

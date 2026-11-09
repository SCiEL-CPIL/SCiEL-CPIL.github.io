"""Model-and-optimize: hexagonal seed + force-based radius maximization with deadline guard."""
import time
import numpy as np

N = 200
HALF = 0.5
SEED = 1234567

def _radius_of(c):
    """Compute packing radius for centers c (N,2)."""
    # wall clearance
    w = HALF - np.abs(c)
    wall = np.min(w)
    # pairwise
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = np.min(dist) / 2.0
    return float(min(wall, pair))


def _hexagon_seed(n):
    """Staggered hexagonal rows inside [-1/2,1/2]^2, return n centers."""
    pts = []
    # choose rows and spacing to roughly fill the square
    for attempt in range(200):
        rows = 14
        r0 = 0.036  # guess
        # hex geometry: vertical spacing = r0*sqrt(3), horizontal = 2*r0
        # place rows centred, stagger
        h = 2 * r0
        v = r0 * np.sqrt(3)
        x0 = -HALF + r0
        y0 = -HALF + r0
        # number of rows fitting
        n_rows = int((1 - 2 * r0) / v) + 1
        pts = []
        for i in range(n_rows):
            y = y0 + i * v
            x_start = x0 + (h / 2 if i % 2 else 0.0)
            n_cols = int((HALF - r0 - x_start) / h) + 1 if x_start <= HALF - r0 else 0
            for j in range(n_cols):
                x = x_start + j * h
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= n:
            # trim to n keeping best (use grid ordering)
            return pts[:n]
        r0 *= 0.995
    # fallback grid
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _soft_min_radius(c, beta):
    """Smooth surrogate for packing radius using soft-min.
    
    Returns (surrogate_value, gradient) where gradient points in direction
    that increases the true radius.
    """
    n = len(c)
    # Wall clearance: min_i min_k (HALF - |c_ik|)
    # For each point, wall clearance is min over 4 walls
    # Soft-min over walls per point, then soft-min over points
    
    # Pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    dist = np.sqrt(np.maximum(dist_sq, 1e-16))
    np.fill_diagonal(dist, np.inf)
    
    # Soft-min of pairwise distances / 2
    # Use log-sum-exp trick for soft-min: softmin(x) = -logsumexp(-x/beta) * beta
    # For numerical stability, subtract max
    pair_vals = dist / 2.0
    # Exclude diagonal (inf)
    pair_vals_masked = np.where(np.isinf(pair_vals), -np.inf, pair_vals)
    max_pair = np.max(pair_vals_masked)
    # Soft min of pairs
    exp_terms = np.exp(-(pair_vals - max_pair) / beta)
    exp_terms[~np.isfinite(exp_terms)] = 0.0
    log_sum = np.log(np.sum(exp_terms) + 1e-300)
    pair_softmin = max_pair - beta * log_sum
    
    # Wall clearance per point
    # For point i: min(HALF - c_ix, HALF + c_ix, HALF - c_iy, HALF + c_iy)
    walls = np.stack([
        HALF - c[:, 0],
        HALF + c[:, 0],
        HALF - c[:, 1],
        HALF + c[:, 1]
    ], axis=1)  # shape (n, 4)
    
    max_wall = np.max(walls, axis=1)  # (n,)
    exp_walls = np.exp(-(walls - max_wall[:, None]) / beta)
    log_sum_walls = np.log(np.sum(exp_walls, axis=1) + 1e-300)
    wall_softmin_per_point = max_wall - beta * log_sum_walls  # (n,)
    
    # Soft-min over all points for wall
    max_wp = np.max(wall_softmin_per_point)
    exp_wp = np.exp(-(wall_softmin_per_point - max_wp) / beta)
    log_sum_wp = np.log(np.sum(exp_wp) + 1e-300)
    wall_softmin = max_wp - beta * log_sum_wp
    
    # Overall soft-min: min(pair_softmin, wall_softmin)
    # Use another soft-min
    vals = np.array([pair_softmin, wall_softmin])
    max_v = np.max(vals)
    exp_v = np.exp(-(vals - max_v) / beta)
    log_sum_v = np.log(np.sum(exp_v) + 1e-300)
    surrogate = max_v - beta * log_sum_v
    
    # Gradient computation
    # d(surrogate)/d(c) = -1/beta * [d(pair_softmin)/dc * w_pair + d(wall_softmin)/dc * w_wall]
    # where weights come from the outer soft-min
    
    w_pair = np.exp(-(pair_softmin - max_v) / beta) / np.sum(exp_v)
    w_wall = np.exp(-(wall_softmin - max_v) / beta) / np.sum(exp_v)
    
    # Gradient of pair_softmin w.r.t. c
    # pair_softmin = max_pair - beta * log(sum(exp(-(dist_ij - max_pair)/beta)))
    # d(pair_softmin)/dc_i = -beta * d/dc_i [log(sum_j exp(-(dist_ij - max_pair)/beta))]
    # = -beta * [1/sum * sum_j d/dc_i exp(-(dist_ij - max_pair)/beta)]
    # = -beta * [1/sum * sum_j exp(...) * (-1/beta) * d(dist_ij)/dc_i]
    # = sum_j [exp(...) / sum] * d(dist_ij)/dc_i
    
    # For pair i,j: d(dist_ij)/dc_i = (c_i - c_j) / dist_ij
    # Weight for pair (i,j) in soft-min:
    # w_ij = exp(-(dist_ij/2 - max_pair)/beta) / sum_k exp(-(dist_kj/2 - max_pair)/beta)
    
    # Compute weights for each pair
    # exp_terms already computed above (excluding inf)
    total_exp = np.sum(exp_terms) + 1e-300
    # Pair weights: for each i, j, weight_ij = exp_terms[i,j] / total_exp
    # But we need to be careful: the soft-min is over all pairs (i,j) with i<j
    # Actually dist matrix is symmetric, so we have each pair twice. Let's handle this.
    
    # Gradient of pair_softmin (vectorized):
    # d(pair_softmin)/dc_i = sum_j w_ij * (c_i - c_j) / dist_ij
    # where w_ij = exp_terms[i, j] / total_exp
    w_mat = exp_terms / total_exp  # (n, n)
    # Zero out diagonal
    np.fill_diagonal(w_mat, 0.0)
    # Safe division: where dist is 0 (shouldn't happen after clip), use 0
    inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
    # grad_pair[i] = sum_j w_mat[i,j] * (c[i] - c[j]) / dist[i,j]
    # = sum_j w_mat[i,j] * c[i] / dist[i,j] - sum_j w_mat[i,j] * c[j] / dist[i,j]
    # = c[i] * sum_j w_mat[i,j] / dist[i,j] - sum_j (w_mat[i,j] / dist[i,j]) * c[j]
    w_inv = w_mat * inv_dist  # (n, n)
    row_sum = np.sum(w_inv, axis=1)  # (n,)
    grad_pair = c * row_sum[:, None] - w_inv @ c
    
    # Gradient of wall_softmin:
    # wall_softmin = max_wp - beta * log(sum_i exp(-(wall_i - max_wp)/beta))
    # d(wall_softmin)/dc_i = sum_k w_ik * d(wall_ik)/dc_i
    # where wall_ik is the k-th wall clearance for point i
    # w_ik = exp(-(wall_ik - max_wall_i)/beta) / sum_k exp(...)
    
    # First, weights over points:
    total_wp = np.sum(exp_wp) + 1e-300
    point_weights = exp_wp / total_wp  # (n,)
    
    # For each point i, weights over 4 walls:
    total_walls = np.sum(exp_walls, axis=1) + 1e-300  # (n,)
    wall_weights = exp_walls / total_walls[:, None]  # (n, 4)
    
    # d(wall_ik)/dc_i:
    # wall_0 = HALF - c_x, d/dc_x = -1
    # wall_1 = HALF + c_x, d/dc_x = +1
    # wall_2 = HALF - c_y, d/dc_y = -1
    # wall_3 = HALF + c_y, d/dc_y = +1
    
    grad_wall = np.zeros_like(c)
    for i in range(n):
        pw = point_weights[i]
        if pw < 1e-15:
            continue
        # Wall 0: HALF - c_x
        grad_wall[i, 0] += pw * wall_weights[i, 0] * (-1.0)
        # Wall 1: HALF + c_x
        grad_wall[i, 0] += pw * wall_weights[i, 1] * (+1.0)
        # Wall 2: HALF - c_y
        grad_wall[i, 1] += pw * wall_weights[i, 2] * (-1.0)
        # Wall 3: HALF + c_y
        grad_wall[i, 1] += pw * wall_weights[i, 3] * (+1.0)
    
    # Total gradient: d(surrogate)/dc = -1/beta * (w_pair * grad_pair + w_wall * grad_wall)
    # But we want to MAXIMIZE radius, so we ascend: c += lr * grad
    # The gradient of surrogate w.r.t. c is what we computed above (negative of what increases radius)
    # Actually let me re-derive: surrogate = softmin of distances
    # To maximize surrogate, we want to move in direction of +d(surrogate)/dc
    
    # d(surrogate)/dc = w_pair * d(pair_softmin)/dc + w_wall * d(wall_softmin)/dc
    # (the -1/beta factors are already included in the derivatives above)
    
    # Wait, let me redo this more carefully.
    # softmin(x_1, ..., x_m) = -beta * log(sum exp(-x_i/beta))
    # d(softmin)/dx_i = -exp(-x_i/beta) / sum exp(-x_j/beta)
    # So d(softmin)/dx_i is negative (as expected: increasing x_i increases softmin)
    
    # For our case:
    # surrogate = softmin(pair_softmin, wall_softmin)
    # d(surrogate)/d(pair_softmin) = -exp(-pair_softmin/beta) / sum
    # d(surrogate)/d(wall_softmin) = -exp(-wall_softmin/beta) / sum
    
    # d(pair_softmin)/dc = sum over pairs of weights * (c_i - c_j)/dist
    # This is the gradient that, when we increase c in this direction, increases pair_softmin
    
    # So d(surrogate)/dc = d(surrogate)/d(pair_softmin) * d(pair_softmin)/dc + d(surrogate)/d(wall_softmin) * d(wall_softmin)/dc
    # = (-w_pair) * grad_pair + (-w_wall) * grad_wall
    # where grad_pair and grad_wall are the gradients that increase pair_softmin and wall_softmin
    
    # To maximize surrogate, we move in direction -d(surrogate)/dc = w_pair * grad_pair + w_wall * grad_wall
    
    gradient = w_pair * grad_pair + w_wall * grad_wall
    
    return surrogate, gradient


def _polish(c0, max_iter=5000, deadline=10.0):
    """Final exact local polish: small steps on the true min-distance objective.
    Uses sub-gradient of the min over all constraints (pairs + walls)."""
    c = c0.copy().astype(float)
    best = c.copy()
    best_r = _radius_of(c)
    n = len(c)
    t0 = time.time()
    
    for it in range(max_iter):
        if time.time() - t0 > deadline:
            break
        
        # Compute all constraints
        d = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_half = dist / 2.0
        
        walls = np.stack([
            HALF - c[:, 0], HALF + c[:, 0],
            HALF - c[:, 1], HALF + c[:, 1]
        ], axis=1)
        
        # Find the minimum constraint value
        pair_min = np.min(pair_half)
        wall_min = np.min(walls)
        r = min(pair_min, wall_min)
        
        # Sub-gradient: move away from the binding constraint(s)
        lr = 0.0002 * (1.0 - it / max_iter) + 0.00002
        grad = np.zeros_like(c)
        
        # Pair contributions (all pairs within epsilon of min)
        eps = 1e-6
        mask_pair = (pair_half <= pair_min + eps) & (pair_half < np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                if mask_pair[i, j]:
                    diff = (c[i] - c[j]) / max(dist[i, j], 1e-12)
                    grad[i] += 0.5 * diff
                    grad[j] += 0.5 * (-diff)
        
        # Wall contributions
        mask_wall = (walls <= wall_min + eps)
        for i in range(n):
            for k in range(4):
                if mask_wall[i, k]:
                    if k == 0:
                        grad[i, 0] -= 0.5
                    elif k == 1:
                        grad[i, 0] += 0.5
                    elif k == 2:
                        grad[i, 1] -= 0.5
                    elif k == 3:
                        grad[i, 1] += 0.5
        
        c_new = c + lr * grad
        c_new = np.clip(c_new, -HALF + 1e-10, HALF - 1e-10)
        
        r_new = _radius_of(c_new)
        if r_new > best_r:
            best_r = r_new
            best = c_new.copy()
        c = c_new
    
    return best, best_r


def _optimize(c0, max_iter, rng, deadline):
    """Gradient-based ascent on soft-min surrogate. Returns best center array."""
    c = c0.copy().astype(float)
    best = c.copy()
    best_r = _radius_of(c)
    n = len(c)
    t0 = time.time()
    
    # Start with larger beta, anneal down
    beta_start = 0.01
    beta_end = 0.001
    lr_start = 0.005
    
    for it in range(max_iter):
        if time.time() - t0 > deadline:
            break
        
        # Anneal beta and learning rate
        frac = min(1.0, it / max_iter)
        beta = beta_start * (beta_end / beta_start) ** frac
        lr = lr_start * (1.0 - 0.5 * frac)
        
        surrogate, grad = _soft_min_radius(c, beta)
        
        # Gradient ascent step
        c_new = c + lr * grad
        
        # Clip to valid region
        c_new = np.clip(c_new, -HALF + 1e-9, HALF - 1e-9)
        
        # Check if improved
        r_new = _radius_of(c_new)
        if r_new > best_r:
            best_r = r_new
            best = c_new.copy()
        
        c = c_new
    
    return best, best_r


def construct():
    t_start = time.time()
    deadline = 110.0  # safety margin vs 150s wall
    rng = np.random.default_rng(SEED)

    c0 = _hexagon_seed(N)
    # clamp strictly inside to avoid invalid
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)

    # Multi-restart: alternate between local optimization from current best
    # and from slightly perturbed starts.
    restarts = 0
    while time.time() - t_start < deadline:
        restarts += 1
        # perturbation magnitude shrinks over restarts
        scale = 0.02 / (1.0 + 0.1 * restarts)
        c_start = best_c + rng.normal(0.0, scale, size=best_c.shape)
        c_start = np.clip(c_start, -HALF + 1e-6, HALF - 1e-6)
        # ensure no coincident centres
        d = c_start[:, None, :] - c_start[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 1e-4:
            c_start = best_c.copy()
        # remaining time for this optimization (reserve 15s for final polish)
        rem = deadline - (time.time() - t_start)
        if rem < 20.0:
            break
        c_opt, r_opt = _optimize(c_start, max_iter=200000, rng=rng, deadline=rem - 18.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt

    # Final polish pass on the best solution
    rem = deadline - (time.time() - t_start)
    if rem > 3.0:
        polished, r_pol = _polish(best_c, max_iter=5000, deadline=rem - 1.0)
        if r_pol > best_r:
            best_r = r_pol
            best_c = polished

    # final safety: ensure validity
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
    return best_c


if __name__ == "__main__":
    c = construct()
    print("shape", c.shape, "radius", _radius_of(c))

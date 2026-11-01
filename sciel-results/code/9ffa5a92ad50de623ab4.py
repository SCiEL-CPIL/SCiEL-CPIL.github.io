"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _score(pos):
    """Largest common radius for a placement (validity assumed)."""
    if pos.shape != (N, 2) or not np.all(np.isfinite(pos)):
        return 0.0
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    min_wall = (0.5 - np.abs(pos)).min()
    return min(0.5 * min_pair, min_wall)


def _soft_min_score(pos, beta=100.0):
    """Smooth surrogate: soft-min of all pairwise half-distances and wall distances."""
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    half_dist = 0.5 * dist
    wall = 0.5 - np.abs(pos)

    all_vals = np.concatenate([half_dist.flatten(), wall.flatten()])
    all_vals = all_vals[all_vals < np.inf]
    if len(all_vals) == 0:
        return 0.0

    m = all_vals.min()
    if m < 0:
        return m
    log_sum = np.log(np.sum(np.exp(-beta * (all_vals - m))))
    soft_min = m - np.log(log_sum) / beta
    return soft_min


def _soft_min_grad(pos, beta=100.0):
    """Gradient of the soft-min score w.r.t. positions."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    half_dist = 0.5 * dist
    wall = 0.5 - np.abs(pos)

    all_vals = np.concatenate([half_dist.flatten(), wall.flatten()])
    all_vals = all_vals[all_vals < np.inf]
    m = all_vals.min()

    log_w = -beta * (all_vals - m)
    log_w_max = log_w.max()
    exp_w = np.exp(log_w - log_w_max)
    w = exp_w / exp_w.sum()

    # Gradient of soft_min w.r.t. pos
    # soft_min = m - (1/beta)*log(sum exp(-beta*(v_i - m)))
    # d(soft_min)/d(v_k) = -w_k * (-beta) * (-1/beta) = -w_k  ... wait let me redo
    # Actually: f = m - (1/beta) * log(S) where S = sum exp(-beta*(v_i - m))
    # df/dv_k = delta_{k,argmin} - (1/beta) * (1/S) * dS/dv_k
    # dS/dv_k = -beta * exp(-beta*(v_k - m)) = -beta * S * w_k
    # So df/dv_k = delta_{k,argmin} + w_k
    # For the argmin element: 1 + w_k (but w_k is close to 1, others close to 0)
    # For others: 0 + w_k = w_k
    # But this doesn't seem right. Let me use the standard result:
    # For softmin with temperature: f = -1/beta * log(sum exp(-beta*v_i))
    # df/dv_k = -w_k where w_k = exp(-beta*v_k)/sum(exp(-beta*v_j))
    # But our f includes m shift: f = m - 1/beta*log(sum exp(-beta*(v_i-m)))
    # = m - 1/beta*log(exp(-beta*m)*sum exp(-beta*v_i))
    # = m - 1/beta*(-beta*m + log(sum exp(-beta*v_i)))
    # = m + m - 1/beta*log(sum exp(-beta*v_i))
    # = 2m - 1/beta*log(sum exp(-beta*v_i))
    # Hmm, that's not right either. Let me just compute numerically.
    
    # Actually the correct formula for softmin:
    # f(v) = -1/beta * log(sum_i exp(-beta*v_i))
    # df/dv_k = -exp(-beta*v_k) / sum_j exp(-beta*v_j) = -w_k
    # where w_k = exp(-beta*v_k)/sum_j exp(-beta*v_j)
    
    # Our version with the m shift is equivalent up to a constant, so gradient is the same.
    
    grad = np.zeros_like(pos)
    
    # Pairwise distances part
    # half_dist[i,j] = 0.5 * ||pos_i - pos_j||
    # d(half_dist[i,j])/d(pos_i) = 0.5 * (pos_i - pos_j) / ||pos_i - pos_j||
    # d(half_dist[i,j])/d(pos_j) = 0.5 * (pos_j - pos_i) / ||pos_i - pos_j||
    
    # Wall distances part
    # wall[i,k] = 0.5 - |pos_i[k]|
    # d(wall[i,k])/d(pos_i[k]) = -sign(pos_i[k])
    
    # Reconstruct weights for pairwise and wall separately
    # For efficiency, let's compute the gradient directly
    
    # Pairwise contribution
    safe_dist = np.maximum(dist, 1e-10)
    unit_d = d / safe_dist[:, :, None]
    
    # w_pair[i,j] for i != j
    log_w_pair = -beta * half_dist
    log_w_pair_max = log_w_pair.max()
    exp_w_pair = np.exp(log_w_pair - log_w_pair_max)
    w_pair = exp_w_pair / exp_w_pair.sum()
    np.fill_diagonal(w_pair, 0.0)
    
    # d(soft_min)/d(pos_i) from pairwise = -sum_j w_pair[i,j] * d(half_dist[i,j])/d(pos_i)
    # = -sum_j w_pair[i,j] * 0.5 * (pos_i - pos_j)/dist[i,j]
    # = -0.5 * sum_j w_pair[i,j] * unit_d[i,j]
    
    grad_pair = -0.5 * np.sum(w_pair[:, :, None] * unit_d, axis=1)
    
    # Wall contribution
    wall_flat = wall.flatten()
    log_w_wall = -beta * wall_flat
    log_w_wall_max = log_w_wall.max()
    exp_w_wall = np.exp(log_w_wall - log_w_wall_max)
    w_wall = exp_w_wall / exp_w_wall.sum()
    w_wall = w_wall.reshape(N, 2)
    
    # d(soft_min)/d(pos_i[k]) from wall = -w_wall[i,k] * d(wall[i,k])/d(pos_i[k])
    # = -w_wall[i,k] * (-sign(pos_i[k]))
    # = w_wall[i,k] * sign(pos_i[k])
    
    signs = np.sign(pos)
    signs[np.abs(pos) < 1e-10] = 0.0
    grad_wall = w_wall * signs
    
    grad = grad_pair + grad_wall
    return grad


def _force_step_vec(pos, r, alpha, rep_exp=5):
    """Vectorized force-directed relaxation step."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 1.0)

    overlap_amount = np.maximum(2 * r - dist, 0.0)
    safe_dist = np.maximum(dist, 1e-10)
    mag_matrix = overlap_amount / (safe_dist ** rep_exp)
    np.fill_diagonal(mag_matrix, 0.0)

    unit_d = d / safe_dist[:, :, None]
    force = -np.sum(mag_matrix[:, :, None] * unit_d, axis=1)

    force -= 0.03 * pos

    max_move = 0.3 * r * alpha
    step = force * max_move
    step_norm = np.sqrt(np.sum(step * step, axis=1))
    for i in range(n):
        if step_norm[i] > max_move:
            step[i] *= max_move / step_norm[i]

    pos = pos + step
    lim = 0.5 - r
    pos = np.clip(pos, -lim, lim)
    return pos


def _force_layout(r, iters, alpha0, alpha1, rep_exp=5):
    """Build a non-overlapping layout for radius r via force-directed relaxation."""
    np.random.seed(12345)
    pos = np.random.uniform(-0.5 + r, 0.5 - r, size=(N, 2))
    for it in range(iters):
        alpha = alpha0 * (alpha1 / alpha0) ** (it / max(1, iters - 1))
        pos = _force_step_vec(pos, r, alpha, rep_exp)
    return pos


def _sa_incremental(pos, time_budget, r_target=None):
    """Simulated annealing with incremental distance updates."""
    t_start = time.time()
    best_pos = pos.copy()
    best_r = _score(pos)
    cur_pos = pos.copy()
    cur_r = best_r

    dist_mat = np.sqrt(np.sum((cur_pos[:, None, :] - cur_pos[None, :, :]) ** 2, axis=2))

    np.random.seed(98765)
    temp = 0.003
    cool = 0.99995
    step = 0.015
    min_step = 0.0005
    max_step = 0.05

    accept_count = 0
    total_count = 0
    window_size = 500

    while time.time() - t_start < time_budget:
        if time.time() - t_start > time_budget:
            break

        i = np.random.randint(N)
        old = cur_pos[i].copy()
        new_pos_i = old + np.random.normal(0, step, size=2)
        lim = 0.5 - 0.07
        new_pos_i = np.clip(new_pos_i, -lim, lim)

        d_new = new_pos_i[None, :] - cur_pos
        dist_new_i = np.sqrt(np.sum(d_new * d_new, axis=1))
        dist_new_i[i] = np.inf

        old_row = dist_mat[i, :].copy()
        dist_mat[i, :] = dist_new_i
        dist_mat[:, i] = dist_new_i

        wall_i = (0.5 - np.abs(new_pos_i)).min()
        min_wall_others = (0.5 - np.abs(np.delete(cur_pos, i, axis=0))).min()
        min_wall = min(wall_i, min_wall_others)

        min_pair_with_i = dist_new_i.min()
        sub = dist_mat.copy()
        sub[i, :] = np.inf
        sub[:, i] = np.inf
        min_pair_others = sub.min()
        min_pair = min(min_pair_with_i, min_pair_others)

        new_r = min(0.5 * min_pair, min_wall)

        total_count += 1
        if new_r > cur_r or np.random.rand() < np.exp((new_r - cur_r) / max(temp, 1e-9)):
            cur_r = new_r
            accept_count += 1
            cur_pos[i] = new_pos_i
            if new_r > best_r:
                best_r = new_r
                best_pos = cur_pos.copy()
        else:
            cur_pos[i] = old
            dist_mat[i, :] = old_row
            dist_mat[:, i] = old_row

        temp *= cool
        if temp < 1e-7:
            temp = 1e-7

        if total_count % window_size == 0:
            rate = accept_count / max(1, window_size)
            if rate < 0.15:
                step = max(min_step, step * 0.7)
            elif rate > 0.85:
                step = min(max_step, step * 1.2)
            accept_count = 0

    return best_pos, best_r


def _lbfgs_polish(pos, time_budget, beta=100.0):
    """Polish using L-BFGS-B on the soft-min objective."""
    t_start = time.time()
    cur_pos = pos.copy()
    cur_r = _score(cur_pos)

    def obj_and_grad(x):
        p = x.reshape(-1, 2)
        val = -_soft_min_score(p, beta)
        g = -_soft_min_grad(p, beta)
        return val, g.flatten()

    # Use a wrapper for scipy
    def obj(x):
        return obj_and_grad(x)[0]

    def grad(x):
        return obj_and_grad(x)[1]

    res = minimize(
        obj,
        cur_pos.flatten(),
        jac=grad,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (2 * N),
        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
    )

    if res.fun < -cur_r + 1e-8:
        new_pos = res.x.reshape(-1, 2)
        new_r = _score(new_pos)
        if new_r > cur_r:
            cur_pos = new_pos
            cur_r = new_r

    # Binary search for max feasible r
    lo = cur_r
    hi = cur_r * 1.05
    for _ in range(5):
        if time.time() - t_start > time_budget:
            break
        mid = (lo + hi) / 2

        def violation(x):
            p = x.reshape(-1, 2)
            d = p[:, None, :] - p[None, :, :]
            dist = np.sqrt(np.sum(d * d, axis=2))
            np.fill_diagonal(dist, 2 * mid)
            pair_viol = np.maximum(2 * mid - dist, 0.0)
            wall_viol = np.maximum(np.abs(p) - (0.5 - mid), 0.0)
            return np.sum(pair_viol ** 2) + np.sum(wall_viol ** 2)

        res2 = minimize(
            violation,
            cur_pos.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if res2.fun < 1e-12:
            new_pos = res2.x.reshape(-1, 2)
            new_r = _score(new_pos)
            if new_r >= mid - 1e-9:
                lo = mid
                cur_pos = new_pos
                cur_r = new_r
            else:
                hi = mid
        else:
            hi = mid

    return cur_pos, cur_r


def _hex_init():
    """Hexagonal lattice initialization."""
    pos = []
    r_approx = 0.07
    dx = 2 * r_approx
    dy = dx * np.sqrt(3) / 2
    for row in range(10):
        for col in range(10):
            x = (col - 4.5) * dx
            y = (row - 4.5) * dy
            if row % 2 == 1:
                x += dx / 2
            pos.append([x, y])
    pos = np.array(pos[:N])
    return pos


def _grid_init():
    """Grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_init(seed=42):
    """Random initialization."""
    np.random.seed(seed)
    return np.random.uniform(-0.43, 0.43, size=(N, 2))


def construct():
    t_start = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Multiple initializations
    inits = []
    inits.append(_grid_init())
    inits.append(_hex_init())
    inits.append(_random_init(42))
    inits.append(_random_init(123))

    best_pos = None
    best_r = -1

    for init in inits:
        if time.time() - t_start > deadline - 30:
            break
        r = _score(init)
        if r > best_r:
            best_r = r
            best_pos = init.copy()

    # Force-directed relaxation from best init
    if best_pos is not None and time.time() - t_start < deadline - 25:
        pos = _force_layout(max(best_r, 0.07), 5000, 1.0, 0.02, rep_exp=5)
        r = _score(pos)
        if r > best_r:
            best_r = r
            best_pos = pos

    # SA pass 1
    if best_pos is not None and time.time() - t_start < deadline - 15:
        budget = deadline - 15 - (time.time() - t_start)
        pos, r = _sa_incremental(best_pos, budget)
        if r > best_r:
            best_r = r
            best_pos = pos

    # SA pass 2 with perturbation
    if best_pos is not None and time.time() - t_start < deadline - 10:
        budget = deadline - 10 - (time.time() - t_start)
        np.random.seed(777)
        perturbed = best_pos + np.random.normal(0, 0.008, size=(N, 2))
        lim = 0.5 - 0.07
        perturbed = np.clip(perturbed, -lim, lim)
        pos, r = _sa_incremental(perturbed, budget)
        if r > best_r:
            best_r = r
            best_pos = pos

    # L-BFGS polish
    if best_pos is not None and time.time() - t_start < deadline - 3:
        budget = deadline - 3 - (time.time() - t_start)
        pos, r = _lbfgs_polish(best_pos, budget, beta=100.0)
        if r > best_r:
            best_r = r
            best_pos = pos

    # Final safety: ensure validity
    if best_pos is None:
        return _grid_init()
    best_pos = np.clip(best_pos, -0.5 + 0.07, 0.5 - 0.07)
    if not np.all(np.isfinite(best_pos)):
        return _grid_init()
    return best_pos
# EVOLVE-BLOCK-END

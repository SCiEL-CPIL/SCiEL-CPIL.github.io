"""Packing of 50 equal circles in a unit square via force-directed + SA + soft-min L-BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
HALF = 0.5


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


def _violation(pos, r):
    """Sum of squared violations: pairwise overlap + wall penetration."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 2 * r)
    pair_viol = np.maximum(2 * r - dist, 0.0)
    wall_viol = np.maximum(np.abs(pos) - (0.5 - r), 0.0)
    return np.sum(pair_viol ** 2) + np.sum(wall_viol ** 2)


def _soft_min_obj(pts, r_target, beta=500.0):
    """
    Smooth surrogate for the packing problem.
    We want to maximize r such that all pairwise distances >= 2*r and wall distances >= r.
    Here we minimize a soft penalty that is smooth in r_target.
    
    Objective: sum of soft-violations for pairs and walls.
    For pairs: max(0, 2*r_target - dist_i) approximated by softplus(beta*(2*r_target - dist_i))/beta
    For walls: max(0, r_target - wall_dist_i) approximated by softplus(beta*(r_target - wall_dist_i))/beta
    """
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    pair_viol = 2.0 * r_target - pair_dist
    # Softplus: log(1 + exp(x)) / beta
    # Clip to avoid overflow
    pair_viol_clipped = np.clip(pair_viol, -50, 50)
    soft_pair = np.log1p(np.exp(beta * pair_viol_clipped)) / beta
    
    # Wall distances
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r_target - wall_dist
    wall_viol_clipped = np.clip(wall_viol, -50, 50)
    soft_wall = np.log1p(np.exp(beta * wall_viol_clipped)) / beta
    
    return float(np.sum(soft_pair) + np.sum(soft_wall))


def _soft_min_grad(pts, r_target, beta=500.0):
    """
    Gradient of the soft-min objective with respect to pts.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    # Avoid division by zero
    dist_safe = np.where(dist > 1e-12, dist, 1e-12)
    
    # Gradient of softplus(beta*(2*r - dist)) w.r.t. pts
    # d/dpts [log(1+exp(beta*(2r - dist)))/beta]
    # = (1/beta) * exp(beta*(2r-dist))/(1+exp(beta*(2r-dist))) * (-beta) * d(dist)/dpts
    # = -sigmoid(beta*(2r-dist)) * d(dist)/dpts
    # where d(dist)/dpts_i = (pts_i - pts_j) / dist_ij
    
    pair_viol = 2.0 * r_target - dist
    pair_viol_clipped = np.clip(pair_viol, -50, 50)
    # sigmoid(x) = 1/(1+exp(-x))
    sig = 1.0 / (1.0 + np.exp(-beta * pair_viol_clipped))
    
    # Gradient contribution from pairs
    # For pair (i,j), d(dist_ij)/d(pts_i) = (pts_i - pts_j)/dist_ij
    # So grad_obj w.r.t. pts_i from pair (i,j) = -sig_ij * (pts_i - pts_j)/dist_ij
    # And w.r.t. pts_j = -sig_ij * (pts_j - pts_i)/dist_ij = sig_ij * (pts_i - pts_j)/dist_ij
    
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(i + 1, N):
            d = dist[i, j]
            if d < 1e-12:
                continue
            s = sig[i, j]
            diff_ij = (pts[i] - pts[j]) / d
            grad[i] -= s * diff_ij
            grad[j] += s * diff_ij
    
    # Wall gradients
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r_target - wall_dist
    wall_viol_clipped = np.clip(wall_viol, -50, 50)
    sig_w = 1.0 / (1.0 + np.exp(-beta * wall_viol_clipped))
    
    # d(wall_dist)/d(pts_i) depends on which wall is closest
    for i in range(N):
        dl = d_left[i]
        dr = d_right[i]
        db = d_bottom[i]
        dt = d_top[i]
        wmin = wall_dist[i]
        s = sig_w[i]
        if abs(dl - wmin) < 1e-10:
            grad[i, 0] -= s  # d(wall_dist)/d(x) = 1 for left wall (x + 0.5)
        if abs(dr - wmin) < 1e-10:
            grad[i, 0] += s  # d(wall_dist)/d(x) = -1 for right wall (0.5 - x)
        if abs(db - wmin) < 1e-10:
            grad[i, 1] -= s  # d(wall_dist)/d(y) = 1 for bottom wall (y + 0.5)
        if abs(dt - wmin) < 1e-10:
            grad[i, 1] += s  # d(wall_dist)/d(y) = -1 for top wall (0.5 - y)
    
    return grad


def _force_step(pos, r, alpha, rep_exp=5):
    """One force-directed relaxation step with high-exponent repulsion."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 1.0)
    # Repulsive force: proportional to 1/dist^rep_exp, applied when overlapping.
    overlap = (dist < 2 * r)
    force = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if overlap[i, j]:
                gap = dist[i, j]
                if gap < 1e-12:
                    continue
                mag = (2 * r - gap) / (gap ** rep_exp)
                diff = d[i, j]
                force[i] -= mag * diff
                force[j] += mag * diff
    # Attractive force toward center (weak).
    force -= 0.03 * pos
    # Adaptive step size: limit movement to avoid oscillation.
    max_move = 0.3 * r * alpha
    step = force * max_move
    step_norm = np.sqrt(np.sum(step * step, axis=1))
    for i in range(n):
        if step_norm[i] > max_move:
            step[i] *= max_move / step_norm[i]
    pos = pos + step
    # Clamp inside the square (leaving radius margin).
    lim = 0.5 - r
    pos = np.clip(pos, -lim, lim)
    return pos


def _force_layout(r, iters, alpha0, alpha1, rep_exp=5):
    """Build a non-overlapping layout for radius r via force-directed relaxation."""
    np.random.seed(12345)
    pos = np.random.uniform(-0.5 + r, 0.5 - r, size=(N, 2))
    for it in range(iters):
        alpha = alpha0 * (alpha1 / alpha0) ** (it / max(1, iters - 1))
        pos = _force_step(pos, r, alpha, rep_exp)
    return pos


def _anneal(pos, r, time_budget):
    """Simulated annealing with adaptive step size to maximize minimum radius."""
    best_pos = pos.copy()
    best_r = _score(pos)
    cur_pos = pos.copy()
    cur_r = best_r
    np.random.seed(98765)
    t_start = time.time()
    temp = 0.003
    cool = 0.9998
    step = 0.015
    min_step = 0.001
    accept_count = 0
    total_count = 0
    while time.time() - t_start < time_budget:
        i = np.random.randint(N)
        old = cur_pos[i].copy()
        cur_pos[i] = old + np.random.normal(0, step, size=2)
        lim = 0.5 - 0.07
        cur_pos[i] = np.clip(cur_pos[i], -lim, lim)
        new_r = _score(cur_pos)
        total_count += 1
        if new_r > cur_r or np.random.rand() < np.exp((new_r - cur_r) / max(temp, 1e-9)):
            cur_r = new_r
            accept_count += 1
            if new_r > best_r:
                best_r = new_r
                best_pos = cur_pos.copy()
        else:
            cur_pos[i] = old
        temp *= cool
        # Adapt step size based on acceptance rate
        if total_count % 200 == 0:
            rate = accept_count / max(1, total_count)
            if rate < 0.2:
                step = max(min_step, step * 0.7)
            elif rate > 0.8:
                step = min(0.05, step * 1.3)
            accept_count = 0
            total_count = 0
    return best_pos, best_r


def _lbfgs_polish(pts0, r_target, deadline, beta=500.0, maxiter=500):
    """Use L-BFGS-B to minimize the soft-min objective for target radius."""
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _soft_min_obj(p, r_target, beta)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        g = _soft_min_grad(p, r_target, beta)
        return g.ravel()
    
    x0 = pts0.ravel()
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-10})
    except RuntimeError:
        return pts0, float('inf')
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts, float(res.fun)


def _polish(pos, time_budget):
    """High-precision polish using scipy minimize on violation surrogate."""
    t_start = time.time()
    cur_pos = pos.copy()
    cur_r = _score(cur_pos)
    
    # Binary search for maximum feasible r with polish
    lo = cur_r
    hi = cur_r * 1.05
    for _ in range(4):
        if time.time() - t_start > time_budget:
            break
        mid = (lo + hi) / 2
        deadline = t_start + time_budget
        new_pos, obj_val = _lbfgs_polish(cur_pos, mid, deadline, beta=500.0, maxiter=300)
        new_r = _score(new_pos)
        if new_r >= mid - 1e-9:
            lo = mid
            cur_pos = new_pos
            cur_r = new_r
        else:
            hi = mid
    
    return cur_pos, cur_r


def construct():
    t_start = time.time()
    deadline = 115.0
    
    # Baseline grid.
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_pos = grid[:N].copy()
    best_r = _score(best_pos)
    
    # Force-directed initialization with high-exponent repulsion.
    if time.time() - t_start < deadline - 30:
        pos = _force_layout(best_r, 6000, 1.0, 0.03, rep_exp=5)
        r = _score(pos)
        if r > best_r:
            best_r, best_pos = r, pos
    
    # Simulated annealing to push the radius higher.
    if time.time() - t_start < deadline - 15:
        budget = deadline - 15 - (time.time() - t_start)
        pos, r = _anneal(best_pos, best_r, budget)
        if r > best_r:
            best_r, best_pos = r, pos
    
    # Final high-precision polish pass using soft-min L-BFGS.
    if time.time() - t_start < deadline - 2:
        budget = deadline - 2 - (time.time() - t_start)
        pos, r = _polish(best_pos, budget)
        if r > best_r:
            best_r, best_pos = r, pos
    
    # Final safety: ensure validity (finite, inside square).
    best_pos = np.clip(best_pos, -0.5 + 0.07, 0.5 - 0.07)
    if not np.all(np.isfinite(best_pos)):
        return grid[:N]
    return best_pos

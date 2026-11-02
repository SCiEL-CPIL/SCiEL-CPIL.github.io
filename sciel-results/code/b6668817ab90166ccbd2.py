"""Packing of 50 equal circles in a unit square via SA + log-sum-exp L-BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50
HALF = 0.5
DEADLINE = 115.0  # Safety margin before 120s kill

# Precompute upper-triangle indices
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])


def _score(pos):
    """Largest common radius for a placement."""
    if pos.shape != (N, 2) or not np.all(np.isfinite(pos)):
        return 0.0
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    min_wall = (0.5 - np.abs(pos)).min()
    return min(0.5 * min_pair, min_wall)


def _soft_min_obj_grad(x, tau):
    """
    Objective: minimize -soft_min(all constraint distances) using log-sum-exp.
    Returns (f, g).
    """
    centers = x.reshape(-1, 2)

    # Wall distances per circle
    wall_dists_x = HALF - np.abs(centers[:, 0])
    wall_dists_y = HALF - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)

    # Pairwise distances (upper triangle)
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_dists = dists / 2.0

    all_dists = np.concatenate([walls, pair_dists])

    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    soft_min = min_d - tau * np.log(sum_exp)

    f = -soft_min

    grad = np.zeros_like(x)

    # Wall gradient
    for i in range(N):
        w = weights[i]
        if w < 1e-12:
            continue
        if wall_dists_x[i] < wall_dists_y[i]:
            grad[2 * i] += w * np.sign(centers[i, 0])
        elif wall_dists_y[i] < wall_dists_x[i]:
            grad[2 * i + 1] += w * np.sign(centers[i, 1])
        else:
            grad[2 * i] += 0.5 * w * np.sign(centers[i, 0])
            grad[2 * i + 1] += 0.5 * w * np.sign(centers[i, 1])

    # Pair gradient (vectorized)
    pair_weights = weights[N:]
    dists_safe = np.maximum(dists, 1e-12)
    factor = pair_weights / dists_safe
    contrib = factor[:, None] * diff

    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])

    return f, grad


def _initial_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    return grid


def _hex_init():
    best_grid = None
    best_r = 0
    rng = np.random.default_rng(42)

    for attempt in range(12):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))
        s_width = 1.0 / (cols + 0.3)
        s_height = 1.0 / (rows * 0.866 + 0.3)
        s = min(s_width, s_height)
        centers = []
        for j in range(rows):
            for i in range(cols):
                x = (i - (cols - 1) / 2) * s
                y = (j - (rows - 1) / 2) * s * 0.866
                if j % 2 == 1:
                    x += s / 2
                centers.append([x, y])
        centers = np.array(centers[:N])
        if len(centers) < N:
            while len(centers) < N:
                centers = np.vstack([centers, rng.uniform(-0.45, 0.45, 2)])
        centers = np.clip(centers, -0.49, 0.49)
        r = _score(centers)
        if r > best_r:
            best_r = r
            best_grid = centers.copy()

    return best_grid if best_grid is not None else _initial_grid()


def _anneal(pos, time_budget):
    """Simulated annealing to maximize minimum radius."""
    best_pos = pos.copy()
    best_r = _score(pos)
    cur_pos = pos.copy()
    cur_r = best_r
    rng = np.random.default_rng(98765)
    t_start = time.time()
    temp = 0.003
    cool = 0.9998
    step = 0.015
    min_step = 0.001
    accept_count = 0
    total_count = 0
    while time.time() - t_start < time_budget:
        i = rng.integers(N)
        old = cur_pos[i].copy()
        cur_pos[i] = old + rng.normal(0, step, size=2)
        lim = 0.5 - 0.07
        cur_pos[i] = np.clip(cur_pos[i], -lim, lim)
        new_r = _score(cur_pos)
        total_count += 1
        if new_r > cur_r or rng.random() < np.exp((new_r - cur_r) / max(temp, 1e-9)):
            cur_r = new_r
            accept_count += 1
            if new_r > best_r:
                best_r = new_r
                best_pos = cur_pos.copy()
        else:
            cur_pos[i] = old
        temp *= cool
        if total_count % 200 == 0:
            rate = accept_count / max(1, total_count)
            if rate < 0.2:
                step = max(min_step, step * 0.7)
            elif rate > 0.8:
                step = min(0.05, step * 1.3)
            accept_count = 0
            total_count = 0
    return best_pos, best_r


def _lbfgs_polish(pts0, tau, deadline, maxiter=500):
    """Use L-BFGS-B to minimize the soft-min objective."""
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        f, _ = _soft_min_obj_grad(x, tau)
        return f

    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        _, g = _soft_min_obj_grad(x, tau)
        return g

    x0 = pts0.ravel()
    try:
        res = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-10})
    except RuntimeError:
        return pts0
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts


def _polish(pos, time_budget):
    """High-precision polish using log-sum-exp L-BFGS with progressive tau."""
    t_start = time.time()
    cur_pos = pos.copy()
    cur_r = _score(cur_pos)

    # Progressive tau values from loose to tight
    taus = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    for tau in taus:
        if time.time() - t_start > time_budget:
            break
        # Binary search for max feasible r at this tau
        lo = cur_r
        hi = cur_r * 1.03
        for _ in range(3):
            if time.time() - t_start > time_budget:
                break
            mid = (lo + hi) / 2
            new_pos = _lbfgs_polish(cur_pos, tau, time.time() + 5.0, maxiter=200)
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
    deadline = DEADLINE

    # Multiple initializations
    candidates = []
    candidates.append(_initial_grid())
    candidates.append(_hex_init())

    # Pick best initial
    best_pos = candidates[0]
    best_r = _score(best_pos)
    for c in candidates[1:]:
        r = _score(c)
        if r > best_r:
            best_r = r
            best_pos = c

    # Simulated annealing
    if time.time() - t_start < deadline - 40:
        budget = deadline - 40 - (time.time() - t_start)
        pos, r = _anneal(best_pos, budget)
        if r > best_r:
            best_r, best_pos = r, pos

    # L-BFGS polish with progressive tau
    if time.time() - t_start < deadline - 5:
        budget = deadline - 5 - (time.time() - t_start)
        pos, r = _polish(best_pos, budget)
        if r > best_r:
            best_r, best_pos = r, pos

    # Final safety
    best_pos = np.clip(best_pos, -0.5 + 0.07, 0.5 - 0.07)
    if not np.all(np.isfinite(best_pos)):
        return candidates[0]
    return best_pos

import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5


def _score(c):
    """Largest common radius for centres c (N,2)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_bound = np.min(HALF - np.abs(c))
    return min(min_pair, min_bound)


def _feasible(pts, r, max_iter=1500, deadline=None):
    """
    Try to find a configuration of N circles of radius r inside the square.
    Uses projected gradient descent: push overlapping circles apart and keep
    them within bounds. Returns (success, pts).
    """
    pts = np.asarray(pts, dtype=np.float64).copy()
    n = len(pts)
    lo = -HALF + r
    hi = HALF - r
    # If the required box is non-positive, infeasible
    if hi < lo:
        return False, pts

    step = 0.1
    prev_min = -np.inf
    stall = 0

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        diff = pts[:, None, :] - pts[None, :, :]          # (n,n,2)
        dist = np.sqrt(np.sum(diff * diff, axis=2))       # (n,n)
        np.fill_diagonal(dist, np.inf)

        # Pairwise repulsion for overlapping pairs (dist < 2r)
        overlap = (dist < 2.0 * r) & (dist > 1e-12)
        force_mag = np.where(overlap, (2.0 * r - dist), 0.0)
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        force = diff * (force_mag * inv_dist)[:, :, None]
        grad = np.sum(force, axis=1)

        # Wall forces
        for k in range(2):
            too_right = pts[:, k] > hi
            too_left = pts[:, k] < lo
            grad[too_right, k] -= (pts[too_right, k] - hi) * 5.0
            grad[too_left, k] += (pts[too_left, k] - lo) * 5.0

        new_pts = pts + step * grad
        new_pts = np.clip(new_pts, lo, hi)

        # Evaluate current minimum clearance
        d2 = new_pts[:, None, :] - new_pts[None, :, :]
        dist2 = np.sqrt(np.sum(d2 * d2, axis=2))
        np.fill_diagonal(dist2, np.inf)
        min_pair = np.min(dist2) / 2.0
        min_wall = np.min(HALF - np.abs(new_pts))
        cur_min = min(min_pair, min_wall)

        if cur_min >= r - 1e-9:
            return True, new_pts

        pts = new_pts

        # Adaptive step: shrink if stagnating
        if cur_min > prev_min + 1e-10:
            stall = 0
            step = min(step * 1.05, 0.3)
        else:
            stall += 1
            if stall > 30:
                step *= 0.7
                stall = 0

        prev_min = cur_min

        if it > 200 and step < 1e-5:
            break

    # Final check
    d2 = pts[:, None, :] - pts[None, :, :]
    dist2 = np.sqrt(np.sum(d2 * d2, axis=2))
    np.fill_diagonal(dist2, np.inf)
    min_pair = np.min(dist2) / 2.0
    min_wall = np.min(HALF - np.abs(pts))
    success = (min_pair >= r - 1e-9) and (min_wall >= r - 1e-9)
    return success, pts


def _soft_min_obj(points_flat, beta):
    """
    Smooth surrogate for maximizing min distance.
    -softmin(x) = (1/beta) * log(sum(exp(-beta * x_i)))
    Minimizing this pushes all x_i to be large and equal.
    """
    pts = points_flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0

    wall_dists = np.concatenate([
        0.5 - np.abs(pts[:, 0]),
        0.5 - np.abs(pts[:, 1])
    ])

    all_dists = np.concatenate([pair_dists, wall_dists])
    min_d = np.min(all_dists)
    if min_d < -100:
        return 1e10
    shifted = all_dists - min_d
    log_sum_exp = min_d + (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
    return -log_sum_exp


def _polish_lbfgs(points, deadline, betas=(50, 200, 500, 1000, 2000)):
    """Polish a configuration using L-BFGS-B on the soft-min objective."""
    if time.time() > deadline - 5:
        return points, _score(points)

    best_points = np.asarray(points, dtype=np.float64).copy()
    best_score = _score(best_points)

    for beta in betas:
        if time.time() > deadline - 5:
            break

        def obj(x):
            return _soft_min_obj(x, beta)

        try:
            res = minimize(obj, best_points.flatten(), method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10})
            cand = res.x.reshape(N, 2)
            new_score = _score(cand)
            if new_score > best_score:
                best_score = new_score
                best_points = cand.copy()
        except Exception:
            pass

    return best_points, best_score


def _hex_seed(r):
    """Generate a hexagonal lattice seed with spacing 2r."""
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    while y <= HALF - r + 1e-9:
        x = -HALF + r + (0.5 * dx if (row % 2) else 0.0)
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    pts = np.array(list(zip(xs, ys)), dtype=np.float64)
    if len(pts) >= N:
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks], dtype=np.float64)
        pts = np.vstack([pts, grid[:N - len(pts)]])
    return np.clip(pts, -HALF + r, HALF - r)


def _grid_seed(r):
    """Generate a grid seed."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks], dtype=np.float64)
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def _random_seed(r):
    return np.random.uniform(-HALF + r, HALF - r, size=(N, 2))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Initial best from a reasonable hex seed
    r0 = 0.071
    best_pts = _hex_seed(r0).copy()
    best_r = _score(best_pts)

    # Binary search on radius using feasibility
    r_low = 0.06
    r_high = 0.085

    # First, try to establish a solid lower bound
    for trial in [0.070, 0.071, 0.0713, 0.0714]:
        if time.time() > deadline - 20:
            break
        ok, pts = _feasible(_hex_seed(trial), trial, max_iter=800, deadline=deadline - 20)
        if ok:
            r_low = max(r_low, trial)
            s = _score(pts)
            if s > best_r:
                best_r = s
                best_pts = pts.copy()

    # Binary search
    for it in range(20):
        if time.time() > deadline - 25:
            break
        r_mid = (r_low + r_high) / 2.0

        success = False
        trial_pts = None

        seeds = [_hex_seed(r_mid), _grid_seed(r_mid),
                 np.clip(_hex_seed(r_mid) + np.random.randn(N, 2) * 0.005, -HALF + r_mid, HALF - r_mid)]

        for seed in seeds:
            if time.time() > deadline - 25:
                break
            ok, pts = _feasible(seed, r_mid, max_iter=1000, deadline=deadline - 25)
            if ok:
                success = True
                trial_pts = pts
                break

        if success:
            r_low = r_mid
            s = _score(trial_pts)
            if s > best_r:
                best_r = s
                best_pts = trial_pts.copy()
        else:
            r_high = r_mid

    # Polish the best configuration with L-BFGS soft-min
    if time.time() < deadline - 5:
        polished_pts, polished_score = _polish_lbfgs(best_pts, deadline)
        if polished_score > best_r:
            best_r = polished_score
            best_pts = polished_pts.copy()

    # Final validity check
    best_pts = np.clip(np.asarray(best_pts, dtype=np.float64), -HALF, HALF)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_seed(0.071)

    return best_pts

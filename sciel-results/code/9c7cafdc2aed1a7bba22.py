"""Fresh design: differentiable multi-start radius maximizer with deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 120.0
START = time.perf_counter()


def _time_left():
    return max(0.0, DEADLINE - (time.perf_counter() - START))


def _pairwise(pos):
    """(n, n) squared distances and (n, n) displacement vectors (pos[i]-pos[j])."""
    d = pos[:, None, :] - pos[None, :, :]
    sq = (d * d).sum(-1)
    return sq, d


def _score(pos):
    sq, _ = _pairwise(pos)
    np.fill_diagonal(sq, np.inf)
    min_pair = np.sqrt(sq.min())
    wall = 1.0 - np.linalg.norm(pos, axis=1).max()
    return 0.5 * min(min_pair, 2.0 * wall)


def _valid(pos):
    if pos is None or pos.shape != (N, 2):
        return False
    if not np.all(np.isfinite(pos)):
        return False
    sq, _ = _pairwise(pos)
    np.fill_diagonal(sq, 0.0)
    if np.sqrt(sq.min()) <= 1e-9:          # coincident centres
        return False
    if np.linalg.norm(pos, axis=1).max() > 1.0 + 1e-9:  # outside container
        return False
    return True


def _repair(pos, iters=150):
    """Resolve hard overlaps / wall violations so the candidate is valid."""
    pos = np.array(pos, dtype=float, copy=True)
    for _ in range(iters):
        sq, d = _pairwise(pos)
        np.fill_diagonal(sq, np.inf)
        i, j = np.unravel_index(np.argmin(sq), sq.shape)
        dist = np.sqrt(sq[i, j])
        r = np.linalg.norm(pos, axis=1)
        wall = 1.0 - r
        # smallest clearance: a pair or a wall
        if dist < 2.0 * max(wall, 1e-9) or dist < 1e-9:
            if dist < 1e-9:
                ang = np.random.uniform(0, 2 * np.pi)
                pos[i] += 1e-4 * np.array([np.cos(ang), np.sin(ang)])
                continue
            n = d[i, j] / dist
            shift = 0.5 * (2.0 * max(wall, 1e-9) - dist)
            pos[i] += 0.5 * shift * n
            pos[j] -= 0.5 * shift * n
        else:
            k = int(np.argmax(r))
            if r[k] > 1.0:
                pos[k] *= (1.0 - 1e-6) / r[k]
            else:
                break
        if r.max() <= 1.0 and dist >= 1e-9:
            break
    return pos


def _objective(p, r, beta):
    """Smooth objective: -r + penalty for the few binding constraints."""
    pos = p[:2 * N].reshape(N, 2)
    sq, d = _pairwise(pos)
    np.fill_diagonal(sq, np.inf)
    rad = np.linalg.norm(pos, axis=1)
    # overlap slack: want dist >= 2r  ->  slack = dist - 2r (positive = ok)
    s_pair = np.sqrt(sq) - 2.0 * r
    s_wall = 1.0 - r - rad
    # combine into one slack tensor: use min over both per relevant index
    # penalty focuses on smallest slacks (most violated) via softmin
    slacks = np.concatenate([s_pair.ravel(), s_wall.ravel()])
    m = slacks.min()
    w = np.exp(-beta * (slacks - m))
    w = w / w.sum()
    pen = (w * slacks).sum()
    return -r + 0.5 * pen


def _objective_grad(p, r, beta):
    pos = p[:2 * N].reshape(N, 2)
    sq, d = _pairwise(pos)
    np.fill_diagonal(sq, np.inf)
    rad = np.linalg.norm(pos, axis=1)
    s_pair = np.sqrt(sq) - 2.0 * r
    s_wall = 1.0 - r - rad
    slacks = np.concatenate([s_pair.ravel(), s_wall.ravel()])
    m = slacks.min()
    w = np.exp(-beta * (slacks - m))
    w = w / w.sum()
    # d pen / d pos
    g = np.zeros((N, 2))
    w_pair = w[:N * N].reshape(N, N)
    w_wall = w[N * N:]
    # pair contribution: d s_pair_ij / d pos_i = (pos_i - pos_j)/dist
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dist = np.sqrt(max(sq[i, j], 1e-12))
            g[i] += w_pair[i, j] * 0.5 * d[i, j] / dist
            g[j] += w_pair[i, j] * 0.5 * (-d[i, j]) / dist
    # wall contribution: d s_wall_i / d pos_i = -pos_i/|pos_i|
    for i in range(N):
        ri = max(rad[i], 1e-12)
        g[i] += w_wall[i] * 0.5 * (-pos[i] / ri)
    grad = np.zeros(2 * N + 1)
    grad[:2 * N] = g.ravel()
    # d pen / d r: -2 (pair) + 1 (wall)
    grad[2 * N] = 0.5 * (-2.0 * w_pair.sum() + 1.0 * w_wall.sum())
    # total objective grad: -1 for r term
    grad[2 * N] += -1.0
    return grad


def _optimize_from(pos, r0):
    """Local differentiable optimization starting from a feasible packing."""
    pos = np.array(pos, dtype=float, copy=True)
    r = _score(pos)
    if r <= 0:
        r = max(r0, 1e-3)
    best_pos, best_r = pos.copy(), r
    beta = 4.0
    for it in range(120):
        if _time_left() < 3.0:
            break
        x0 = np.concatenate([pos.ravel(), [r]])
        res = minimize(_objective, x0, args=(r, beta), jac=_objective_grad,
                       method='L-BFGS-B',
                       bounds=[(0, 1) for _ in range(2 * N)] + [(1e-4, 0.9)],
                       options={'maxiter': 40, 'maxls': 20, 'ftol': 1e-9, 'gtol': 1e-7})
        p = res.x
        new_pos = p[:2 * N].reshape(N, 2)
        new_r = float(p[2 * N])
        # accept if it improves the *true* score and is repairable
        cand = _repair(new_pos, iters=80)
        cs = _score(cand)
        if cs > best_r + 1e-12:
            best_pos, best_r = cand, cs
            r = cs
        else:
            # keep r growing slowly to escape
            r = min(0.9, r * 1.02 + 1e-4)
        beta = min(40.0, beta * 1.15)
    return best_pos, best_r


def _init_rings():
    """Two/three concentric rings, evenly spaced."""
    pts = [np.array([0.0, 0.0])]
    remaining = N - 1
    rings = [13, 13, 13, 11]  # candidate ring sizes
    radii = [0.33, 0.62, 0.86, 0.985]
    idx = 0
    for size, rr in zip(rings, radii):
        size = min(size, remaining)
        if size <= 0:
            break
        ang = np.linspace(0, 2 * np.pi, size, endpoint=False)
        pts.append(rr * np.column_stack([np.cos(ang), np.sin(ang)]))
        remaining -= size
        idx += 1
    pts = np.vstack(pts)
    if len(pts) < N:
        ang = np.linspace(0, 2 * np.pi, N - len(pts), endpoint=False)
        pts = np.vstack([pts, 0.5 * np.column_stack([np.cos(ang), np.sin(ang)])])
    return pts[:N]


def _init_hex():
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def _init_random():
    ang = np.random.uniform(0, 2 * np.pi, N)
    rad = np.sqrt(np.random.uniform(0, 1, N)) * 0.95
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])


def _init_mixed():
    base = _init_rings()
    base = base + 0.03 * np.random.randn(N, 2)
    return base


def construct():
    global START
    START = time.perf_counter()
    rng = np.random.default_rng(12345)
    np.random.seed(12345)
    initializers = [_init_rings, _init_hex, _init_mixed, _init_random,
                    _init_rings, _init_hex, _init_random, _init_mixed]

    best_pos, best_r = None, -1.0
    for init_fn in initializers:
        if _time_left() < 12.0:
            break
        try:
            pos = init_fn()
        except Exception:
            continue
        pos = _repair(pos, iters=200)
        r = _score(pos)
        if r > best_r:
            best_pos, best_r = pos.copy(), r
        # optimize
        try:
            op, orr = _optimize_from(pos, r)
        except Exception:
            op, orr = pos, r
        if orr > best_r:
            best_pos, best_r = op.copy(), orr

    # final safety repair + validation
    if best_pos is None:
        best_pos = _init_hex()
    best_pos = _repair(best_pos, iters=300)
    if not _valid(best_pos):
        best_pos = _init_hex()
        best_pos = _repair(best_pos, iters=300)
    return best_pos

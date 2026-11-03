"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# Fixed per-run wall-clock budget with safety margin (evaluator kills at 150s).
_RUN_DEADLINE = 142.0


def _penalty(pos, r, w=1e4, margin=1e-6):
    """Penalty for violations: circle-circle overlap and wall contact."""
    n = len(r)
    # wall penalties
    wall = np.maximum(0.0, r + margin - pos[:, 0]) + \
           np.maximum(0.0, r + margin - (1.0 - pos[:, 1])) + \
           np.maximum(0.0, r + margin - pos[:, 1]) + \
           np.maximum(0.0, r + margin - (1.0 - pos[:, 0]))
    # circle-circle
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    need = r[:, None] + r[None, :] + margin
    over = np.maximum(0.0, need - d)
    np.fill_diagonal(over, 0.0)
    return w * (np.sum(wall) + np.sum(over))


def _is_valid(pos, r, tol=1e-5):
    n = len(r)
    if np.any(r < -tol) or np.any(r > 0.5 + tol):
        return False
    if np.any(pos < -tol) or np.any(pos > 1.0 + tol) or np.any(pos < -tol) or np.any(pos > 1.0 + tol):
        return False
    # walls
    if np.any(pos[:, 0] - r < -tol) or np.any(1.0 - pos[:, 0] - r < -tol) or \
       np.any(pos[:, 1] - r < -tol) or np.any(1.0 - pos[:, 1] - r < -tol):
        return False
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    if np.any(d < (r[:, None] + r[None, :]) - tol):
        return False
    return True


def _optimize(x0, deadline):
    """Run one L-BFGS-B optimization, return (pos, r, obj, valid)."""
    n = N
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))

    def obj(v):
        pos = v[:2 * n].reshape(n, 2)
        r = v[2 * n:]
        return -np.sum(r) + _penalty(pos, r)

    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-8})
    v = res.x
    pos = v[:2 * n].reshape(n, 2)
    r = v[2 * n:]
    objval = obj(v)
    valid = _is_valid(pos, r)
    return pos, r, objval, valid


def _grid_seed(k):
    """Grid-like seed with jitter, k = jitter scale."""
    side = int(np.ceil(np.sqrt(N)))
    rng = np.random.default_rng(k)
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cells.append((cx, cy))
    pos = np.array(cells[:N], dtype=float)
    r0 = 0.5 / side - 1e-3
    r = np.full(N, r0)
    # jitter
    if k > 0:
        pos = pos + rng.normal(0.0, k, size=pos.shape)
        pos = np.clip(pos, 0.0, 1.0)
    return pos, r


def _random_seed(rng):
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    r = np.full(N, 0.06)
    return pos, r


def construct():
    t_start = time.time()
    rng = np.random.default_rng(12345)

    best = None  # (sum_r, pos, r)

    def consider(pos, r):
        nonlocal best
        if not _is_valid(pos, r):
            return
        s = float(np.sum(r))
        if best is None or s > best[0]:
            best = (s, pos.copy(), r.copy())

    # Seed 0: clean grid, then optimize
    pos, r = _grid_seed(0.0)
    x0 = np.concatenate([pos.ravel(), r])
    pos, r, _, valid = _optimize(x0, _RUN_DEADLINE)
    consider(pos, r)
    if time.time() > _RUN_DEADLINE - 20:
        return _pack(best)

    # Seed 1: jittered grid, optimize
    pos, r = _grid_seed(0.02)
    x0 = np.concatenate([pos.ravel(), r])
    pos, r, _, valid = _optimize(x0, _RUN_DEADLINE)
    consider(pos, r)
    if time.time() > _RUN_DEADLINE - 20:
        return _pack(best)

    # Random restarts
    for k in range(30):
        if time.time() > _RUN_DEADLINE - 15:
            break
        pos, r = _random_seed(rng)
        x0 = np.concatenate([pos.ravel(), r])
        pos, r, _, valid = _optimize(x0, _RUN_DEADLINE)
        consider(pos, r)

    # Local refinement of best: re-optimize from best
    if best is not None:
        s, pos, r = best
        x0 = np.concatenate([pos.ravel(), r])
        pos, r, _, valid = _optimize(x0, _RUN_DEADLINE)
        consider(pos, r)

    if best is None:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return _pack(best)


def _pack(best):
    s, pos, r = best
    out = np.empty((N, 3))
    out[:, 0] = pos[:, 0]
    out[:, 1] = pos[:, 1]
    out[:, 2] = r
    return out

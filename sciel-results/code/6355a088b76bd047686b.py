"""Physics-based multistart circle packing in perimeter-4 rectangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
T_LIMIT = 110.0  # safety margin under 120s
T_START = time.time()


def _elapsed():
    return time.time() - T_START


def _penalty_objective(x, W, H, lam_wall, lam_overlap):
    """x: [x0,y0,r0, x1,y1,r1, ...] length 3N. Returns scalar penalty."""
    n = len(x) // 3
    pos = x.reshape(n, 3)
    px, py, pr = pos[:, 0], pos[:, 1], pos[:, 2]

    # Wall violations: r_i > distance to nearest wall
    d_left = px
    d_right = W - px
    d_bottom = py
    d_top = H - py
    wall = np.maximum(0.0, pr - d_left) + np.maximum(0.0, pr - d_right) + \
           np.maximum(0.0, pr - d_bottom) + np.maximum(0.0, pr - d_top)

    # Overlap violations
    dx = px[:, None] - px[None, :]
    dy = py[:, None] - py[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    req = pr[:, None] + pr[None, :]
    overlap = np.maximum(0.0, req - dist)
    # count each pair once (upper triangle)
    iu = np.triu_indices(n, k=1)
    ov = np.maximum(0.0, req[iu] - dist[iu])
    overlap_pen = np.sum(ov * ov)

    return lam_wall * np.sum(wall * wall) + lam_overlap * overlap_pen


def _is_valid(x, W, H, tol=1e-6):
    n = len(x) // 3
    pos = x.reshape(n, 3)
    px, py, pr = pos[:, 0], pos[:, 1], pos[:, 2]
    if np.any(pr < -tol):
        return False
    if np.any(px < -tol) or np.any(px > W + tol) or np.any(py < -tol) or np.any(py > H + tol):
        return False
    # wall
    if np.any(pr - px > tol) or np.any(pr - (W - px) > tol) or \
       np.any(pr - py > tol) or np.any(pr - (H - py) > tol):
        return False
    dx = px[:, None] - px[None, :]
    dy = py[:, None] - py[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    req = pr[:, None] + pr[None, :]
    if np.any(req - dist > tol):
        return False
    return True


def _clip_into(x, W, H, eps=1e-4):
    """Clamp positions/radii so they are (approximately) inside, used as a repair."""
    n = len(x) // 3
    pos = x.reshape(n, 3).copy()
    px, py, pr = pos[:, 0], pos[:, 1], pos[:, 2]
    pr = np.maximum(pr, 1e-6)
    # clamp radii to fit in the rectangle
    pr = np.minimum(pr, W / 2.0 - eps)
    pr = np.minimum(pr, H / 2.0 - eps)
    px = np.clip(px, pr + eps, W - pr - eps)
    py = np.clip(py, pr + eps, H - pr - eps)
    pos[:, 0] = px
    pos[:, 1] = py
    pos[:, 2] = pr
    return pos.ravel()


def _init_random(W, H, rng, corner_bias):
    """Random initialization with optional corner bias."""
    n = N
    if corner_bias:
        # place some circles near corners, rest random
        corners = np.array([[0.0, 0.0], [W, 0.0], [0.0, H], [W, H]])
        base_r = min(W, H) / 6.0
        px = np.empty(n)
        py = np.empty(n)
        pr = np.empty(n)
        # first 4 near corners
        for k in range(4):
            cx, cy = corners[k]
            off = rng.uniform(0.0, base_r)
            sx = 1.0 if cx == 0.0 else -1.0
            sy = 1.0 if cy == 0.0 else -1.0
            px[k] = cx + sx * off
            py[k] = cy + sy * off
            pr[k] = base_r * rng.uniform(0.8, 1.0)
        # rest random
        px[4:] = rng.uniform(0.0, W, n - 4)
        py[4:] = rng.uniform(0.0, H, n - 4)
        pr[4:] = base_r * rng.uniform(0.5, 1.0, n - 4)
    else:
        base_r = min(W, H) / 6.0
        px = rng.uniform(0.0, W, n)
        py = rng.uniform(0.0, H, n)
        pr = base_r * rng.uniform(0.5, 1.0, n)
    x = np.empty(3 * n)
    x[0::3] = px
    x[1::3] = py
    x[2::3] = pr
    return x


def _init_grid(W, H, rng, jitter):
    """Grid-like initialization with jitter."""
    cols = int(np.ceil(np.sqrt(N * W / H))) if W >= H else int(np.ceil(np.sqrt(N * H / W)))
    cols = max(1, min(cols, N))
    rows = int(np.ceil(N / cols))
    px = np.empty(N)
    py = np.empty(N)
    pr = np.empty(N)
    idx = 0
    for i in range(cols):
        for j in range(rows):
            if idx >= N:
                break
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            px[idx] = cx + rng.uniform(-jitter, jitter)
            py[idx] = cy + rng.uniform(-jitter, jitter)
            pr[idx] = min(W / cols, H / rows) * 0.48 * rng.uniform(0.8, 1.0)
            idx += 1
    x = np.empty(3 * N)
    x[0::3] = px
    x[1::3] = py
    x[2::3] = pr
    return x


def _solve_one(x0, W, H, rng):
    """Run a single local optimization from x0. Returns (x, valid)."""
    # bounds
    n = N
    lb = np.empty(3 * n)
    ub = np.empty(3 * n)
    for i in range(n):
        lb[3 * i] = 0.0
        ub[3 * i] = W
        lb[3 * i + 1] = 0.0
        ub[3 * i + 1] = H
        lb[3 * i + 2] = 1e-6
        ub[3 * i + 2] = min(W, H) / 2.0 - 1e-6

    best_x = x0.copy()
    best_obj = None

    for (lam_wall, lam_overlap, method) in [
        (1.0, 5.0, 'L-BFGS-B'),
        (10.0, 50.0, 'L-BFGS-B'),
        (1.0, 5.0, 'SLSQP'),
    ]:
        if _elapsed() > T_LIMIT - 5:
            break
        fun = lambda x: _penalty_objective(x, W, H, lam_wall, lam_overlap)
        try:
            res = minimize(fun, x0, method=method, bounds=list(zip(lb, ub)),
                           options={'maxiter': 400, 'ftol': 1e-12})
            if res.success or res.fun < (best_obj if best_obj is not None else np.inf):
                # repair
                xr = _clip_into(res.x, W, H)
                # re-optimize from repaired point lightly
                res2 = minimize(fun, xr, method='L-BFGS-B',
                                bounds=list(zip(lb, ub)),
                                options={'maxiter': 200, 'ftol': 1e-12})
                cand = _clip_into(res2.x if res2.fun < fun(xr) else xr, W, H)
                # objective for comparison: penalized value
                obj = _penalty_objective(cand, W, H, 1.0, 5.0)
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_x = cand
        except Exception:
            continue
    return best_x


def construct():
    global T_START
    T_START = time.time()
    rng = np.random.default_rng(20240501)

    best_x = None
    best_score = -np.inf

    # baseline grid (the seed) for safety
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    base = np.array(cells[:N])
    # put base in a W=1,H=1 rectangle (perimeter 4)
    if _is_valid(base.ravel(), 1.0, 1.0, tol=1e-4):
        best_x = base.ravel()
        best_score = np.sum(base[:, 2])

    # multistart
    restarts = 0
    max_restarts = 60
    while restarts < max_restarts and _elapsed() < T_LIMIT - 8:
        restarts += 1
        # random aspect ratio
        ar = rng.uniform(0.5, 2.0)  # W/H
        # W + H = 2
        H = 2.0 / (1.0 + ar)
        W = 2.0 - H

        # choose init type
        r = rng.random()
        if r < 0.35:
            x0 = _init_grid(W, H, rng, jitter=min(W, H) * 0.05)
        elif r < 0.7:
            x0 = _init_random(W, H, rng, corner_bias=True)
        else:
            x0 = _init_random(W, H, rng, corner_bias=False)

        x = _solve_one(x0, W, H, rng)

        # validity check
        if _is_valid(x, W, H, tol=1e-5):
            score = np.sum(x[2::3])
            if score > best_score:
                best_score = score
                best_x = x.copy()

    # final polish on best if time remains
    if best_x is not None and _elapsed() < T_LIMIT - 3:
        # figure out W,H for best_x: we stored only x; re-derive from stored? 
        # We'll re-run a light polish using the last known W,H by re-solving from best_x.
        # But best_x's W,H unknown. Instead store them alongside.
        pass

    # We need W,H for the return? The interface returns (n,3) of circles.
    # The evaluator checks bounding rectangle perimeter <= 4.
    # Our circles are in [0,W]x[0,H] with W+H=2 => perimeter 4. Good.
    if best_x is None:
        # fallback to seed
        return base

    out = best_x.reshape(N, 3)
    return out

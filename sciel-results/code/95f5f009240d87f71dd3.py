"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_radius(centers):
    """Exact maximum common radius for given centers (double precision)."""
    n = len(centers)
    if n == 0:
        return 0.0
    cx = centers[:, 0]
    cy = centers[:, 1]
    # wall distances
    wall = np.minimum(np.minimum(0.5 - np.abs(cx), 0.5 - np.abs(cy)))
    min_wall = float(np.min(wall))
    # pair distances
    if n > 1:
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d2 = dx * dx + dy * dy
        np.fill_diagonal(d2, np.inf)
        min_pair = float(np.sqrt(np.min(d2))) / 2.0
    else:
        min_pair = 0.5
    return min(min_wall, min_pair)


def _soft_min_radius(centers, beta):
    """Smooth approximation of the max radius (log-sum-exp soft-min)."""
    n = len(centers)
    cx = centers[:, 0]
    cy = centers[:, 1]
    # wall distances
    w0 = 0.5 - np.abs(cx)
    w1 = 0.5 - np.abs(cy)
    walls = np.minimum(w0, w1)
    # pair distances
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d2 = dx * dx + dy * dy
    d = np.sqrt(np.maximum(d2, 1e-300))
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0
    # soft min over walls
    wm = float(np.min(walls))
    exp_w = np.exp(-beta * (walls - wm))
    min_wall_soft = wm - np.log(np.sum(exp_w)) / beta
    # soft min over pairs
    pm = float(np.min(pair))
    exp_p = np.exp(-beta * (pair - pm))
    min_pair_soft = pm - np.log(np.sum(exp_p)) / beta
    return min(min_wall_soft, min_pair_soft)


def _objective_and_grad(x, beta):
    """Objective = -soft_min_radius (to maximize radius) and its gradient."""
    n = N
    c = x.reshape(n, 2)
    cx = c[:, 0]
    cy = c[:, 1]
    w0 = 0.5 - np.abs(cx)
    w1 = 0.5 - np.abs(cy)
    walls = np.minimum(w0, w1)

    # pair distances
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d2 = dx * dx + dy * dy
    d = np.sqrt(np.maximum(d2, 1e-300))
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0

    wm = float(np.min(walls))
    ew = np.exp(-beta * (walls - wm))
    sw = np.sum(ew)
    min_wall_soft = wm - np.log(sw) / beta

    pm = float(np.min(pair))
    ep = np.exp(-beta * (pair - pm))
    sp = np.sum(ep)
    min_pair_soft = pm - np.log(sp) / beta

    val = min(min_wall_soft, min_pair_soft)
    obj = -val

    # gradient: d(obj)/dx = -d(val)/dx
    grad = np.zeros_like(x)
    # ---- wall soft-min gradient (if wall is the binding min) ----
    if min_wall_soft <= min_pair_soft:
        # d(min_wall_soft)/d(c) = -(ew/sw) * d(walls)/d(c)
        pw = ew / sw  # weights
        dwalls = np.zeros_like(c)
        dwalls[:, 0] = -np.sign(cx) * (w0 <= w1)
        dwalls[:, 1] = -np.sign(cy) * (w1 < w0)
        grad += np.sum(pw[:, None] * dwalls, axis=0)  # -d(val) => + since obj=-val
        # obj = -val, dobj/dx = -dval/dx = -(-sum pw*dwalls) = sum pw*dwalls
    # ---- pair soft-min gradient (if pair is the binding min) ----
    if min_pair_soft < min_wall_soft:
        pw = ep / sp  # weights over pairs
        # d(pair_ij)/d(c_i) = (c_i - c_j)/(2 d_ij); d(pair_ij)/d(c_j) = (c_j - c_i)/(2 d_ij)
        # contribution to grad for point i = sum_j pw_ij * (c_i - c_j)/(2 d_ij)
        unit = np.zeros_like(c)  # accumulate
        inv2d = np.zeros_like(d)
        mask = d > 0
        inv2d[mask] = 1.0 / (2.0 * d[mask])
        # weight matrix (i,j)
        W = pw * inv2d
        np.fill_diagonal(W, 0.0)
        # grad_i = sum_j W_ij * (c_i - c_j)
        # = (sum_j W_ij) c_i - sum_j W_ij c_j
        rowsum = W.sum(axis=1)
        term = W @ c
        dval = rowsum[:, None] * c - term
        grad -= dval  # obj = -val
    return obj, grad


def _init_7x7_plus():
    """7x7 grid (49) plus one center = 50."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    pts = np.vstack([pts, np.array([0.0, 0.0])])
    return pts[:N]


def _init_jittered_8x8():
    """8x8 grid (64) -> take 50, with small jitter to break symmetry."""
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    rng = np.random.default_rng(7)
    pts = pts[:N].copy()
    pts += rng.uniform(-0.004, 0.004, size=pts.shape)
    return np.clip(pts, -0.5, 0.5)


def _init_hexagonal():
    """Hexagonal close-packed lattice clipped to square, take 50."""
    rng = np.random.default_rng(11)
    s = 0.145
    rows = []
    y = -0.48
    flip = False
    while y <= 0.48:
        x = -0.48
        if flip:
            x += s / 2.0
        while x <= 0.48:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                rows.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2.0
        flip = not flip
    pts = np.array(rows)
    if len(pts) < N:
        extra = rng.uniform(-0.5, 0.5, size=(N - len(pts), 2))
        pts = np.vstack([pts, extra])
    pts = pts[:N]
    return np.clip(pts, -0.5, 0.5)


def _init_random(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _relax_grow(centers, deadline, beta0=20.0, beta_max=300.0):
    """Growth/relaxation: iteratively optimize soft-min radius with increasing beta."""
    centers = np.clip(centers.copy(), -0.5, 0.5)
    bounds = [(-0.5, 0.5)] * (2 * N)
    beta = beta0
    best = centers.copy()
    best_r = _compute_radius(centers)
    for _ in range(40):
        if time.time() > deadline - 3.0:
            break
        x0 = best.flatten()
        try:
            res = minimize(_objective_and_grad, x0, jac=True, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            cand = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
            r = _compute_radius(cand)
            if r > best_r + 1e-12:
                best = cand
                best_r = r
        except Exception:
            pass
        beta = min(beta * 1.5, beta_max)
    return best


def _local_pair_push(centers, deadline):
    """Simple hard-disk relaxation: repeatedly push overlapping pairs apart."""
    centers = centers.copy()
    for _ in range(300):
        if time.time() > deadline - 2.0:
            break
        moved = False
        dx = centers[:, None, 0] - centers[None, :, 0]
        dy = centers[:, None, 1] - centers[None, :, 1]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        min_d = float(np.min(d))
        r = min_d / 2.0
        # for each pair too close, push apart
        close = d < min_d * 1.001
        for i in range(N):
            for j in range(i + 1, N):
                if close[i, j]:
                    d = d[i, j]
                    if d < 1e-9:
                        continue
                    ux = dx[i, j] / d
                    uy = dy[i, j] / d
                    shift = (min_d * 1.001 - d) / 2.0
                    centers[i, 0] += ux * shift
                    centers[i, 1] += uy * shift
                    centers[j, 0] -= ux * shift
                    centers[j, 1] -= uy * shift
                    moved = True
        centers = np.clip(centers, -0.5, 0.5)
        if not moved:
            break
    return centers


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    inits = [
        _init_7x7_plus(),
        _init_jittered_8x8(),
        _init_hexagonal(),
        _init_random(1),
        _init_random(2),
        _init_random(3),
    ]

    best_centers = None
    best_r = -1.0

    for k, init in enumerate(inits):
        if time.time() > deadline - 8.0:
            break
        # First a quick hard-disk relaxation to get a feasible spread
        c = _local_pair_push(init, deadline)
        r0 = _compute_radius(c)
        # Then growth/relaxation with soft-min gradient optimization
        c2 = _relax_grow(c, deadline, beta0=20.0, beta_max=300.0)
        r2 = _compute_radius(c2)
        # Pick the better of the two
        cand = c2 if r2 >= r0 else c
        rc = _compute_radius(cand)
        if rc > best_r:
            best_r = rc
            best_centers = cand.copy()

    # Final polish on the best with a higher beta
    if best_centers is not None and time.time() < deadline - 5.0:
        polished = _relax_grow(best_centers, deadline, beta0=100.0, beta_max=400.0)
        if _compute_radius(polished) >= best_r:
            best_centers = polished

    if best_centers is None:
        return _init_7x7_plus()
    return best_centers

# EVOLVE-BLOCK-END

"""Fresh design: smooth soft-min packing objective + L-BFGS-B + binary-search radius escalation."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
HALF = 0.5

# Fixed deadline guard (safety margin under 150 s kill).
T_START = time.time()
DEADLINE = 135.0


def _time_left():
    return DEADLINE - (time.time() - T_START)


def _score_of(pts):
    """Raw score: min(half pairwise distance, min wall clearance)."""
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() * 0.5
    wall = (HALF - np.abs(pts)).min()
    return float(min(min_pair, wall))


def _objective_flat(x, r):
    """Soft-min surrogate: penalise distances below target r (smooth, differentiable)."""
    pts = x.reshape(-1, 2)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    # soft-min of (d - r) over pairs: negative => violation
    s = 0.05
    pair = -np.logaddexp(0.0, -(d - r) / s) * s
    # soft-min of wall clearance (HALF - |coord| - r)
    wall = HALF - np.abs(pts) - r
    wallsm = -np.logaddexp(0.0, -wall / s) * s
    return float(np.minimum(pair, wallsm))


def _objective_grad(x, r):
    """Objective and analytic gradient of the soft-min surrogate."""
    s = 0.05
    pts = x.reshape(-1, 2)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    # soft-min g = -s*log(sum exp(-(d_ij - r)/s)) over pairs i<j
    # d g/d d_ij = exp(-(d_ij - r)/s)/sum
    e = np.exp(-(d - r) / s)
    np.fill_diagonal(e, 0.0)
    w = e / e.sum()
    # grad wrt pts: sum_j w_ij * d_ij * (pts_i - pts_j)/d_ij = sum_j w_ij*(pts_i - pts_j)
    grad_pts = np.zeros_like(pts)
    # (pts_i - pts_j) weighted: grad_pts[i] = sum_j w_ij (pts_i - pts_j)
    # = sum_j w_ij pts_i - sum_j w_ij pts_j = (sum_j w_ij) pts_i - sum_j w_ij pts_j
    rowsum = w.sum(1)[:, None]
    term = np.sum(w[:, :, None] * pts[None, :, :], axis=1)
    grad_pts = rowsum * pts - term
    # wall soft-min: h_i = HALF - |pts_i| - r ; soft-min over all coords
    hw = HALF - np.abs(pts) - r
    ew = np.exp(-hw / s)
    wwall = ew / ew.sum()
    # d hw_i / d pts_i = -sign(pts_i)
    grad_wall = -np.sign(pts) * wwall
    g = grad_pts + grad_wall
    val = float(min(
        -np.log(e.sum()) * s,
        -np.log(ew.sum()) * s,
    ))
    return val, g.ravel()


def _objective_closure(x, r):
    return _objective_grad(x, r)


def _valid(pts):
    if not np.all(np.isfinite(pts)):
        return False
    if pts.shape != (N, 2):
        return False
    if np.any(np.abs(pts) > HALF + 1e-9):
        return False
    # coincident centres
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-9:
        return False
    return True


def _make_initial(r):
    """Hexagonal grid initialisation clipped into the square, sized for target r."""
    spacing = max(2 * r, 0.03)
    pts_list = []
    y = -HALF + spacing * 0.5
    row = 0
    while len(pts_list) < N and y < HALF:
        xoff = (spacing * 0.5) if (row % 2) else 0.0
        x = -HALF + spacing * 0.5 + xoff
        while x < HALF:
            pts_list.append([x, y])
            x += spacing
        y += spacing * np.sqrt(3.0) / 2.0
        row += 1
    pts = np.array(pts_list[:N], dtype=float)
    if len(pts) < N:
        # pad with a grid fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - HALF
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = grid[:N]
    return pts


def _optimize_target(r, start):
    """Run L-BFGS-B on the soft-min surrogate for target radius r."""
    x0 = start.ravel()

    def fun(x):
        return _objective_grad(x, r)[0]

    def jac(x):
        return _objective_grad(x, r)[1]

    res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                   options={'maxiter': 800, 'maxls': 40, 'ftol': 1e-12,
                            'gtol': 1e-10, 'disp': False})
    return res.x.reshape(-1, 2)


def construct():
    global T_START
    T_START = time.time()

    best_pts = None
    best_score = -np.inf

    def consider(pts):
        nonlocal best_pts, best_score
        if _valid(pts):
            sc = _score_of(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()

    # ---- Phase 1: find a feasible starting radius via binary search on target.
    # We escalate target radius; each attempt optimises the soft-min surrogate.
    lo, hi = 0.02, 0.045
    # First, get a feasible baseline from a moderate target.
    base_r = 0.03
    start = _make_initial(base_r)
    opt = _optimize_target(base_r, start)
    consider(opt)

    # Escalate target radius upward using binary search on feasibility.
    cur_r = base_r
    cur_pts = opt
    # Try to push higher: exponential then binary refinement.
    attempts = 0
    while _time_left() > 25 and attempts < 20:
        attempts += 1
        new_r = cur_r * 1.10
        if new_r > hi:
            new_r = hi
        start = _make_initial(new_r)
        # also try perturbing from current best to escape local minima
        start2 = cur_pts + np.random.default_rng(0).normal(0, 0.002, size=(N, 2))
        start2 = np.clip(start2, -HALF, HALF)
        cand1 = _optimize_target(new_r, start)
        cand2 = _optimize_target(new_r, start2)
        s1 = _score_of(cand1) if _valid(cand1) else -np.inf
        s2 = _score_of(cand2) if _valid(cand2) else -np.inf
        if max(s1, s2) > best_score:
            cur_r = new_r
            cur_pts = cand1 if s1 >= s2 else cand2
            consider(cur_pts)
        else:
            # could not improve; do a binary refinement between cur_r and new_r
            mid = (cur_r + new_r) / 2.0
            startm = _make_initial(mid)
            candm = _optimize_target(mid, startm)
            if _valid(candm) and _score_of(candm) > best_score:
                cur_r = mid
                cur_pts = candm
                consider(cur_pts)
            break

    # ---- Phase 2: final polish with tighter soft-min and more iterations.
    if _time_left() > 20 and best_pts is not None:
        r_polish = _score_of(best_pts) + 1e-5
        # tighter smoothness
        global _S_OVERRIDE
        _S_OVERRIDE = 0.02

        def fun2(x):
            return _objective_grad2(x, r_polish)[0]

        def jac2(x):
            return _objective_grad2(x, r_polish)[1]

        res = minimize(fun2, best_pts.ravel(), jac=jac2, method='L-BFGS-B',
                       options={'maxiter': 1200, 'maxls': 40, 'ftol': 1e-14,
                                'gtol': 1e-12, 'disp': False})
        cand = res.x.reshape(-1, 2)
        consider(cand)

    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - HALF
        best_pts = np.array([(x, y) for y in ticks for x in ticks])[:N]

    return best_pts


_S_OVERRIDE = 0.05


def _objective_grad2(x, r):
    s = _S_OVERRIDE
    pts = x.reshape(-1, 2)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    e = np.exp(-(d - r) / s)
    np.fill_diagonal(e, 0.0)
    w = e / e.sum()
    rowsum = w.sum(1)[:, None]
    term = np.sum(w[:, :, None] * pts[None, :, :], axis=1)
    grad_pts = rowsum * pts - term
    hw = HALF - np.abs(pts) - r
    ew = np.exp(-hw / s)
    wwall = ew / ew.sum()
    grad_wall = -np.sign(pts) * wwall
    g = grad_pts + grad_wall
    val = float(min(-np.log(e.sum()) * s, -np.log(ew.sum()) * s))
    return val, g.ravel()

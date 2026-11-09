# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_TOL = 1e-7
_N = 10


def _is_valid(centers, sides):
    n = len(sides)
    c = np.asarray(centers, dtype=float)
    s = np.asarray(sides, dtype=float)
    if c.shape != (n, 2) or s.shape != (n,):
        return False
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(s))):
        return False
    if np.any(s <= 0):
        return False
    # boundary
    for i in range(n):
        if c[i, 0] - s[i] / 2 < -_TOL or c[i, 0] + s[i] / 2 > 1 + _TOL:
            return False
        if c[i, 1] - s[i] / 2 < -_TOL or c[i, 1] + s[i] / 2 > 1 + _TOL:
            return False
    # overlap (forbidden)
    for i in range(n):
        for j in range(i + 1, n):
            t = (s[i] + s[j]) / 2 - _TOL
            if abs(c[i, 0] - c[j, 0]) < t and abs(c[i, 1] - c[j, 1]) < t:
                return False
    return True


def _objective(x):
    n = _N
    c = x[: 2 * n].reshape(n, 2)
    s = x[2 * n:]
    # sides must be positive
    if np.any(s <= 1e-9):
        return 1e9
    # boundary violations (soft, large penalty)
    pen = 0.0
    for i in range(n):
        lo = c[i, 0] - s[i] / 2
        hi = c[i, 0] + s[i] / 2
        if lo < 0:
            pen += (1e6 + 1e3 * (1 - lo)) * (-lo)
        if hi > 1:
            pen += (1e6 + 1e3 * (1 + hi)) * (hi - 1)
        lo = c[i, 1] - s[i] / 2
        hi = c[i, 1] + s[i] / 2
        if lo < 0:
            pen += (1e6 + 1e3 * (1 - lo)) * (-lo)
        if hi > 1:
            pen += (1e6 + 1e3 * (1 + hi)) * (hi - 1)
    # overlap violations (soft, large penalty)
    for i in range(n):
        for j in range(i + 1, n):
            t = (s[i] + s[j]) / 2
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            # require separation: dx^2 + dy^2 >= t^2
            d2 = dx * dx + dy * dy
            if d2 < t * t:
                gap = t - np.sqrt(d2)
                pen += 1e6 * gap + 1e3 * gap * gap
    return pen - np.sum(s)


def _initial_grid(n=10):
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _initial_packed(n=10):
    # A better hand-crafted layout: one big square in a corner, others around it.
    # Big square side 0.5 in bottom-left.
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    # square 0: big, bottom-left
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    # remaining 9 in the top strip (y from 0.5 to 1) and right strip (x from 0.5 to 1)
    # Top strip: x in [0,1], y in [0.5,1] -> height 0.5, width 1.0
    # Place 6 squares of side 0.5/3? Let's do a 3x2 grid in top strip.
    # Actually simpler: top strip split into 3 columns x 2 rows of side 1/6.
    # Right strip: x in [0.5,1], y in [0,0.5] -> width 0.5, height 0.5 -> 2x2 grid side 0.25
    # That gives 6 + 4 = 10 total.
    idx = 1
    # top strip: 3 cols, 2 rows, side = 1/6
    st = 1.0 / 6
    for r in range(2):
        for c in range(3):
            centers[idx] = [(c + 0.5) * st, 0.5 + (r + 0.5) * st]
            sides[idx] = st
            idx += 1
    # right strip: 2 cols, 2 rows, side = 0.25
    sr = 0.25
    for r in range(2):
        for c in range(2):
            centers[idx] = [0.5 + (c + 0.5) * sr, (r + 0.5) * sr]
            sides[idx] = sr
            idx += 1
    return centers, sides


def _optimize_from(centers, sides, deadline):
    n = _N
    x0 = np.concatenate([centers.ravel(), sides])
    # Nelder-Mead with tight settings
    best_x = x0.copy()
    best_f = _objective(x0)
    # Multiple random perturbations to escape local minima
    rng = np.random.default_rng(12345)
    attempts = 0
    while time.time() < deadline:
        # try the current best
        res = minimize(_objective, best_x, method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        attempts += 1
        if res.success or res.fun < best_f:
            cand = res.x
            if cand[2 * n:].min() > 1e-9:
                best_x = cand
                best_f = res.fun
        if time.time() > deadline:
            break
        # perturb for next attempt
        if attempts >= 8:
            break
        pert = best_x.copy()
        # small random jitter
        pert[:2 * n] += rng.normal(0, 0.02, 2 * n)
        pert[2 * n:] *= (1 + rng.normal(0, 0.05, n))
        # clip to feasible-ish region
        for i in range(n):
            pert[2 * n + i] = min(max(pert[2 * n + i], 1e-4), 1.0)
            pert[2 * i] = min(max(pert[2 * i], 0.0), 1.0)
            pert[2 * i + 1] = min(max(pert[2 * i + 1], 0.0), 1.0)
        if _objective(pert) < best_f:
            best_x = pert
            best_f = _objective(pert)
    return best_x, best_f


def squares_in_square10() -> tuple:
    start = time.time()
    deadline = start + 600.0  # safety margin

    # Candidate initial layouts
    layouts = []
    g_c, g_s = _initial_grid(_N)
    layouts.append((g_c, g_s))
    p_c, p_s = _initial_packed(_N)
    layouts.append((p_c, p_s))

    best_centers = None
    best_sides = None
    best_score = -np.inf

    for c, s in layouts:
        if not _is_valid(c, s):
            # try to repair by shrinking
            s = s * 0.95
            c = c.copy()
            if not _is_valid(c, s):
                continue
        x, f = _optimize_from(c, s, deadline)
        if time.time() > deadline:
            break
        cc = x[:2 * _N].reshape(_N, 2)
        ss = x[2 * _N:]
        if ss.min() <= 1e-9:
            continue
        if _is_valid(cc, ss):
            sc = float(np.sum(ss))
            if sc > best_score:
                best_score = sc
                best_centers = cc
                best_sides = ss
        if time.time() > deadline - 120:
            break

    if best_centers is None:
        # fallback to grid
        return _initial_grid(_N)

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END

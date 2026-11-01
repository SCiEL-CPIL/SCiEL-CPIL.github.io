"""Circle packing in unit square, n=32.
Structural approach: smooth differentiable objective over centers,
directly maximizing a lower bound on the sum of radii via L-BFGS-B
with random + hexagonal restarts and a hard deadline guard.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 32
T0 = time.time()
DEADLINE = 150.0
MARGIN = 6.0          # safety margin before the 150s kill
EPS = 1e-9            # clearance / boundary epsilon
MAXF = 2000           # max function evals per L-BFGS-B run


def _softmin(a, s):
    # smooth approximation of min(a); larger s -> closer to true min
    return -np.log(np.sum(np.exp(-s * a))) / s


def _clearance(x, s):
    # returns (sum of radius lower-bound, per-circle radius lower-bound)
    c = x.reshape(-1, 2)
    b = np.minimum(c, 1.0 - c).reshape(-1)          # boundary clearance
    d = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1) / 2.0                         # nearest-neighbor bound
    r = np.minimum(b, nn)
    return float(np.sum(r)), r


def _negobj(x, s):
    val, _ = _clearance(x, s)
    return -val


def _hex_start(n):
    # hexagonal lattice (density-optimal) clipped inside the square
    a = 0.135
    dx = a
    dy = a * np.sqrt(3.0)
    pts = []
    for row in range(-1, 14):
        for col in range(-1, 14):
            x = 0.5 + col * dx + (0.5 * dx if (row % 2) else 0.0)
            y = 0.5 + row * dy
            if 0.0 < x < 1.0 and 0.0 < y < 1.0:
                pts.append((x, y))
    pts = np.array(pts[:n])
    return pts


def _optimize(x0, s):
    try:
        res = minimize(_negobj, x0, args=(s,), method='L-BFGS-B',
                       bounds=[(EPS, 1.0 - EPS)] * (2 * N),
                       options={'maxiter': 600, 'maxfun': MAXF,
                                'ftol': 1e-12, 'gtol': 1e-8})
        x = np.clip(res.x, EPS, 1.0 - EPS)
        return x
    except Exception:
        return None


def _finalize(x):
    c = np.clip(x.reshape(-1, 2), EPS, 1.0 - EPS)
    b = np.minimum(c, 1.0 - c)
    d = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1) / 2.0
    r = np.maximum(np.minimum(b, nn) - 1e-10, 1e-9)
    out = np.zeros((N, 3))
    out[:, 0] = c[:, 0]
    out[:, 1] = c[:, 1]
    out[:, 2] = r
    return out


def construct():
    global T0
    rng = np.random.default_rng(12345)
    best_x = None
    best_val = -np.inf

    def consider(x):
        nonlocal best_x, best_val
        if x is None:
            return
        x = np.clip(x, EPS, 1.0 - EPS)
        val, _ = _clearance(x, 80.0)
        if val > best_val:
            best_val = val
            best_x = x.copy()

    # seed with hexagonal start (good warm start, deterministic)
    hx = _hex_start(N).reshape(-1)
    consider(hx)
    hx_opt = _optimize(hx, 40.0)
    consider(hx_opt)

    # random restarts with annealed softmin
    n_restarts = 10
    for k in range(n_restarts):
        if time.time() > DEADLINE - MARGIN:
            break
        s = 40.0 * (0.55 ** k)
        x0 = rng.uniform(0.05, 0.95, size=(N, 2)).reshape(-1)
        x = _optimize(x0, s)
        consider(x)

    if best_x is None:
        # fallback: trivial grid (guaranteed valid)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return _finalize(best_x)

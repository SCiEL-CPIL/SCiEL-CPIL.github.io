import time
import numpy as np
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 125.0   # safety margin below the 150 s wall-clock kill


def _softmin(v, k):
    """Smooth minimum of vector v (log-sum-exp trick)."""
    m = np.min(v)
    if np.isfinite(m):
        return m - np.log(np.sum(np.exp(k * (v - m)))) / k
    return m


def _radius_from_aspect(w, r0):
    """Feasible uniform radius for aspect w (width w, height 2-w), 5x5 grid."""
    s = 5
    cell = min(w / s, (2.0 - w) / s)
    return 0.5 * cell - 1e-6


def construct():
    # ---------- quick baseline (grid) ----------
    w0 = 2.0 / (1 + 1)  # 1x2 -> square-ish; just a safe start
    best = None
    best_val = -np.inf

    def evaluate(w, centers, radii):
        """Return (value, feasible). value = sum(radii) if feasible else -inf-ish."""
        h = 2.0 - w
        # boundary feasibility
        ok = True
        for i in range(len(centers)):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-7 or x + r > w + 1e-7 or y - r < -1e-7 or y + r > h + 1e-7:
                ok = False
        # pairwise feasibility
        if ok:
            d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
            minsep = np.min(d - np.outer(radii, radii))
            if minsep < -1e-7:
                ok = False
        if ok:
            return np.sum(radii), True
        return -1e9, False

    # baseline uniform grid on a square
    side = int(np.ceil(np.sqrt(N)))
    w = 2.0 / (1 + 1)
    h = 2.0 - w
    centers = []
    radii = []
    r0 = _radius_from_aspect(w, None)
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            centers.append(((i + 0.5) / side * w, (j + 0.5) / side * h))
            radii.append(r0)
    centers = np.array(centers[:N])
    radii = np.array(radii[:N])
    v, feas = evaluate(w, centers, radii)
    if feas:
        best_val = v
        best = np.column_stack([centers, radii])

    # ---------- optimization ----------
    def make_objective(w, h, r0):
        # variables: [cx_1..cx_N, cy_1..cy_N, r_1..r_N]
        def loss(z):
            cx = z[0:N]
            cy = z[N:2 * N]
            r = z[2 * N:3 * N]
            # smooth pair separation violations (want d_ij >= r_i + r_j)
            d = np.linalg.norm(cx[:, None] - cx[None, :], axis=2)
            d = np.linalg.norm(np.stack([cx[:, None] - cx[None, :], cy[:, None] - cy[None, :]], axis=-1), axis=2)
            sep = d - np.outer(r, r)
            # take off-diagonal only
            iu = np.triu_indices(N, 1)
            sep = sep[iu]
            # boundary violations (want margin >= 0)
            bnd = np.concatenate([
                cx - r,
                (w - cx) - r,
                cy - r,
                (h - cy) - r,
            ])
            allv = np.concatenate([sep, bnd])
            k = 40.0
            viol = _softmin(allv, k)
            # penalty for negative soft-min (violation)
            pen = np.maximum(0.0, -viol)
            return -(np.sum(r)) + 200.0 * pen * pen + 1e-4 * np.sum(r * r)
        return loss

    rng = np.random.default_rng(12345)
    best = None
    best_val = -np.inf
    best_w = None

    # candidate aspect ratios (width w, height 2-w); w in (0,2)
    aspect_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

    for w in aspect_list:
        if time.time() > DEADLINE - 5:
            break
        h = 2.0 - w
        r0 = _radius_from_aspect(w, None)
        # initial centers on grid
        side = int(np.ceil(np.sqrt(N)))
        cx = np.array([(i + 0.5) / side * w for i in range(side) for j in range(side)][:N])
        cy = np.array([(j + 0.5) / side * h for i in range(side) for j in range(side)][:N])
        r = np.full(N, r0)
        z0 = np.concatenate([cx, cy, r])
        loss = make_objective(w, h, r0)

        # a couple of jittered starts
        for trial in range(2):
            if time.time() > DEADLINE - 5:
                break
            if trial == 0:
                zz = z0.copy()
            else:
                zz = z0.copy()
                zz[:2 * N] += rng.normal(0, 0.02 * min(w, h), size=2 * N)
                zz[2 * N:] *= 1.0
            res = minimize(loss, zz, method='L-BFGS-B',
                           options={'maxiter': 400, 'maxfun': 2000,
                                    'ftol': 1e-12, 'gtol': 1e-9})
            z = res.x
            cx, cy, r = z[0:N], z[N:2 * N], z[2 * N:3 * N]
            # clip tiny negatives
            r = np.maximum(r, 0.0)
            v, feas = evaluate(w, np.column_stack([cx, cy]), r)
            if feas and v > best_val:
                best_val = v
                best_w = w
                best = np.column_stack([cx, cy, r])

    # ---------- high-precision polish on best ----------
    if best is not None and time.time() < DEADLINE - 3:
        w = best_w
        h = 2.0 - w
        cx, cy, r = best[:, 0], best[:, 1], best[:, 2]
        z = np.concatenate([cx, cy, r])
        loss = make_objective(w, h, np.mean(r))
        res = minimize(loss, z, method='L-BFGS-B',
                       options={'maxiter': 2000, 'maxfun': 20000,
                                'ftol': 1e-16, 'gtol': 1e-12})
        z = res.x
        cx, cy, r = z[0:N], z[N:2 * N], z[2 * N:3 * N]
        r = np.maximum(r, 0.0)
        v, feas = evaluate(w, np.column_stack([cx, cy]), r)
        if feas and v > best_val:
            best_val = v
            best = np.column_stack([cx, cy, r])

    # ---------- final validity check; fall back to grid if needed ----------
    if best is None:
        # guaranteed-feasible square grid
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        r0 = 0.5 / side - 1e-3
        cells = [((i + 0.5) / side, (j + 0.5) / side, r0) for i in range(side) for j in range(side)]
        best = np.array(cells[:N])
        return best

    # re-check best
    v, feas = evaluate(best_w, best[:, :2], best[:, 2])
    if not feas:
        side = int(np.ceil(np.sqrt(N)))
        r0 = 0.5 / side - 1e-3
        cells = [((i + 0.5) / side, (j + 0.5) / side, r0) for i in range(side) for j in range(side)]
        best = np.array(cells[:N])

    return best

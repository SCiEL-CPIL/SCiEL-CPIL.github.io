"""Circle packing in unit square: maximize sum of radii for N disjoint circles."""
import time
import numpy as np

N = 26
_T0 = time.time()
_DEADLINE = 110.0  # safety margin before the 120 s kill

def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _violation_and_obj(pts, r):
    """Return (total_violation, sum_radii). Vectorized."""
    n = r.shape[0]
    # wall violations
    vw = np.maximum(0.0, r - pts[:, 0])
    vw += np.maximum(0.0, r - (1.0 - pts[:, 0]))
    vw += np.maximum(0.0, r - pts[:, 1])
    vw += np.maximum(0.0, r - (1.0 - pts[:, 1]))
    wall_v = float(np.sum(vw))
    # pairwise overlap
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    req = r[:, None] + r[None, :]
    ov = np.maximum(0.0, req - dist)
    np.fill_diagonal(ov, 0.0)
    pair_v = float(np.sum(ov))
    return wall_v + pair_v, float(np.sum(r))


def _grad(pts, r, lam):
    """Gradient of  sum(r) - lam*(wall_v + pair_v)  w.r.t. (pts, r)."""
    n = r.shape[0]
    g_pts = np.zeros_like(pts)
    g_r = np.ones_like(r)  # d sum(r)/d r
    # wall terms
    g_pts[:, 0] -= lam * (np.maximum(0.0, r - pts[:, 0]) > 0) * 1.0
    g_pts[:, 0] += lam * (np.maximum(0.0, r - (1.0 - pts[:, 0])) > 0) * 1.0
    g_pts[:, 1] -= lam * (np.maximum(0.0, r - pts[:, 1]) > 0) * 1.0
    g_pts[:, 1] += lam * (np.maximum(0.0, r - (1.0 - pts[:, 1])) > 0) * 1.0
    g_r -= lam * (
        (np.maximum(0.0, r - pts[:, 0]) > 0).astype(float)
        + (np.maximum(0.0, r - (1.0 - pts[:, 0])) > 0).astype(float)
        + (np.maximum(0.0, r - pts[:, 1]) > 0).astype(float)
        + (np.maximum(0.0, r - (1.0 - pts[:, 1])) > 0).astype(float)
    )
    # pairwise terms
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    dist = np.where(dist > 1e-12, dist, 1e-12)
    req = r[:, None] + r[None, :]
    act = (req - dist) > 0
    unit = d / dist[:, :, None]
    # contribution to g_pts[i] from pair (i,j): -lam * act * unit[i,j]
    g_pts -= lam * np.sum(act[:, :, None] * unit, axis=1)
    # contribution to g_r[i]: -lam * act[i,j]
    g_r -= lam * np.sum(act, axis=1)
    return g_pts, g_r


def _grad_obj(pts, r):
    """Gradient of objective (sum r) under hard feasibility for L-BFGS-B."""
    n = r.shape[0]
    g_pts = np.zeros_like(pts)
    g_r = np.ones_like(r)
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    dist = np.where(dist > 1e-12, dist, 1e-12)
    req = r[:, None] + r[None, :]
    act = (req - dist) > 0
    unit = d / dist[:, :, None]
    g_pts -= np.sum(act[:, :, None] * unit, axis=1)
    g_r -= np.sum(act, axis=1)
    return g_pts, g_r


def _feasible(pts, r, tol=1e-7):
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    req = r[:, None] + r[None, :]
    if np.any(req - dist > tol):
        return False
    if np.any(r - pts[:, 0] > tol) or np.any(r - (1.0 - pts[:, 0]) > tol):
        return False
    if np.any(r - pts[:, 1] > tol) or np.any(r - (1.0 - pts[:, 1]) > tol):
        return False
    return True


def _enforce_hard(pts, r):
    """Project to feasibility: shrink radii to fit walls and pairwise gaps."""
    for _ in range(60):
        # wall
        for i in range(r.shape[0]):
            cap = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
            if r[i] > cap:
                r[i] = max(cap, 0.0)
        # pairwise: shrink the larger radius of each overlapping pair
        improved = False
        for i in range(r.shape[0]):
            for j in range(i + 1, r.shape[0]):
                gap = np.linalg.norm(pts[i] - pts[j])
                need = r[i] + r[j]
                if need > gap + 1e-12:
                    excess = need - gap
                    # reduce the bigger radius (or split)
                    if r[i] >= r[j]:
                        cut = min(excess, r[i])
                        r[i] -= cut
                        excess -= cut
                        if excess > 0:
                            r[j] -= min(excess, r[j])
                    else:
                        cut = min(excess, r[j])
                        r[j] -= cut
                        excess -= cut
                        if excess > 0:
                            r[i] -= min(excess, r[i])
                    improved = True
        if not improved:
            break
    return pts, r


def _initial_grid(n):
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append([(i + 0.5) / side, (j + 0.5) / side])
    pts = np.array(pts[:n])
    base = 0.5 / side
    r = np.full(n, base * 0.9)
    return pts, r


def _expand_greedy(pts, r):
    """Try to grow radii by binary search on a global scale factor."""
    best = (pts.copy(), r.copy())
    lo, hi = 1.0, 3.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        rr = r * mid
        pts2, rr2 = _enforce_hard(pts.copy(), rr.copy())
        if float(np.sum(rr2)) >= float(np.sum(best[1])) - 1e-9:
            best = (pts2, rr2)
            lo = mid
        else:
            hi = mid
    return best


def _gradient_descent(pts, r, lam0, iters, step0):
    lam = lam0
    step = step0
    prev_v, prev_obj = _violation_and_obj(pts, r)
    best = (pts.copy(), r.copy(), prev_obj)
    for it in range(iters):
        if _time_left() < 5.0:
            break
        g_pts, g_r = _grad(pts, r, lam)
        pts_new = pts + step * g_pts
        r_new = np.maximum(1e-6, r + step * g_r)
        # clip centers into a loose box to keep numerics sane
        pts_new = np.clip(pts_new, -0.5, 1.5)
        v_new, obj_new = _violation_and_obj(pts_new, r_new)
        if obj_new > prev_obj and v_new < prev_v + 1e-9:
            pts, r = pts_new, r_new
            prev_v, prev_obj = v_new, obj_new
            if obj_new > best[2]:
                best = (pts.copy(), r.copy(), obj_new)
            step = min(step * 1.05, 0.05)
        else:
            step *= 0.5
            lam = min(lam * 1.2, 1e6)
            if step < 1e-6:
                # nudge with a fresh small random perturbation to escape
                pts = pts + 0.005 * np.random.randn(*pts.shape)
                pts = np.clip(pts, 0.0, 1.0)
                step = 0.02
        # periodically enforce hard feasibility to keep radii meaningful
        if it % 40 == 0:
            pts, r = _enforce_hard(pts, r)
            prev_v, prev_obj = _violation_and_obj(pts, r)
            if obj_new if False else float(np.sum(r)) > best[2]:
                best = (pts.copy(), r.copy(), float(np.sum(r)))
    return best


def _local_search(pts, r, maxiter=400):
    """L-BFGS-B on hard-constrained objective (sum r), maximizing via negation."""
    from scipy.optimize import minimize
    n = r.shape[0]
    x0 = np.concatenate([pts.ravel(), r.ravel()])
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n

    def fun(x):
        p = x[:2 * n].reshape(n, 2)
        rr = x[2 * n:]
        # objective
        obj = float(np.sum(rr))
        # hard constraint penalty (large) to keep feasibility
        d = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(d, axis=2)
        req = rr[:, None] + rr[None, :]
        ov = np.maximum(0.0, req - dist)
        np.fill_diagonal(ov, 0.0)
        vw = (np.maximum(0.0, rr - p[:, 0]) + np.maximum(0.0, rr - (1.0 - p[:, 0]))
              + np.maximum(0.0, rr - p[:, 1]) + np.maximum(0.0, rr - (1.0 - p[:, 1])))
        pen = 1e4 * (float(np.sum(ov)) + float(np.sum(vw)))
        return -obj + pen

    def jac(x):
        p = x[:2 * n].reshape(n, 2)
        rr = x[2 * n:]
        g_pts, g_r = _grad_obj(p, rr)
        # add penalty gradient
        d = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(d, axis=2)
        dist = np.where(dist > 1e-12, dist, 1e-12)
        req = rr[:, None] + rr[None, :]
        act = (req - dist) > 0
        unit = d / dist[:, :, None]
        g_pts = g_pts + 1e4 * np.sum(act[:, :, None] * unit, axis=1)
        g_r = g_r + 1e4 * np.sum(act, axis=1)
        g_r += 1e4 * (
            (np.maximum(0.0, rr - p[:, 0]) > 0).astype(float)
            + (np.maximum(0.0, rr - (1.0 - p[:, 0])) > 0).astype(float)
            + (np.maximum(0.0, rr - p[:, 1]) > 0).astype(float)
            + (np.maximum(0.0, rr - (1.0 - p[:, 1])) > 0).astype(float)
        )
        return np.concatenate([g_pts.ravel(), g_r])

    res = minimize(fun, x0, jac=jac, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6})
    x = res.x
    p = x[:2 * n].reshape(n, 2)
    rr = x[2 * n:]
    p, rr = _enforce_hard(p, rr)
    return p, rr


def construct():
    rng = np.random.default_rng(12345)
    best = None
    best_val = -1.0

    # --- Phase 1: gradient descent from a few diverse starts ---
    starts = []
    base_pts, base_r = _initial_grid(N)
    starts.append((base_pts.copy(), base_r.copy()))
    for k in range(3):
        p, rr = base_pts.copy(), base_r.copy()
        p = p + 0.03 * rng.standard_normal(p.shape)
        p = np.clip(p, 0.0, 1.0)
        starts.append((p, rr))

    for p, rr in starts:
        if _time_left() < 15.0:
            break
        p, rr = _expand_greedy(p, rr)
        p, rr, val = _gradient_descent(p, rr, lam0=10.0, iters=4000, step0=0.01)
        if val > best_val:
            best_val = val
            best = (p.copy(), rr.copy())

    # --- Phase 2: local search polish on the best (and a couple alternates) ---
    candidates = [best]
    for p, rr in starts:
        p, rr = _expand_greedy(p, rr)
        candidates.append((p, rr))
    # dedupe by value
    seen = set()
    uniq = []
    for p, rr in candidates:
        key = (round(float(np.sum(rr)), 6), tuple(np.round(p, 4).ravel()))
        if key not in seen:
            seen.add(key)
            uniq.append((p, rr))

    for p, rr in uniq[:4]:
        if _time_left() < 12.0:
            break
        try:
            p2, rr2 = _local_search(p, rr, maxiter=300)
            val = float(np.sum(rr2))
            if _feasible(p2, rr2) and val > best_val:
                best_val = val
                best = (p2.copy(), rr2.copy())
        except Exception:
            pass

    # --- Final safety: ensure feasibility of the returned best ---
    if best is None:
        p, rr = _initial_grid(N)
        p, rr = _enforce_hard(p, rr)
    else:
        p, rr = best
        p, rr = _enforce_hard(p, rr)
        if not _feasible(p, rr):
            # fallback to a guaranteed-feasible grid
            p, rr = _initial_grid(N)
            p, rr = _enforce_hard(p, rr)

    return np.column_stack([p[:, 0], p[:, 1], rr])

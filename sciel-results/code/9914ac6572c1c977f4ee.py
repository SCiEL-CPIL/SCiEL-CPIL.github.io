"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.

DIFFERENT STRATEGY: force-directed / repulsive-potential annealing (physics-based
global search) instead of penalty-based L-BFGS-B. Circles repel via a smooth 1/d^2
potential plus wall repulsion, with velocity damping, temperature-driven jitter and
deterministic multi-restart; a short gradient polish + exact radius maximization
finish each candidate.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
EPS = 1e-9


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _valid(cands):
    """Return (arr, score) for a valid packing of N circles, else (None, -1)."""
    valid = []
    for x, y, r in cands:
        if r <= 0:
            continue
        if x < r + EPS or x > 1 - r - EPS or y < r + EPS or y > 1 - r - EPS:
            continue
        ok = True
        for x2, y2, r2 in valid:
            if (x - x2) ** 2 + (y - y2) ** 2 < (r + r2) ** 2 - 1e-9:
                ok = False
                break
        if ok:
            valid.append((x, y, r))
    if len(valid) < N:
        return None, -1.0
    arr = np.array(valid[:N])
    return arr, float(arr[:, 2].sum())


def _polish(pos, rad, deadline):
    """Short gradient polish: fix radii, relax positions to remove overlap, then max radii."""
    pos = np.array(pos, dtype=float).copy()
    rad = np.array(rad, dtype=float).copy()

    def pos_obj(p_flat, rad_fixed):
        p = p_flat.reshape(N, 2)
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        viol = np.maximum(rad_fixed[:, None] + rad_fixed[None, :] - d, 0.0)
        w = np.maximum(rad_fixed - p, 0.0) + np.maximum(rad_fixed - (1.0 - p), 0.0)
        return np.sum(viol ** 2) + np.sum(w ** 2)

    def pos_grad(p_flat, rad_fixed):
        p = p_flat.reshape(N, 2)
        g = np.zeros_like(p_flat)
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        d = np.maximum(d, EPS)
        diff = p[:, None, :] - p[None, :, :]
        viol = np.maximum(rad_fixed[:, None] + rad_fixed[None, :] - d, 0.0)
        d_pair = 2.0 * viol
        dp = (d_pair)[:, :, None] * diff / d[:, :, None]
        g = dp.sum(axis=1).reshape(-1)
        a = np.maximum(rad_fixed - p, 0.0)
        b = np.maximum(rad_fixed - (1.0 - p), 0.0)
        g += (-2.0 * a + 2.0 * b).reshape(-1)
        return g

    try:
        res = minimize(pos_obj, pos.reshape(-1), jac=pos_grad, args=(rad,),
                       method='L-BFGS-B',
                       options={'maxiter': 80, 'maxfun': 1200, 'gtol': 1e-7})
        pos = res.x.reshape(N, 2)
    except Exception:
        pass

    rad = _max_radius(pos, rad)
    # one more position pass with the enlarged radii
    try:
        res = minimize(pos_obj, pos.reshape(-1), jac=pos_grad, args=(rad,),
                       method='L-BFGS-B',
                       options={'maxiter': 60, 'maxfun': 900, 'gtol': 1e-7})
        pos = res.x.reshape(N, 2)
    except Exception:
        pass
    rad = _max_radius(pos, rad)
    return pos, rad


def construct():
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before the 150s kill

    rng = np.random.default_rng(1234)
    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        rad = _max_radius(np.array(pos, dtype=float), np.array(rad, dtype=float))
        arr, s = _valid(list(zip(pos[:, 0], pos[:, 1], rad)))
        if arr is not None and s > best_score:
            best_score = s
            best = arr

    # ---------- deterministic initial positions (grid / hexagonal / rings) ----------
    def grid(side):
        cells = [((i + 0.5) / side, (j + 0.5) / side)
                 for i in range(side) for j in range(side)][:N]
        return np.array(cells, dtype=float)

    def hex_layout():
        pts = []
        rows = 5
        for row in range(rows):
            y = (row + 0.5) / rows
            xs = 1 / rows
            offset = 0.5 / rows if row % 2 else 0.0
            for col in range(rows):
                x = (col + 0.5) / rows + offset
                if 0 <= x <= 1:
                    pts.append((x, y))
            if len(pts) >= N:
                break
        return np.array(pts[:N], dtype=float)

    def rings():
        pts = [(0.5, 0.5)]
        for rad, cnt in [(0.2, 6), (0.35, 12), (0.44, 10)]:
            for k in range(cnt):
                a = 2 * np.pi * k / cnt
                pts.append((0.5 + rad * np.cos(a), 0.5 + rad * np.sin(a)))
            if len(pts) >= N:
                break
        return np.array(pts[:N], dtype=float)

    starts = []
    for side in range(3, 9):
        starts.append(grid(side))
    starts.append(hex_layout())
    starts.append(rings())
    # random uniform starts
    for _ in range(6):
        starts.append(rng.random((N, 2)) * 0.84 + 0.08)

    # ---------- force-directed annealing ----------
    def anneal(pos0, n_iter, t0, t1, k_rep, k_wall, damp, jitter, seed):
        rng_l = np.random.default_rng(seed)
        pos = np.array(pos0, dtype=float).copy()
        vel = np.zeros_like(pos)
        for it in range(n_iter):
            frac = it / max(1, n_iter - 1)
            t = t0 + (t1 - t0) * frac
            F = np.zeros_like(pos)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pos[i] - pos[j]
                    d = np.hypot(diff[0], diff[1])
                    if d < 1e-6:
                        # deterministic symmetric nudge to break exact coincidence
                        diff = np.array([1e-3 * ((i + j) % 7 - 3), 1e-3 * ((i - j) % 5 - 2)])
                        d = np.hypot(diff[0], diff[1])
                    f = k_rep / (d * d)
                    fvec = f * diff / d
                    F[i] += fvec
                    F[j] -= fvec
                # walls
                for ax in range(2):
                    if pos[i, ax] < 0.2:
                        F[i, ax] += k_wall * (0.2 - pos[i, ax])
                    if pos[i, ax] > 0.8:
                        F[i, ax] -= k_wall * (pos[i, ax] - 0.8)
            vel = damp * vel + 0.02 * F
            pos = pos + vel
            if jitter > 0.0:
                pos = pos + jitter * t * rng_l.standard_normal((N, 2))
            # clamp to square (keep centers interior)
            pos = np.clip(pos, 0.02, 0.98)
        return pos

    # ---------- main search loop ----------
    for r in range(len(starts)):
        if time.time() > DEADLINE - 5:
            break
        pos0 = starts[r]
        rad0 = np.full(N, 0.5 / max(3, int(np.sqrt(N))))
        rad0 = _max_radius(pos0, rad0)

        # a few annealing temperatures / restarts per start
        for trial in range(2):
            if time.time() > DEADLINE - 5:
                break
            seed = 1000 + 7 * r + trial
            p = anneal(pos0, n_iter=140, t0=1.0, t1=0.05,
                       k_rep=0.5, k_wall=1.2, damp=0.55,
                       jitter=0.05 if trial == 0 else 0.0, seed=seed)
            rad = _max_radius(p, rad0)
            # polish
            pp, rr = _polish(p, rad, DEADLINE)
            consider(pp, rr)
            if time.time() > DEADLINE - 5:
                break

    # ---------- final polish on the best found ----------
    if best is not None:
        pp, rr = _polish(best[:, :2], best[:, 2], DEADLINE)
        consider(pp, rr)

    if best is None:
        # extremely unlikely fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)

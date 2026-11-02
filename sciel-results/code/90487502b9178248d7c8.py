"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
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


def construct():
    t_start = time.time()
    DEADLINE = 140.0

    rng = np.random.default_rng(1234)
    best = None
    best_score = -1.0

    def consider(cands):
        nonlocal best, best_score
        # keep only valid (non-overlapping, in-square) circles
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
            return
        arr = np.array(valid[:N])
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr

    # ---- Phase A: grid initialisation, each circle at its maximal radius ----
    cells = []
    pos = None
    r = None
    for side in range(3, 9):
        if time.time() > DEADLINE:
            break
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
        cells = cells[:N]
        pos = np.array(cells, dtype=float)
        r = np.full(N, 0.5 / side - 1e-3)
        for _ in range(12):
            r = _max_radius(pos, r)
        consider(list(zip(pos[:, 0], pos[:, 1], r)))

    # ---- Phase B: penalty-based gradient optimisation ----
    def make_obj(x, lam):
        p = x[:2 * N].reshape(N, 2)
        rad = x[2 * N:]

        # pairwise overlap penalty
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        viol = np.maximum(rad[:, None] + rad[None, :] - d, 0.0)
        pair = np.sum(viol ** 2)
        # wall violation penalty
        w = np.maximum(rad - p, 0.0) + np.maximum(rad - (1.0 - p), 0.0)
        wall = np.sum(w ** 2)
        return -np.sum(rad) + lam * (pair + wall)

    def make_grad(x, lam):
        p = x[:2 * N].reshape(N, 2)
        rad = x[2 * N:]
        g = np.empty_like(x)
        g[:2 * N] = 0.0
        g[2 * N:] = -1.0  # d(-sum rad)/dr = -1

        # pairwise
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        d = np.maximum(d, EPS)
        diff = p[:, None, :] - p[None, :, :]
        viol = np.maximum(rad[:, None] + rad[None, :] - d, 0.0)
        # contribution 2*viol * (d rad_a + d rad_b - d d_ij)
        d_pair = 2.0 * viol
        d_dij = -np.einsum('ijk,ijk->ij', diff, d)  # d|diff|/d|diff|
        d_ra = d_pair * (1.0 + d_dij)
        d_rb = d_pair * (1.0 - d_dij)
        np.fill_diagonal(d_ra, 0.0)
        np.fill_diagonal(d_rb, 0.0)
        g[2 * N:] += lam * (d_ra.sum(axis=1) + d_rb.sum(axis=1))
        # position gradient from pairs: 2*viol * (d_ra - d_rb) * diff / d
        dp = (d_ra - d_rb)[:, :, None] * diff / d[:, :, None]
        np.fill_diagonal(dp.sum(axis=0), 0.0)
        g[:2 * N] += lam * dp.sum(axis=1).reshape(-1)

        # walls: penalty = sum max(rad - p,0)^2 + max(rad - (1-p),0)^2
        # d wrt rad: 2*max(rad-p,0) + 2*max(rad-(1-p),0)
        # d wrt p: -2*max(rad-p,0) + 2*max(rad-(1-p),0)
        a = np.maximum(rad - p, 0.0)
        b = np.maximum(rad - (1.0 - p), 0.0)
        g[2 * N:] += lam * 2.0 * (a + b)
        gp = lam * (-2.0 * a + 2.0 * b)
        g[:2 * N] += gp.reshape(-1)
        return g

    def _polish(cands):
        """Final polish: fix positions, maximize radii exactly; then fix radii, optimize positions."""
        if not cands or len(cands) < N:
            return cands
        arr = np.array(cands[:N])
        pos = arr[:, :2].copy()
        rad = arr[:, 2].copy()
        
        # Iterate: fix pos -> max rad, fix rad -> optimize pos
        for it in range(30):
            if time.time() > DEADLINE - 5:
                break
            # Maximize radii given fixed positions
            rad = _max_radius(pos, rad)
            
            # Optimize positions given fixed radii (move circles to reduce violations)
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
                d_dij = -np.einsum('ijk,ijk->ij', diff, d)
                dp = (d_pair)[:, :, None] * diff / d[:, :, None]
                np.fill_diagonal(dp.sum(axis=0), 0.0)
                g = dp.sum(axis=1).reshape(-1)
                a = np.maximum(rad_fixed - p, 0.0)
                b = np.maximum(rad_fixed - (1.0 - p), 0.0)
                g += (-2.0 * a + 2.0 * b).reshape(-1)
                return g
            
            res = minimize(pos_obj, pos.reshape(-1), jac=pos_grad, args=(rad,),
                          method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 3000, 'gtol': 1e-7})
            pos = res.x.reshape(N, 2)
        
        # Final radius computation
        rad = _max_radius(pos, rad)
        out = np.column_stack([pos, rad])
        return out.tolist()

    # More diverse initializations
    all_starts = []
    for side in range(3, 9):
        if time.time() > DEADLINE:
            break
        cells_s = []
        for i in range(side):
            for j in range(side):
                cells_s.append(((i + 0.5) / side, (j + 0.5) / side))
        cells_s = cells_s[:N]
        pos_s = np.array(cells_s, dtype=float)
        r_s = np.full(N, 0.5 / side - 1e-3)
        for _ in range(12):
            r_s = _max_radius(pos_s, r_s)
        all_starts.append(np.concatenate([pos_s, r_s]))
        
        # Random perturbations
        for k in range(3):
            if time.time() > DEADLINE:
                break
            p0 = pos_s + 0.08 * (rng.random((N, 2)) * 2 - 1)
            r0 = np.full(N, 0.5 / side)
            r0 = _max_radius(p0, r0)
            all_starts.append(np.concatenate([p0, r0]))

    # Random uniform initialization
    for k in range(5):
        if time.time() > DEADLINE:
            break
        p0 = rng.random((N, 2)) * 0.8 + 0.1
        r0 = np.full(N, 0.03)
        r0 = _max_radius(p0, r0)
        all_starts.append(np.concatenate([p0, r0]))

    if best is not None:
        all_starts.append(best.copy())

    for lam in (200.0, 50.0, 15.0, 5.0, 2.0, 0.7, 0.25, 0.1):
        if time.time() > DEADLINE:
            break
        for x0 in all_starts:
            if time.time() > DEADLINE:
                break
            try:
                maxiter = 500 if lam < 1.0 else 300
                gtol = 1e-7 if lam < 1.0 else 1e-5
                res = minimize(make_obj, x0, jac=make_grad, method='L-BFGS-B',
                               args=(lam,),
                               options={'maxiter': maxiter, 'maxfun': 6000,
                                        'gtol': gtol, 'maxls': 30, 'maxcor': 10})
                cands = list(zip(res.x[:2 * N].reshape(N, 2)[:, 0],
                                  res.x[:2 * N].reshape(N, 2)[:, 1],
                                  res.x[2 * N:]))
                polished = _polish(cands)
                consider(polished)
            except Exception:
                pass

    # Final polish on the best solution found
    if best is not None:
        cands = best.tolist()
        polished = _polish(cands)
        consider(polished)

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

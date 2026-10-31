"""Circle packing in unit square: maximize sum of radii for N=26 disjoint circles."""
import time
import numpy as np
from scipy.optimize import linprog

N = 26
T_LIMIT = 150.0          # hard wall-clock limit (s)
SAFETY = 8.0             # return this many seconds before the limit
DEADLINE = time.time() + (T_LIMIT - SAFETY)


def _softmin(a, t):
    # a: (n,); returns (softmin, dsoftmin/da)
    a = np.asarray(a)
    m = a.min()
    z = -t * (a - m)
    ez = np.exp(z - z.max())
    logsum = z.max() + np.log(ez.sum())
    val = m - (1.0 / t) * logsum
    grad = -(ez / ez.sum())
    return val, grad


def _clearances(P, r):
    # pairwise clearances d_ij - (r_i+r_j), and wall clearances r_i - d_i
    n = len(r)
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair = d - (r[None, :] + r[:, None])
    # wall distances
    dx = np.minimum(P[:, 0], 1.0 - P[:, 0])
    dy = np.minimum(P[:, 1], 1.0 - P[:, 1])
    wall = np.minimum(dx, dy)
    wallc = r - wall
    return pair, wallc


def _valid(P, r, tol=1e-9):
    # returns True if all circles are inside [0,1]^2 with disjoint interiors (with tol)
    n = len(r)
    if np.any(P < -tol) or np.any(P > 1.0 + tol) or np.any(r < -tol):
        return False
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    if np.any(d < (r[None, :] + r[:, None]) - tol):
        return False
    return True


def _score(P, r):
    return float(np.sum(r))


def _repair(P, r, iters=30):
    # Greedy shrink/shift to restore disjointness and wall feasibility.
    P = P.copy()
    r = r.copy()
    n = len(r)
    for _ in range(iters):
        changed = False
        # walls
        for i in range(n):
            dx = min(P[i, 0], 1.0 - P[i, 0])
            dy = min(P[i, 1], 1.0 - P[i, 1])
            dmin = min(dx, dy)
            if r[i] > dmin:
                r[i] = max(1e-6, dmin * 0.999999)
                changed = True
        # pairwise: shrink the smaller one and/or nudge
        for i in range(n):
            for j in range(i + 1, n):
                delta = P[i] - P[j]
                dist = np.linalg.norm(delta)
                need = r[i] + r[j]
                if dist < need - 1e-12:
                    # push apart and/or shrink
                    if dist < 1e-12:
                        u = np.array([1.0, 0.0])
                    else:
                        u = delta / dist
                    overlap = need - dist
                    # push each half, and shrink a bit
                    P[i] = P[i] + 0.5 * overlap * u
                    P[j] = P[j] - 0.5 * overlap * u
                    # clamp to walls and shrink if needed
                    for k in (i, j):
                        P[k, 0] = min(1.0, max(0.0, P[k, 0]))
                        P[k, 1] = min(1.0, max(0.0, P[k, 1]))
                        dx = min(P[k, 0], 1.0 - P[k, 0])
                        dy = min(P[k, 1], 1.0 - P[k, 1])
                        dmin = min(dx, dy)
                        if r[k] > dmin:
                            r[k] = max(1e-6, dmin * 0.999999)
                    # re-check and shrink further if still overlapping
                    dist2 = np.linalg.norm(P[i] - P[j])
                    need2 = r[i] + r[j]
                    if dist2 < need2 - 1e-12:
                        # shrink the larger radius
                        if r[i] >= r[j]:
                            r[i] = max(1e-6, r[i] - (need2 - dist2) * 0.6)
                        else:
                            r[j] = max(1e-6, r[j] - (need2 - dist2) * 0.6)
                    changed = True
        if not changed:
            break
    return P, r


def _init_grid():
    # A 5x5 grid gives 25 circles; we need 26. Use 5x5 and add one, then repair.
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    # 25 cells; add a 26th in the center-ish of the largest gap (use a small circle)
    # Place it at a corner-ish offset; repair will fix overlaps.
    cells.append([0.1, 0.1, 0.05])
    P = np.array([c[:2] for c in cells], dtype=float)
    r = np.array([c[2] for c in cells], dtype=float)
    return P, r


def _init_honeycomb():
    # Hexagonal packing: rows offset. For 26, use 5 rows with 5,6,5,5,5 pattern.
    rows = [5, 6, 5, 5, 5]
    cells = []
    row_y = [0.12, 0.31, 0.5, 0.69, 0.88]
    for ri, count in enumerate(rows):
        y = row_y[ri]
        for i in range(count):
            if count == 6:
                x = (i + 0.5) / 6
            else:
                # offset by half spacing
                x = (i + 0.5 + 0.25) / 5
                x = min(0.9, max(0.1, x))
            r = 0.08
            cells.append([x, y, r])
    P = np.array([c[:2] for c in cells], dtype=float)
    r = np.array([c[2] for c in cells], dtype=float)
    return P, r


def _optimize(P, r, t_start=8.0, t_end=200.0, lr=0.02, max_iters=4000):
    n = len(r)
    best_score = _score(P, r)
    best_P, best_r = P.copy(), r.copy()
    t = t_start
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        pair, wallc = _clearances(P, r)
        # softmin of all clearances (pairwise + wall)
        all_c = np.concatenate([pair.ravel(), wallc.ravel()])
        soft, grad = _softmin(all_c, t)
        # gradient of objective: maximize sum(r) + lambda * soft(clearances)
        # d soft / d P, d r
        dP = np.zeros_like(P)
        dr = np.zeros_like(r)
        # pairwise part
        gp = grad[:pair.size].reshape(n, n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                g = gp[i, j]
                if g == 0:
                    continue
                delta = P[i] - P[j]
                dist = np.linalg.norm(delta)
                if dist < 1e-12:
                    continue
                u = delta / dist
                dP[i] += g * u
                dP[j] -= g * u
                dr[i] -= g
                dr[j] -= g
        # wall part
        gw = grad[pair.size:]
        # wall clearance c_i = r_i - min(dx_i, dy_i); derivative w.r.t r_i is 1, w.r.t P_i is -d(min)/dP
        for i in range(n):
            g = gw[i]
            if g == 0:
                continue
            dr[i] += g
            # d min(dx,dy)/dP
            dx = np.minimum(P[i, 0], 1.0 - P[i, 0])
            dy = np.minimum(P[i, 1], 1.0 - P[i, 1])
            if dx < dy:
                ddx = 1.0 if P[i, 0] < 0.5 else -1.0
                ddy = 0.0
            else:
                ddx = 0.0
                ddy = 1.0 if P[i, 1] < 0.5 else -1.0
            dP[i, 0] -= g * ddx
            dP[i, 1] -= g * ddy
        # objective gradient: sum(r) gives +1 to each r; soft gives above
        dr_obj = dr + 1.0
        dP_obj = dP
        # projected gradient ascent
        P_new = P + lr * dP_obj
        r_new = r + lr * dr_obj
        # project to box and positivity
        P_new[:, 0] = np.clip(P_new[:, 0], 1e-6, 1.0 - 1e-6)
        P_new[:, 1] = np.clip(P_new[:, 1], 1e-6, 1.0 - 1e-6)
        r_new = np.clip(r_new, 1e-6, 0.5)
        # repair to ensure feasibility
        P_rep, r_rep = _repair(P_new, r_new, iters=8)
        s = _score(P_rep, r_rep)
        if s > best_score:
            best_score = s
            best_P, best_r = P_rep.copy(), r_rep.copy()
        P, r = P_rep, r_rep
        # anneal t
        t = min(t_end, t * 1.02)
        # decay learning rate slowly
        if it % 200 == 0:
            lr *= 0.9
    return best_P, best_r, best_score


def _polish(P, r, max_iters=500):
    # Exact local solve: maximize sum(r) s.t. clearances >= 0, using active-set + Newton.
    n = len(r)
    best_score = _score(P, r)
    best_P, best_r = P.copy(), r.copy()
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        pair, wallc = _clearances(P, r)
        # active set: constraints with clearance < 1e-7
        active_pair = np.argwhere(pair < 1e-7)
        active_wall = np.where(wallc < 1e-7)[0]
        if len(active_pair) == 0 and len(active_wall) == 0:
            # increase all radii a tiny bit
            r = np.minimum(0.5, r + 1e-5)
            P, r = _repair(P, r, iters=5)
            s = _score(P, r)
            if s > best_score:
                best_score = s
                best_P, best_r = P.copy(), r.copy()
            continue
        # Build KKT: maximize sum(r) s.t. c_k(x) >= 0 for active k.
        # Lagrangian: sum(r) - sum_k mu_k * c_k(x), mu_k >= 0.
        # Stationarity: d/dP: sum over active pairs of mu * (p_i - p_j)/d_ij ... 
        # This is a QP-like; use a simple projected gradient with exact step.
        # Compute gradient of objective + active constraints
        dP = np.zeros_like(P)
        dr = np.ones_like(r)  # d sum(r)/dr
        for (i, j) in active_pair:
            delta = P[i] - P[j]
            dist = np.linalg.norm(delta)
            if dist < 1e-12:
                continue
            u = delta / dist
            dP[i] += u
            dP[j] -= u
            dr[i] -= 1.0
            dr[j] -= 1.0
        for i in active_wall:
            dx = np.minimum(P[i, 0], 1.0 - P[i, 0])
            dy = np.minimum(P[i, 1], 1.0 - P[i, 1])
            if dx < dy:
                ddx = 1.0 if P[i, 0] < 0.5 else -1.0
                ddy = 0.0
            else:
                ddx = 0.0
                ddy = 1.0 if P[i, 1] < 0.5 else -1.0
            dr[i] += 1.0
            dP[i, 0] -= ddx
            dP[i, 1] -= ddy
        # Line search: try small steps
        best_step_score = -1
        best_step_P, best_step_r = P.copy(), r.copy()
        for step in [0.01, 0.005, 0.002, 0.001, 0.0005]:
            P_try = P + step * dP
            r_try = r + step * dr
            P_try[:, 0] = np.clip(P_try[:, 0], 1e-6, 1.0 - 1e-6)
            P_try[:, 1] = np.clip(P_try[:, 1], 1e-6, 1.0 - 1e-6)
            r_try = np.clip(r_try, 1e-6, 0.5)
            P_rep, r_rep = _repair(P_try, r_try, iters=5)
            s = _score(P_rep, r_rep)
            if s > best_step_score:
                best_step_score = s
                best_step_P, best_step_r = P_rep.copy(), r_rep.copy()
        if best_step_score > best_score:
            best_score = best_step_score
            best_P, best_r = best_step_P, best_step_r
            P, r = best_step_P, best_step_r
        else:
            # try a pure radius increase
            r_try = np.minimum(0.5, r + 1e-4)
            P_rep, r_rep = _repair(P, r_try, iters=5)
            s = _score(P_rep, r_rep)
            if s > best_score:
                best_score = s
                best_P, best_r = P_rep.copy(), r_rep.copy()
                P, r = P_rep, r_rep
            else:
                break
    return best_P, best_r, best_score


def construct():
    t0 = time.time()
    # Try multiple initializations and keep the best
    inits = [
        _init_grid(),
        _init_honeycomb(),
    ]
    # Add a random-ish perturbation init
    rng = np.random.default_rng(42)
    P_rand = rng.uniform(0.1, 0.9, size=(N, 2))
    r_rand = np.full(N, 0.08)
    inits.append((P_rand, r_rand))

    best_P, best_r, best_score = None, None, -1.0
    for P0, r0 in inits:
        if time.time() > DEADLINE:
            break
        P0, r0 = _repair(P0.copy(), r0.copy(), iters=20)
        P_opt, r_opt, s = _optimize(P0, r0, t_start=8.0, t_end=150.0, lr=0.015, max_iters=2500)
        if s > best_score:
            best_score = s
            best_P, best_r = P_opt, r_opt
        if time.time() > DEADLINE:
            break

    if best_P is None:
        P0, r0 = _init_grid()
        best_P, best_r = _repair(P0, r0, iters=20)

    # Final polish
    best_P, best_r, _ = _polish(best_P, best_r, max_iters=300)

    # Final validity repair
    best_P, best_r = _repair(best_P, best_r, iters=20)

    # Ensure strict validity: if any residual overlap, shrink
    if not _valid(best_P, best_r, tol=1e-9):
        # Aggressive shrink
        for _ in range(50):
            pair, wallc = _clearances(best_P, best_r)
            if np.min(pair) > -1e-12 and np.min(wallc) > -1e-12:
                break
            # Find worst violation
            all_c = np.concatenate([pair.ravel(), wallc.ravel()])
            idx = np.argmin(all_c)
            if idx < pair.size:
                i, j = np.unravel_index(idx, pair.shape)
                # shrink both
                best_r[i] *= 0.99
                best_r[j] *= 0.99
            else:
                i = idx - pair.size
                best_r[i] *= 0.99
        best_P, best_r = _repair(best_P, best_r, iters=20)

    # Final check
    if not _valid(best_P, best_r, tol=1e-9):
        # Last resort: shrink all radii until valid
        scale = 1.0
        for _ in range(200):
            if _valid(best_P, best_r * scale, tol=1e-9):
                break
            scale *= 0.98
        best_r = best_r * scale

    out = np.column_stack([best_P, best_r])
    return out

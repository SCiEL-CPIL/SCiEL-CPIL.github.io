import numpy as np
import time

N = 21
T0 = time.time()
DEADLINE = T0 + 130.0  # safety margin below 150s kill

def _remaining():
    return max(0.0, DEADLINE - time.time())


def _objective(x):
    # x = [a, then per circle: cx, cy, r]
    a = x[0]
    W = a / 2.0
    H = 2.0 - a
    r = x[3::3]
    # sum of radii minus penalty for negative aspect
    return -np.sum(r) + 1e4 * max(0.0, 0.05 - a)


def _objective_grad(x):
    g = np.zeros_like(x)
    a = x[0]
    g[0] = 1e4 if a < 0.05 else 0.0
    g[3::3] = -1.0
    return g


def _constraints(x):
    a = x[0]
    W = a / 2.0
    H = 2.0 - a
    cx = x[1::3]
    cy = x[2::3]
    r = x[3::3]
    cons = []
    # aspect lower bound
    cons.append(a - 0.05)
    # wall clearance
    cons.append(cx - r)
    cons.append(W - cx - r)
    cons.append(cy - r)
    cons.append(H - cy - r)
    # circle-circle
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            cons.append(np.sqrt(dx * dx + dy * dy) - r[i] - r[j])
    return np.concatenate(cons)


def _constraint_jac(x):
    a = x[0]
    W = a / 2.0
    H = 2.0 - a
    cx = x[1::3]
    cy = x[2::3]
    r = x[3::3]
    m = len(_constraints(x))
    J = np.zeros((m, x.size))
    row = 0
    # aspect
    J[row, 0] = 1.0
    row += 1
    # cx - r
    for i in range(N):
        J[row, 1 + 3 * i] = 1.0
        J[row, 3 + 3 * i] = -1.0
        row += 1
    # W - cx - r
    for i in range(N):
        J[row, 0] += -0.5
        J[row, 1 + 3 * i] = -1.0
        J[row, 3 + 3 * i] = -1.0
        row += 1
    # cy - r
    for i in range(N):
        J[row, 2 + 3 * i] = 1.0
        J[row, 3 + 3 * i] = -1.0
        row += 1
    # H - cy - r
    for i in range(N):
        J[row, 0] += -1.0
        J[row, 2 + 3 * i] = -1.0
        J[row, 3 + 3 * i] = -1.0
        row += 1
    # circle-circle
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy) + 1e-12
            J[row, 1 + 3 * i] = dx / d
            J[row, 2 + 3 * i] = dy / d
            J[row, 1 + 3 * j] = -dx / d
            J[row, 2 + 3 * j] = -dy / d
            J[row, 3 + 3 * i] = -1.0
            J[row, 3 + 3 * j] = -1.0
            row += 1
    return J


def _seed_grid():
    # square-ish grid inside unit square, then scaled later
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
    cells = cells[:N]
    return cells


def _seed_random(rng):
    cells = []
    for _ in range(N):
        cells.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(0.03, 0.09)])
    return cells


def _to_x(a, cells):
    W = a / 2.0
    H = 2.0 - a
    x = np.zeros(1 + 3 * N)
    x[0] = a
    for i, (cx, cy, r) in enumerate(cells):
        # scale grid coords (unit square) to rectangle
        x[1 + 3 * i] = cx * W
        x[2 + 3 * i] = cy * H
        x[3 + 3 * i] = r * min(W, H)
    return x


def _from_x(x):
    a = x[0]
    W = a / 2.0
    H = 2.0 - a
    cx = x[1::3]
    cy = x[2::3]
    r = x[3::3]
    return cx, cy, r, a, W, H


def _validity(x):
    cx, cy, r, a, W, H = _from_x(x)
    if a <= 0.05 or H <= 0.05:
        return False
    if np.any(r <= 1e-9):
        return False
    if np.any(cx < -1e-7) or np.any(cy < -1e-7):
        return False
    if np.any(cx + r > W + 1e-7) or np.any(cy + r > H + 1e-7):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d < r[i] + r[j] - 1e-7:
                return False
    return True


def _newton_polish(x, max_iter=80, tol=1e-9):
    # Build active contact graph: pairs at (or near) contact + wall contacts.
    # Solve for radii given positions via linear system on active constraints,
    # then iterate positions to satisfy active circle-circle distances.
    cx, cy, r, a, W, H = _from_x(x)
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()

    # Determine active pairs: those with distance close to sum of radii
    def active_pairs():
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < r[i] + r[j] + 1e-6:
                    pairs.append((i, j, d))
        return pairs

    for _ in range(max_iter):
        pairs = active_pairs()
        if not pairs:
            break
        # Build linear system: for each active pair, (cx_i-cx_j)^2+(cy_i-cy_j)^2 = (r_i+r_j)^2
        # Linearize around current positions: treat positions fixed, solve radii
        # Actually solve for radii given fixed positions: r_i + r_j = d_ij
        # Build A r = b for active pairs
        A = np.zeros((len(pairs), N))
        b = np.zeros(len(pairs))
        for k, (i, j, d) in enumerate(pairs):
            A[k, i] = 1.0
            A[k, j] = 1.0
            b[k] = d
        # Add wall constraints as soft: prefer circles touching at least one wall
        # Solve least squares with positivity
        try:
            r_new, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        # Enforce positivity and wall feasibility
        r_new = np.maximum(r_new, 1e-6)
        # Check wall feasibility; shrink if needed
        feasible = True
        for i in range(N):
            if cx[i] - r_new[i] < -1e-6 or cx[i] + r_new[i] > W + 1e-6:
                feasible = False
                r_new[i] = min(cx[i], W - cx[i])
            if cy[i] - r_new[i] < -1e-6 or cy[i] + r_new[i] > H + 1e-6:
                feasible = False
                r_new[i] = min(r_new[i], min(cy[i], H - cy[i]))
        if not np.all(r_new > 1e-6):
            break
        # Update radii
        diff = np.max(np.abs(r_new - r))
        r = r_new
        # Now adjust positions to satisfy active circle-circle distances exactly:
        # For each active pair, project positions so distance = r_i + r_j
        changed = False
        for (i, j, d) in active_pairs():
            target = r[i] + r[j]
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            cur = np.hypot(dx, dy)
            if cur < 1e-12:
                continue
            # Move both circles along the line to achieve target
            scale = target / cur
            midx = (cx[i] + cx[j]) / 2.0
            midy = (cy[i] + cy[j]) / 2.0
            cx[i] = midx + (cx[i] - midx) * scale
            cy[i] = midy + (cy[i] - midy) * scale
            cx[j] = midx + (cx[j] - midx) * scale
            cy[j] = midy + (cy[j] - midy) * scale
            changed = True
        # Re-check wall feasibility after position moves; nudge inward
        for i in range(N):
            if cx[i] - r[i] < 0:
                cx[i] = r[i]
            if cx[i] + r[i] > W:
                cx[i] = W - r[i]
            if cy[i] - r[i] < 0:
                cy[i] = r[i]
            if cy[i] + r[i] > H:
                cy[i] = H - r[i]
        if not changed and diff < tol:
            break

    # Build final x
    x_new = x.copy()
    x_new[1::3] = cx
    x_new[2::3] = cy
    x_new[3::3] = r
    return x_new


def _score(x):
    if not _validity(x):
        return -np.inf
    return float(np.sum(x[3::3]))


def construct():
    rng = np.random.default_rng(12345)
    best_x = None
    best_score = -np.inf

    # Build initial candidates
    candidates = []
    # Grid seed
    cells0 = _seed_grid()
    for a_try in [1.0, 1.2, 0.8, 1.5, 0.7]:
        candidates.append(_to_x(a_try, cells0))
    # Random seeds
    for _ in range(30):
        a_try = rng.uniform(0.6, 1.6)
        cells = _seed_random(rng)
        candidates.append(_to_x(a_try, cells))

    # Solve each candidate with SLSQP
    for x0 in candidates:
        if _remaining() < 5.0:
            break
        try:
            res = _slsqp_solve(x0)
            if res is not None:
                sc = _score(res)
                if sc > best_score:
                    best_score = sc
                    best_x = res.copy()
        except Exception:
            continue

    # Polish the best candidate
    if best_x is not None:
        if _remaining() > 5.0:
            polished = _newton_polish(best_x)
            sc = _score(polished)
            if sc > best_score:
                best_score = sc
                best_x = polished

    # If we still have time, do a second round of random restarts focused on best aspect
    if best_x is not None and _remaining() > 15.0:
        a_best = best_x[0]
        for _ in range(40):
            if _remaining() < 5.0:
                break
            a_try = max(0.6, min(1.6, a_best + rng.normal(0, 0.2)))
            cells = _seed_random(rng)
            x0 = _to_x(a_try, cells)
            try:
                res = _slsqp_solve(x0)
                if res is not None:
                    sc = _score(res)
                    if sc > best_score:
                        best_score = sc
                        best_x = res.copy()
            except Exception:
                continue
        if best_x is not None and _remaining() > 5.0:
            polished = _newton_polish(best_x)
            sc = _score(polished)
            if sc > best_score:
                best_score = sc
                best_x = polished

    if best_x is None:
        # Fallback: return the grid solution (valid, score ~1.0)
        cells = _seed_grid()
        a = 1.0
        W = a / 2.0
        H = 2.0 - a
        out = np.zeros((N, 3))
        for i, (cx, cy, r) in enumerate(cells):
            out[i] = [cx * W, cy * H, r * min(W, H)]
        return out

    cx, cy, r, a, W, H = _from_x(best_x)
    out = np.column_stack([cx, cy, r])
    return out


def _slsqp_solve(x0):
    from scipy.optimize import minimize
    cons = [{'type': 'ineq', 'fun': _constraints, 'jac': _constraint_jac}]
    res = minimize(_objective, x0, method='SLSQP', jac=_objective_grad,
                   constraints=cons, bounds=[(0.05, 1.6)] + [(0, None)] * (3 * N),
                   options={'maxiter': 200, 'ftol': 1e-8})
    if res.success or _score(res.x) > _score(x0):
        return res.x
    return None

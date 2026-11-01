"""Circle packing in perimeter-4 rectangle: Direct-constraint NLP with multi-start and local escape."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_TIME_LIMIT = 145.0


def _build_constraints():
    """Build constraint dicts for SLSQP with all pairwise and wall constraints."""
    cons = []
    # Wall constraints: x_i >= r_i, x_i + r_i <= w, y_i >= r_i, y_i + r_i <= h = 2 - w
    for i in range(N):
        xi, yi, ri = 3*i, 3*i+1, 3*i+2
        # x_i - r_i >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda z, a=xi, b=ri: z[a] - z[b],
                     'jac': lambda z, a=xi, b=ri: np.concatenate([np.eye(3*N+1)[a], np.zeros(1)]) - np.concatenate([np.eye(3*N+1)[b], np.zeros(1)])})
        # w - x_i - r_i >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda z, a=xi, b=ri: z[-1] - z[a] - z[b],
                     'jac': lambda z, a=xi, b=ri: np.concatenate([np.zeros(3*N), 1.0]) - np.concatenate([np.eye(3*N+1)[a], np.zeros(1)]) - np.concatenate([np.eye(3*N+1)[b], np.zeros(1)])})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda z, a=yi, b=ri: z[a] - z[b],
                     'jac': lambda z, a=yi, b=ri: np.concatenate([np.eye(3*N+1)[a], np.zeros(1)]) - np.concatenate([np.eye(3*N+1)[b], np.zeros(1)])})
        # (2 - w) - y_i - r_i >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda z, a=yi, b=ri: 2.0 - z[-1] - z[a] - z[b],
                     'jac': lambda z, a=yi, b=ri: np.concatenate([np.zeros(3*N), -1.0]) - np.concatenate([np.eye(3*N+1)[a], np.zeros(1)]) - np.concatenate([np.eye(3*N+1)[b], np.zeros(1)])})
    # Pairwise: dist(i,j) - r_i - r_j >= 0
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = 3*i, 3*i+1, 3*i+2
            xj, yj, rj = 3*j, 3*j+1, 3*j+2
            cons.append({'type': 'ineq',
                         'fun': lambda z, a=xi, b=yi, c=xj, d=yj, e=ri, f=rj:
                             np.sqrt((z[a]-z[c])**2 + (z[b]-z[d])**2) - z[e] - z[f],
                         'jac': lambda z, a=xi, b=yi, c=xj, d=yj, e=ri, f=rj:
                             _pair_jac(z, a, b, c, d, e, f)})
    return cons


def _pair_jac(z, a, b, c, d, e, f):
    """Jacobian for pairwise distance constraint."""
    dx = z[a] - z[c]
    dy = z[b] - z[d]
    dist = np.sqrt(dx*dx + dy*dy) + 1e-15
    jac = np.zeros(3*N+1)
    jac[a] = dx / dist
    jac[b] = dy / dist
    jac[c] = -dx / dist
    jac[d] = -dy / dist
    jac[e] = -1.0
    jac[f] = -1.0
    return jac


def _objective(z):
    """Maximize sum of radii (minimize negative)."""
    r = z[2::3][:N]
    return -np.sum(r)


def _is_valid(x, y, r, w, h, tol=1e-9):
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i+1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            rs = r[i] + r[j]
            if d2 < rs*rs - tol:
                return False
    return True


def _force_relax(x, y, r, w, h, max_iter=800, max_time=20.0, start_time=None):
    """Force-directed relaxation to resolve overlaps and push to boundaries."""
    if start_time is None:
        start_time = time.time()
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    dt = 0.003
    min_dt = 1e-5
    for it in range(max_iter):
        if time.time() - start_time > max_time:
            break
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy) + 1e-10
                min_d = r[i] + r[j]
                if d < min_d:
                    force = (min_d - d) / d
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
                    fr[i] -= force * 0.05
                    fr[j] -= force * 0.05
        # Wall forces
        fx -= 10.0 * np.maximum(r - x, 0)
        fx += 10.0 * np.maximum(x + r - w, 0)
        fy -= 10.0 * np.maximum(r - y, 0)
        fy += 10.0 * np.maximum(y + r - h, 0)
        x += dt * fx
        y += dt * fy
        r += dt * fr
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        r = np.maximum(r, 0.01)
        max_f = max(np.max(np.abs(fx)), np.max(np.abs(fy)), np.max(np.abs(fr)))
        if max_f < 1e-12:
            break
        elif max_f > 5:
            dt = max(min_dt, dt * 0.5)
        else:
            dt = min(0.01, dt * 1.03)
    return x, y, r


def _greedy_expand(x, y, r, w, h, max_rounds=30, time_budget=10.0):
    """Expand each circle to its max possible radius given current positions."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    start = time.time()
    for _ in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return x, y, r


def _local_escape(x, y, r, w, h, n_perturb=5, scale=0.15, max_time=8.0):
    """Perturb a random subset of circles and re-relax to escape local minima."""
    x_new = x.copy()
    y_new = y.copy()
    r_new = r.copy()
    idx = np.random.choice(N, size=min(n_perturb, N), replace=False)
    for i in idx:
        x_new[i] += np.random.uniform(-scale, scale)
        y_new[i] += np.random.uniform(-scale, scale)
        r_new[i] *= np.random.uniform(0.7, 1.0)
    x_new = np.clip(x_new, 0, w)
    y_new = np.clip(y_new, 0, h)
    r_new = np.maximum(r_new, 0.01)
    x_new, y_new, r_new = _force_relax(x_new, y_new, r_new, w, h, max_iter=400, max_time=max_time)
    return x_new, y_new, r_new


def _slsqp_solve(x, y, r, w, h, constraints, maxiter=400, max_time=15.0, start_time=None):
    """Run SLSQP with direct constraints."""
    if start_time is None:
        start_time = time.time()
    z0 = np.zeros(3*N + 1)
    z0[0::3][:N] = x
    z0[1::3][:N] = y
    z0[2::3][:N] = r
    z0[-1] = w
    bounds = [(0, 2.0)] * (3*N) + [(0.3, 1.7)]
    try:
        res = minimize(_objective, z0, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        if res.success or (hasattr(res, 'x') and np.all(np.isfinite(res.x))):
            return res.x
    except Exception:
        pass
    return z0


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    constraints = _build_constraints()
    
    best_value = 0.0
    best_x = best_y = best_r = best_w = best_h = None

    def evaluate(x, y, r, w, h):
        nonlocal best_value, best_x, best_y, best_r, best_w, best_h
        if not _is_valid(x, y, r, w, h):
            return False
        val = float(np.sum(r))
        if val > best_value:
            best_value = val
            best_x, best_y, best_r, best_w, best_h = x.copy(), y.copy(), r.copy(), w, h
            return True
        return False

    # Multi-start over aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.1, 0.9]
    rng = np.random.default_rng(12345)

    for w0 in aspect_ratios:
        if time.time() > deadline - 25:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Grid-based initial placement
        cols = int(np.ceil(np.sqrt(N * w0 / h0)))
        rows = int(np.ceil(N / cols))
        cw = w0 / cols
        ch = h0 / rows
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, cols)
            x[i] = (col + 0.5) * cw
            y[i] = (row + 0.5) * ch
            r[i] = min(cw, ch) / 2.0 * 0.9

        # Force relax to resolve overlaps
        x, y, r = _force_relax(x, y, r, w0, h0, max_iter=500, max_time=8.0, start_time=start_time)
        
        # SLSQP with direct constraints
        if time.time() > deadline - 20:
            break
        z = _slsqp_solve(x, y, r, w0, h0, constraints, maxiter=300, max_time=12.0, start_time=start_time)
        x_sol = z[0::3][:N]
        y_sol = z[1::3][:N]
        r_sol = z[2::3][:N]
        w_sol = z[-1]
        h_sol = 2.0 - w_sol
        evaluate(x_sol, y_sol, r_sol, w_sol, h_sol)

        # Greedy expand after SLSQP
        if best_value > 0 and time.time() < deadline - 15:
            x_e, y_e, r_e = _greedy_expand(x_sol, y_sol, r_sol, w_sol, h_sol, max_rounds=20, time_budget=5.0)
            evaluate(x_e, y_e, r_e, w_sol, h_sol)

        # Local escape: perturb and re-solve
        if time.time() < deadline - 12:
            x_esc, y_esc, r_esc = _local_escape(x_sol, y_sol, r_sol, w_sol, h_sol, n_perturb=5, scale=0.12, max_time=5.0)
            z2 = _slsqp_solve(x_esc, y_esc, r_esc, w_sol, h_sol, constraints, maxiter=200, max_time=6.0, start_time=start_time)
            x2 = z2[0::3][:N]
            y2 = z2[1::3][:N]
            r2 = z2[2::3][:N]
            w2 = z2[-1]
            h2 = 2.0 - w2
            evaluate(x2, y2, r2, w2, h2)
            if best_value > 0 and time.time() < deadline - 10:
                x_e2, y_e2, r_e2 = _greedy_expand(x2, y2, r2, w2, h2, max_rounds=15, time_budget=4.0)
                evaluate(x_e2, y_e2, r_e2, w2, h2)

    # Final polish on best result
    if best_value > 0 and best_x is not None and time.time() < deadline - 10:
        # Force relax + greedy expand
        x_f, y_f, r_f = _force_relax(best_x, best_y, best_r, best_w, best_h, max_iter=300, max_time=5.0, start_time=start_time)
        x_f, y_f, r_f = _greedy_expand(x_f, y_f, r_f, best_w, best_h, max_rounds=30, time_budget=5.0)
        evaluate(x_f, y_f, r_f, best_w, best_h)

        # One more SLSQP on polished result
        if time.time() < deadline - 8:
            z_f = _slsqp_solve(best_x, best_y, best_r, best_w, best_h, constraints, maxiter=200, max_time=5.0, start_time=start_time)
            xf = z_f[0::3][:N]
            yf = z_f[1::3][:N]
            rf = z_f[2::3][:N]
            wf = z_f[-1]
            hf = 2.0 - wf
            evaluate(xf, yf, rf, wf, hf)
            if best_value > 0 and time.time() < deadline - 5:
                xg, yg, rg = _greedy_expand(xf, yf, rf, wf, hf, max_rounds=20, time_budget=3.0)
                evaluate(xg, yg, rg, wf, hf)

    # Fallback
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Safety shrink
    r_out = best_r.copy()
    for _ in range(10):
        r_trial = r_out * (1 - 1e-10)
        if _is_valid(best_x, best_y, r_trial, best_w, best_h, tol=1e-12):
            r_out = r_trial
        else:
            break

    return np.column_stack((best_x, best_y, r_out))

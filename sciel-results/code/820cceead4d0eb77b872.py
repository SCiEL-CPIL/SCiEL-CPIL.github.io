"""Circle packing in perimeter-4 rectangle: Direct-constraint NLP + Radial Inflate & Slide."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_TIME_LIMIT = 143.0


def _objective(z):
    """Maximize sum of radii (minimize negative)."""
    r = z[2::3][:N]
    return -np.sum(r)


def _objective_grad(z):
    """Gradient of -sum(r)."""
    g = np.zeros_like(z)
    g[2::3][:N] = -1.0
    return g


def _build_constraints():
    """
    Build constraint dicts for SLSQP.
    Variable layout: z[3i]=x_i, z[3i+1]=y_i, z[3i+2]=r_i, z[-1]=w, h=2-w.
    Inequality constraints: fun(z) >= 0.
    """
    cons = []

    # Wall constraints per circle
    for i in range(N):
        xi, yi, ri = 3 * i, 3 * i + 1, 3 * i + 2
        # x_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda z, a=xi, b=ri: z[a] - z[b],
                     'jac': lambda z, a=xi, b=ri: np.concatenate([[1.0], np.zeros(a), [-1.0], np.zeros(3*N - a - 1)])})
        # x_i + r_i <= w  =>  w - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, a=xi, b=ri: z[-1] - z[a] - z[b],
                     'jac': lambda z, a=xi, b=ri: np.concatenate([np.zeros(a), [-1.0], np.zeros(3*N - a - 1), [-1.0], [1.0]])})
        # y_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda z, a=yi, b=ri: z[a] - z[b],
                     'jac': lambda z, a=yi, b=ri: np.concatenate([[1.0], np.zeros(a), [-1.0], np.zeros(3*N - a - 1)])})
        # y_i + r_i <= h = 2 - w  =>  2 - w - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, a=yi, b=ri: 2.0 - z[-1] - z[a] - z[b],
                     'jac': lambda z, a=yi, b=ri: np.concatenate([np.zeros(a), [-1.0], np.zeros(3*N - a - 1), [-1.0], [-1.0]])})

    # Pairwise non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    # Equivalent: d^2 - (r_i+r_j)^2 >= 0  (but this is non-smooth at d=0)
    # Use: d - (r_i+r_j) >= 0 with d = sqrt(...)
    # SLSQP handles this; provide jac numerically-free via chain rule.
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ri = 3 * i, 3 * i + 1, 3 * i + 2
            xj, yj, rj = 3 * j, 3 * j + 1, 3 * j + 2

            def fun(z, a=xi, b=yi, c=ri, d=xj, e=yj, f=rj):
                dx = z[a] - z[d]
                dy = z[b] - z[e]
                dist = np.sqrt(dx * dx + dy * dy)
                return dist - z[c] - z[f]

            def jac(z, a=xi, b=yi, c=ri, d=xj, e=yj, f=rj):
                dx = z[a] - z[d]
                dy = z[b] - z[e]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-15
                g = np.zeros(3 * N + 1)
                g[a] = dx / dist
                g[b] = dy / dist
                g[d] = -dx / dist
                g[e] = -dy / dist
                g[c] = -1.0
                g[f] = -1.0
                return g

            cons.append({'type': 'ineq', 'fun': fun, 'jac': jac})

    return cons


def _max_radius_analytic(x, y, r, w, h, i):
    """Compute the maximum radius circle i can have given all other circles' positions."""
    # Wall limits
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    # Neighbor limits
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _radial_inflate_and_slide(z, constraints, max_time=30.0, start_time=None):
    """
    Repeatedly: for each circle, compute max radius analytically, set it,
    then run a short SLSQP to re-relax positions. This "inflates" the packing
    outward and lets the optimizer find new positions.
    """
    if start_time is None:
        start_time = time.time()

    z = z.copy().astype(float)
    n = N
    total_rounds = 0

    while time.time() - start_time < max_time and total_rounds < 40:
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2:
            break

        # Compute max radii
        r_new = np.zeros(n)
        for i in range(n):
            r_new[i] = _max_radius_analytic(x, y, r, w, h, i)

        # If no improvement, try a small perturbation to escape
        if np.all(r_new <= r + 1e-10):
            # Perturb positions slightly
            rng = np.random.default_rng(total_rounds)
            z_pert = z.copy()
            z_pert[0::3][:n] += rng.normal(0, 0.005, n)
            z_pert[1::3][:n] += rng.normal(0, 0.005, n)
            # Clamp
            z_pert[0::3][:n] = np.clip(z_pert[0::3][:n], 0.01, w - 0.01)
            z_pert[1::3][:n] = np.clip(z_pert[1::3][:n], 0.01, h - 0.01)
            z = z_pert
            total_rounds += 1
            continue

        # Set new radii
        z[2::3][:n] = r_new

        # Run short SLSQP to re-relax
        remaining = max_time - (time.time() - start_time)
        if remaining < 2:
            break

        try:
            res = minimize(_objective, z, jac=_objective_grad,
                           method='SLSQP', bounds=_get_bounds(),
                           constraints=constraints,
                           options={'maxiter': int(min(200, max(50, remaining * 20))),
                                    'ftol': 1e-10, 'maxls': 50})
            if res.success or res.fun < _objective(z):
                z = res.x
        except Exception:
            pass

        total_rounds += 1

    return z


def _get_bounds():
    """Bounds for all variables."""
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0))   # x
        bounds.append((0.0, 2.0))   # y
        bounds.append((0.001, 1.0)) # r
    bounds.append((0.25, 1.75))     # w
    return bounds


def _is_valid(x, y, r, w, h, tol=1e-8):
    """Check validity."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(N):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _value(x, y, r):
    return float(np.sum(r))


def _check_and_get_value(z, tol=1e-8):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    if _is_valid(x, y, r, w, h, tol):
        return True, _value(x, y, r)
    return False, 0.0


def _slsqp_optimize(z0, constraints, max_time=30.0, start_time=None, maxiter=300):
    """Run SLSQP with direct constraints."""
    if start_time is None:
        start_time = time.time()
    remaining = max_time - (time.time() - start_time)
    if remaining < 1:
        return z0
    try:
        res = minimize(_objective, z0, jac=_objective_grad,
                       method='SLSQP', bounds=_get_bounds(),
                       constraints=constraints,
                       options={'maxiter': int(min(maxiter, max(50, remaining * 30))),
                                'ftol': 1e-12, 'maxls': 80})
        return res.x
    except Exception:
        return z0


def _generate_initial_guesses():
    """Generate diverse initial configurations."""
    guesses = []
    rng = np.random.default_rng(123)

    # Structured: grid-like with different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        # Place circles in a grid pattern
        cols = max(1, int(np.ceil(np.sqrt(N * w0 / h0))))
        rows = max(1, int(np.ceil(N / cols)))
        cw = w0 / cols
        ch = h0 / rows
        z = np.zeros(3 * N + 1)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5) * cw
            cy = (row + 0.5) * ch
            r0 = min(cw, ch) / 2.0 * 0.95
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

    # Random starts
    for seed in range(8):
        rng2 = np.random.default_rng(seed)
        w0 = rng2.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng2.uniform(0.04, 0.10)
            cx = rng2.uniform(r0, w0 - r0)
            cy = rng2.uniform(r0, h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

    return guesses


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    constraints = _build_constraints()
    best_value = 0.0
    best_result = None

    guesses = _generate_initial_guesses()

    for idx, z0 in enumerate(guesses):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        if remaining < 15:
            break

        # Quick SLSQP from this start
        alloc = min(remaining * 0.6, 25.0)
        z_opt = _slsqp_optimize(z0, constraints, max_time=alloc,
                                start_time=start_time + elapsed, maxiter=250)

        valid, val = _check_and_get_value(z_opt)
        if valid and val > best_value:
            best_value = val
            best_result = z_opt.copy()

        # Radial inflate & slide from this result
        if valid and remaining > 10:
            z_inflated = _radial_inflate_and_slide(z_opt, constraints,
                                                   max_time=min(remaining * 0.3, 20.0),
                                                   start_time=start_time + elapsed)
            valid2, val2 = _check_and_get_value(z_inflated)
            if valid2 and val2 > best_value:
                best_value = val2
                best_result = z_inflated.copy()

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 10:
        remaining = deadline - time.time()

        # Another round of radial inflate
        z_final = _radial_inflate_and_slide(best_result, constraints,
                                            max_time=min(remaining * 0.5, 25.0),
                                            start_time=time.time())
        valid, val = _check_and_get_value(z_final)
        if valid and val > best_value:
            best_value = val
            best_result = z_final.copy()

        # Final SLSQP tight
        if time.time() < deadline - 3:
            z_tight = _slsqp_optimize(best_result, constraints,
                                      max_time=min(deadline - time.time() - 1, 10.0),
                                      start_time=time.time(), maxiter=200)
            valid, val = _check_and_get_value(z_tight)
            if valid and val > best_value:
                best_value = val
                best_result = z_tight.copy()

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Safety: tiny shrink to ensure strict validity
    x = best_result[0::3][:N].copy()
    y = best_result[1::3][:N].copy()
    r = best_result[2::3][:N].copy()
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(30):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))

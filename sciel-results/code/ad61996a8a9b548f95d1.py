import numpy as np
import time
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _penalty(params, smooth=True):
    # params: [x1,y1,r1, ..., xN,yN,rN, w]
    coords = params[:3*N]
    w = params[3*N]
    h = 2.0 - w

    x = coords[0::3]
    y = coords[1::3]
    r = np.maximum(coords[2::3], 0.0)

    # Boundary + box violations
    pen_boundary = (
        np.sum(np.maximum(0.0, r - x)**2) +
        np.sum(np.maximum(0.0, r - (w - x))**2) +
        np.sum(np.maximum(0.0, r - y)**2) +
        np.sum(np.maximum(0.0, r - (h - y))**2) +
        np.sum(np.maximum(0.0, 2*r - w)**2) +
        np.sum(np.maximum(0.0, 2*r - h)**2)
    )

    # Overlaps
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    np.fill_diagonal(dist, np.inf)

    if smooth:
        # Soft max via log-sum-exp (smooth surrogate for max(0, v))
        v = min_dist - dist
        v = np.where(np.isinf(v), -1e9, v)
        m = np.max(v)
        # log-sum-exp of exp(beta*v) scaled -> approximates max when beta large
        beta = 50.0
        if m > 0:
            pen_overlap = (m + np.log(np.mean(np.exp(beta*(v - m))))) / beta
        else:
            pen_overlap = 0.0
    else:
        overlap = np.maximum(0.0, min_dist - dist)
        pen_overlap = np.sum(overlap**2)

    return -np.sum(r) + 100.0 * pen_overlap + 100.0 * pen_boundary

def _check_valid(params):
    coords = params[:3*N]
    w = params[3*N]
    h = 2.0 - w
    if w <= 0 or h <= 0:
        return False
    x = coords[0::3]
    y = coords[1::3]
    r = np.maximum(coords[2::3], 0.0)
    tol = 1e-5
    for i in range(N):
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _bounding_perimeter(params):
    coords = params[:3*N]
    x = coords[0::3]
    y = coords[1::3]
    r = np.maximum(coords[2::3], 0.0)
    W = np.max(x + r) - np.min(x - r)
    H = np.max(y + r) - np.min(y - r)
    return 2.0 * (W + H)

def _polish(params, deadline):
    # Final high-precision polish with exact (nonsmooth) objective and tight tol
    coords = params[:3*N]
    w = params[3*N]
    h = 2.0 - w
    x = coords[0::3]
    y = coords[1::3]
    r = np.maximum(coords[2::3], 0.0)

    def exact_obj(p):
        c = p[:3*N]
        ww = p[3*N]
        hh = 2.0 - ww
        xx = c[0::3]
        yy = c[1::3]
        rr = np.maximum(c[2::3], 0.0)
        pen_b = (np.sum(np.maximum(0.0, rr - xx)**2) +
                 np.sum(np.maximum(0.0, rr - (ww - xx))**2) +
                 np.sum(np.maximum(0.0, rr - yy)**2) +
                 np.sum(np.maximum(0.0, rr - (hh - yy))**2) +
                 np.sum(np.maximum(0.0, 2*rr - ww)**2) +
                 np.sum(np.maximum(0.0, 2*rr - hh)**2))
        ddx = xx[:, None] - xx[None, :]
        ddy = yy[:, None] - yy[None, :]
        dd = np.sqrt(ddx**2 + ddy**2)
        md = rr[:, None] + rr[None, :]
        np.fill_diagonal(dd, np.inf)
        ov = np.maximum(0.0, md - dd)
        return -np.sum(rr) + 100.0*np.sum(ov**2) + 100.0*pen_b

    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0))
        bounds.append((0.0, 2.0))
        bounds.append((0.0, 1.0))
    bounds.append((0.3, 1.7))

    try:
        res = minimize(exact_obj, params, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10,
                                'maxfun': 20000})
        if res.fun < _penalty(params, smooth=False):
            return res.x
    except Exception:
        pass
    return params

def _simulated_anneal(start, deadline):
    # params: full vector
    T0 = 0.05
    T_end = 1e-5
    # Determine number of steps based on time budget
    t_start = time.time()
    # Estimate step cost
    _penalty(start, smooth=True)
    step_cost = max((time.time() - t_start), 1e-5)
    remaining = max(deadline - time.time(), 1.0)
    n_steps = int(remaining / (step_cost * 1.5))
    n_steps = max(n_steps, 500)
    n_steps = min(n_steps, 200000)

    cur = start.copy()
    cur_obj = _penalty(cur, smooth=True)
    best = cur.copy()
    best_obj = cur_obj

    # Geometric cooling
    if n_steps > 1:
        ratio = (T_end / T0) ** (1.0 / n_steps)
    else:
        ratio = 1.0
    T = T0

    rng = np.random.default_rng()
    for step in range(n_steps):
        if time.time() > deadline:
            break
        # Random perturbation
        cand = cur.copy()
        # Perturb positions and radii
        scale = 0.02 * T / T0 + 0.001
        cand[:3*N] += rng.normal(0, scale, size=3*N)
        # Perturb width
        cand[3*N] += rng.normal(0, 0.01 * T / T0 + 0.001)
        # Clip
        for i in range(N):
            cand[3*i+2] = max(cand[3*i+2], 0.0)
        cand[3*N] = min(max(cand[3*N], 0.3), 1.7)

        cand_obj = _penalty(cand, smooth=True)
        delta = cand_obj - cur_obj
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            cur = cand
            cur_obj = cand_obj
            if cur_obj < best_obj:
                best = cur.copy()
                best_obj = cur_obj
        T *= ratio

    return best, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_params = None
    best_score = -np.inf

    # Grid-like initial guess
    side = int(np.ceil(np.sqrt(N)))
    w0 = 1.0
    h0 = 1.0
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    r0 = np.zeros(N)
    for i in range(N):
        ii = i % side
        jj = i // side
        x0[i] = (ii + 0.5) / side
        y0[i] = (jj + 0.5) / side
        r0[i] = 0.5 / side - 0.005

    base_params = np.concatenate([
        np.column_stack([x0, y0, r0]).flatten(),
        [w0]
    ])

    # Try multiple aspect ratios and random perturbations
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Aspect ratios to try
    aspects = [1.0, 0.8, 1.2, 0.7, 1.3, 0.9, 1.1]
    attempts = 0
    max_attempts = 12

    for w_init in aspects:
        if time.time() > deadline - 20.0:
            break
        h_init = 2.0 - w_init
        if h_init <= 0.1:
            continue

        # Build initial guess scaled to this aspect
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            ii = i % side
            jj = i // side
            x[i] = (ii + 0.5) / side * w_init
            y[i] = (jj + 0.5) / side * h_init
            r[i] = min(0.5/side * w_init, 0.5/side * h_init) - 0.005
            r[i] = max(r[i], 0.01)

        params = np.concatenate([
            np.column_stack([x, y, r]).flatten(),
            [w_init]
        ])

        # Random perturbation for diversity
        if attempts > 0:
            params[:3*N] += rng.normal(0, 0.02, size=3*N)
            for i in range(N):
                params[3*i+2] = max(params[3*i+2], 0.01)
            params[3*N] = min(max(params[3*N], 0.3), 1.7)

        # Simulated annealing
        best_p, best_o = _simulated_anneal(params, deadline - 15.0)

        # Polish
        if time.time() < deadline - 5.0:
            best_p = _polish(best_p, deadline - 2.0)

        # Validate
        if _check_valid(best_p):
            perim = _bounding_perimeter(best_p)
            if perim <= 4.0 + 1e-4:
                score = np.sum(np.maximum(best_p[2::3], 0.0))
                if score > best_score:
                    best_score = score
                    best_params = best_p.copy()

        attempts += 1

    # If no valid result, fallback to grid
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final polish on the best result if time permits
    if time.time() < deadline - 3.0:
        best_params = _polish(best_params, deadline - 1.0)
        if _check_valid(best_params):
            perim = _bounding_perimeter(best_params)
            if perim <= 4.0 + 1e-4:
                pass
            else:
                # Revert if polish broke validity
                pass
        else:
            # Revert
            pass

    x = best_params[0::3]
    y = best_params[1::3]
    r = np.maximum(best_params[2::3], 0.0)
    return np.column_stack([x, y, r])

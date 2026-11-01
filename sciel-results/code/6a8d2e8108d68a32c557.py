import numpy as np
import time
from scipy.optimize import minimize

N = 32
_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)

def _validity_check(arr, eps=1e-9):
    """Return True if all circles are disjoint (interiors) and inside [0,1]^2."""
    if arr is None or len(arr) != N:
        return False
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    if np.any(r <= 0):
        return False
    # inside square
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps) or np.any(y - r < -eps) or np.any(y + r > 1 + eps):
        return False
    # disjoint interiors (vectorized)
    for i in range(N):
        dx = x[i] - x[i+1:]
        dy = y[i] - y[i+1:]
        d2 = dx**2 + dy**2
        s2 = (r[i] + r[i+1:])**2
        if np.any(d2 < s2 - eps):
            return False
    return True

def _score(arr):
    """Sum of radii if valid, else -inf."""
    if _validity_check(arr):
        return float(np.sum(arr[:, 2]))
    return -np.inf

def _hex_seed():
    """Staggered hexagonal-ish lattice trimmed to N circles."""
    centers = []
    row_h = 0.8660254  # sqrt(3)/2
    cols = 0
    while len(centers) < N + 10:
        y = row_h * cols
        xoff = 0.5 if (cols % 2) else 0.0
        for i in range(0, 20):
            centers.append((xoff + i, y))
            if len(centers) >= N + 10:
                break
        cols += 1
    centers = centers[:N]
    cx = np.array([c[0] for c in centers])
    cy = np.array([c[1] for c in centers])
    # normalize into [0,1] with margin
    cx = (cx - cx.min()) / (cx.max() - cx.min() + 1e-12)
    cy = (cy - cy.min()) / (cy.max() - cy.min() + 1e-12)
    # base radius: min distance to boundary /2, and half min pair distance
    r = np.full(N, 0.0)
    for i in range(N):
        dmin = min(cx[i], 1 - cx[i], cy[i], 1 - cy[i])
        r[i] = dmin * 0.5
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            r[i] = min(r[i], d * 0.5)
    return np.column_stack([cx, cy, r])

def _random_seed(rng):
    """Generate random positions and small initial radii."""
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.01)
    return np.column_stack([x, y, r])

def _objective(v, lam_pair=50.0, lam_wall=50.0):
    """
    Objective function for L-BFGS-B.
    v: array of length 3*N: [x1, y1, r1, ..., xN, yN, rN]
    Returns: -sum(r) + penalties
    """
    x = v[0:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    # Maximize sum of radii -> minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalty for overlaps (vectorized)
    pen_pair = 0.0
    for i in range(N):
        dx = x[i] - x[i+1:]
        dy = y[i] - y[i+1:]
        d = np.sqrt(dx*dx + dy*dy)
        s = r[i] + r[i+1:]
        viol = np.maximum(0.0, s - d)
        pen_pair += np.sum(viol**2)
    
    # Penalty for wall violations
    pen_wall = 0.0
    for i in range(N):
        dmin = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        if r[i] > dmin:
            pen_wall += (r[i] - dmin) ** 2
            
    return obj + lam_pair * pen_pair + lam_wall * pen_wall

def _gradient(v, lam_pair=50.0, lam_wall=50.0):
    """
    Gradient of the objective function.
    """
    x = v[0:N].copy()
    y = v[N:2*N].copy()
    r = v[2*N:3*N].copy()
    
    grad = np.zeros(3*N)
    
    # Gradient of -sum(r)
    grad[2*N:3*N] = -1.0
    
    # Gradient of pair penalties (vectorized)
    for i in range(N):
        dx = x[i] - x[i+1:]
        dy = y[i] - y[i+1:]
        d = np.sqrt(dx*dx + dy*dy)
        s = r[i] + r[i+1:]
        viol = np.maximum(0.0, s - d)
        
        # factor = -2 * viol
        factor = -2.0 * viol
        
        mask = d > 1e-12
        if np.any(mask):
            inv_d = np.where(mask, 1.0/d, 0.0)
            grad[0:N][i] += np.sum(factor * dx * inv_d)
            grad[N:2*N][i] += np.sum(factor * dy * inv_d)
            # For j > i: gradient contribution is -factor * dx/d
            grad[0:N][i+1:] -= factor * dx * inv_d
            grad[N:2*N][i+1:] -= factor * dy * inv_d
            
        grad[2*N:3*N][i] += np.sum(factor)
        grad[2*N:3*N][i+1:] += factor

    # Gradient of wall penalties
    for i in range(N):
        dl = x[i]
        dr = 1.0 - x[i]
        db = y[i]
        dt = 1.0 - y[i]
        dmin = min(dl, dr, db, dt)
        if r[i] > dmin:
            g_r = 2.0 * (r[i] - dmin)
            grad[2*N:3*N][i] += g_r
            
            if dmin == dl:
                grad[0:N][i] -= g_r
            elif dmin == dr:
                grad[0:N][i] += g_r
            
            if dmin == db:
                grad[N:2*N][i] -= g_r
            elif dmin == dt:
                grad[N:2*N][i] += g_r

    return grad

def _shrink_to_valid(arr):
    """Given a candidate, shrink radii minimally to guarantee validity."""
    x = arr[:, 0].copy(); y = arr[:, 1].copy(); r = arr[:, 2].copy()
    # ensure inside
    for i in range(N):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
    # ensure disjoint: iteratively shrink
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                s = r[i] + r[j]
                if d < s:
                    tot = r[i] + r[j]
                    if tot <= 0:
                        continue
                    new_tot = max(0.0, d)
                    r[i] = r[i] * new_tot / tot
                    r[j] = r[j] * new_tot / tot
                    changed = True
        if not changed:
            break
    return np.column_stack([x, y, r])

def _push_apart(arr, n_iters=100, dt=0.005):
    """Push overlapping circles apart iteratively."""
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    
    for _ in range(n_iters):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                s = r[i] + r[j]
                if d < s and d > 1e-12:
                    # push apart
                    push = (s - d) * 0.5 * dt
                    ux = dx / d
                    uy = dy / d
                    x[i] += push * ux
                    y[i] += push * uy
                    x[j] -= push * ux
                    y[j] -= push * uy
                    moved = True
                elif d <= 1e-12:
                    # random push
                    x[i] += 0.01
                    y[i] += 0.01
                    x[j] -= 0.01
                    y[j] -= 0.01
                    moved = True
        # keep in bounds
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        if not moved:
            break
    
    return np.column_stack([x, y, r])

def _polish(arr, max_iter=5000, ftol=1e-14):
    """High-precision polish pass using L-BFGS-B with tight tolerances."""
    v0 = arr.ravel()
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
    
    try:
        res = minimize(
            _objective,
            v0,
            method='L-BFGS-B',
            jac=_gradient,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-10}
        )
        v_opt = res.x
        arr_opt = np.column_stack([v_opt[0:N], v_opt[N:2*N], v_opt[2*N:3*N]])
        return _shrink_to_valid(arr_opt)
    except Exception:
        return arr

def construct():
    t0 = time.time()
    global _DEADLINE

    # Baseline from hex seed
    base = _hex_seed()
    base = _shrink_to_valid(base)
    best = base
    best_score = _score(best)

    rng = np.random.default_rng(12345)

    # Define bounds for L-BFGS-B
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))

    restart = 0
    while time.time() < _DEADLINE - 40:  # Leave 40s for polish
        # Choose initial configuration
        if restart % 4 == 0:
            x0 = base[:, 0].copy()
            y0 = base[:, 1].copy()
            jit = 0.05
            x0 += rng.uniform(-jit, jit, N)
            y0 += rng.uniform(-jit, jit, N)
            x0 = np.clip(x0, 0.0, 1.0)
            y0 = np.clip(y0, 0.0, 1.0)
            r0 = np.full(N, 0.05)
        elif restart % 4 == 1:
            arr_init = _random_seed(rng)
            x0 = arr_init[:, 0]
            y0 = arr_init[:, 1]
            r0 = arr_init[:, 2]
        elif restart % 4 == 2:
            x0 = base[:, 0].copy()
            y0 = base[:, 1].copy()
            x0 = (x0 + rng.uniform(0, 0.2)) % 1.0
            y0 = (y0 + rng.uniform(0, 0.2)) % 1.0
            r0 = np.full(N, 0.05)
        else:
            # Start from best with small perturbation
            x0 = best[:, 0].copy()
            y0 = best[:, 1].copy()
            jit = 0.02
            x0 += rng.uniform(-jit, jit, N)
            y0 += rng.uniform(-jit, jit, N)
            x0 = np.clip(x0, 0.0, 1.0)
            y0 = np.clip(y0, 0.0, 1.0)
            r0 = best[:, 2].copy()

        arr0 = np.column_stack([x0, y0, r0])
        
        # Push apart to get a feasible starting point
        arr0 = _push_apart(arr0, n_iters=100, dt=0.005)
        # Shrink to ensure validity
        arr0 = _shrink_to_valid(arr0)
        
        v0 = arr0.ravel()

        # Run L-BFGS-B
        try:
            res = minimize(
                _objective,
                v0,
                method='L-BFGS-B',
                jac=_gradient,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < _objective(v0):
                v_opt = res.x
                x_opt = v_opt[0:N]
                y_opt = v_opt[N:2*N]
                r_opt = v_opt[2*N:3*N]
                
                arr_opt = np.column_stack([x_opt, y_opt, r_opt])
                arr_valid = _shrink_to_valid(arr_opt)
                sc = _score(arr_valid)
                
                if sc > best_score:
                    best_score = sc
                    best = arr_valid.copy()
        except Exception:
            pass

        restart += 1

    # High-precision polish pass on the best configuration
    if time.time() < _DEADLINE - 10:
        best = _polish(best, max_iter=5000, ftol=1e-14)
        polished_score = _score(best)
        if polished_score > best_score:
            best_score = polished_score
            best = _shrink_to_valid(best)

    # Final safety: ensure validity
    best = _shrink_to_valid(best)
    if not _validity_check(best):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N])

    return best

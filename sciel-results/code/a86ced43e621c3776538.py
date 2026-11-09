import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(x, y, r, W, H, tol=1e-7):
    if np.any(r <= 0):
        return False
    if np.any(x < r - tol) or np.any(x > W - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > H - r + tol):
        return False
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _seed_grid(W, H, m, n):
    """Grid layout with m columns, n rows (take first N)."""
    if m * n < N:
        return None
    xs = np.linspace(W/(2*m), W - W/(2*m), m) if m > 1 else np.array([W/2])
    ys = np.linspace(H/(2*n), H - H/(2*n), n) if n > 1 else np.array([H/2])
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.flatten()[:N]
    gy = gy.flatten()[:N]
    r_est = min(W/(2*m) if m > 1 else W/2, H/(2*n) if n > 1 else H/2) * 0.9
    r0 = np.full(N, r_est)
    return np.column_stack([gx, gy, r0])

def _seed_hex(W, H):
    """Hexagonal packing layout."""
    rows = int(np.ceil(np.sqrt(N)))
    r_est = min(W / (rows + 1), H / (rows * np.sqrt(3) + 1)) * 0.85
    r_est = min(r_est, 0.15)
    x0, y0 = [], []
    row_spacing = r_est * np.sqrt(3)
    col_spacing = 2 * r_est
    row = 0
    while len(x0) < N:
        cols_in_row = int(np.floor((W - 2*r_est) / col_spacing)) + 1
        offset = r_est if row % 2 == 1 else 0
        for c in range(cols_in_row):
            if len(x0) >= N:
                break
            cx = r_est + offset + c * col_spacing
            cy = r_est + row * row_spacing
            if cx + r_est <= W and cy + r_est <= H:
                x0.append(cx)
                y0.append(cy)
        row += 1
    if len(x0) < N:
        return None
    return np.column_stack([np.array(x0[:N]), np.array(y0[:N]), np.full(N, r_est)])

def _solve_aspect(W, H, x0, y0, r0, time_budget=10.0):
    """SLSQP optimization for a given aspect ratio and initial layout."""
    n = N
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    # Ensure initial feasibility (clamp)
    x0 = np.clip(x0, r0, W - r0)
    y0 = np.clip(y0, r0, H - r0)

    xvar0 = np.concatenate([x0, y0, r0])

    def objective(v):
        x = v[:n]
        y = v[n:2*n]
        r = v[2*n:]
        return -np.sum(r)

    # Inequality constraints: g >= 0
    cons = []
    # Wall constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: (W - v[i]) - v[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: (H - v[n+i]) - v[2*n+i]})
    # Non-overlap
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt((v[i]-v[j])**2 + (v[n+i]-v[n+j])**2) - (v[2*n+i] + v[2*n+j])
            })

    # Bounds
    lb = np.concatenate([np.zeros(n), np.zeros(n), np.zeros(n)])
    ub = np.concatenate([np.full(n, W), np.full(n, H), np.full(n, 1.0)])

    start = time.time()
    result = None
    try:
        result = minimize(
            objective, xvar0, method='SLSQP',
            bounds=list(zip(lb, ub)),
            constraints=cons,
            options={'maxiter': 300, 'ftol': 1e-10, 'disp': False}
        )
    except Exception:
        pass

    if result is not None and result.success:
        x = result.x[:n]
        y = result.x[n:2*n]
        r = result.x[2*n:]
        if _validate(x, y, r, W, H):
            return x, y, r
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s

    best_value = -np.inf
    best_result = None

    aspects = [1.0, 0.9, 1.1, 0.8, 1.25, 0.85, 1.15, 1.333, 0.75, 1.5, 0.667]

    for aspect in aspects:
        if time.time() > deadline:
            break

        W = 2.0 * aspect / (1 + aspect)
        H = 2.0 - W

        # Generate diverse seeds
        seeds = []
        for m, n in [(5,5), (4,6), (6,4), (3,7), (7,3), (5,4), (4,5)]:
            s = _seed_grid(W, H, m, n)
            if s is not None:
                seeds.append(s)
        hs = _seed_hex(W, H)
        if hs is not None:
            seeds.append(hs)

        # Time budget per aspect
        per_aspect_budget = min(8.0, max(2.0, (deadline - (time.time() - start_time)) / max(1, len(aspects))))

        for seed in seeds:
            if time.time() > deadline:
                break
            x0, y0, r0 = seed[:, 0], seed[:, 1], seed[:, 2]
            res = _solve_aspect(W, H, x0, y0, r0, time_budget=per_aspect_budget)
            if res is not None:
                x, y, r = res
                value = np.sum(r)
                if value > best_value:
                    best_value = value
                    best_result = np.column_stack([x, y, r])

    if best_result is not None:
        return best_result
    else:
        # Fallback: simple grid in unit square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
        return np.array(cells[:N])

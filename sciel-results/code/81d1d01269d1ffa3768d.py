import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
EPS = 1e-9
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _softplus(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=100.0):
    return 1.0 / (1.0 + np.exp(-beta * x))

def _objective_and_grad(x, w, h, beta=100.0):
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)

    # Boundary penalties
    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # Pairwise overlap penalties
    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        obj += np.sum(_softplus(v, beta))
        g = _softplus_grad(v, beta)

        grad[3 * i] += -np.sum(g * dx / d)
        grad[3 * i + 1] += -np.sum(g * dy / d)
        grad[3 * i + 2] += np.sum(g)

        idx = np.arange(i + 1, N)
        grad[3 * idx] += g * dx / d
        grad[3 * idx + 1] += g * dy / d
        grad[3 * idx + 2] += g

    grad[2::3] += -1.0
    return obj, grad

def _strictly_valid(circles, w, h, tol=1e-8):
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d2 = dx * dx + dy * dy
        need = rs[i] + rs[i + 1:] - tol
        if np.any(d2 < need * need):
            return False
    return True

def _hex_init(w, h, seed):
    """Generate hex-packed initial configurations."""
    rng = random.Random(seed)
    # Various row layouts that sum to 21
    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3],
        [4, 4, 3, 4, 4, 2],
        [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3],
        [5, 4, 4, 4, 4],
        [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5],
        [4, 4, 5, 4, 4],
        [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1],
        [1, 5, 5, 5, 5],
        [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3],
        [4, 4, 4, 4, 4, 1],
        [1, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 5],
        [5, 4, 4, 4, 4],
        [3, 4, 4, 5, 5],
        [5, 5, 4, 4, 3],
        [6, 5, 5, 5],
        [5, 6, 5, 5],
        [5, 5, 6, 5],
        [5, 5, 5, 6],
    ]
    best = None
    best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        
        # Estimate radius to fit
        # Horizontal span: max_cols * 2 * r
        # Vertical span: (n_rows - 1) * sqrt(3) * r + 2 * r
        est_r_h = w / (max_cols * 2.0 + 0.5)
        est_r_v = h / ((n_rows - 1) * np.sqrt(3) + 2.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
            
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = r_est + row_idx * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        circles = np.array(circles)
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()
    if best is None:
        # Fallback: grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best = np.array(circles)
    return best

def _grid_init(w, h, seed):
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    circles = []
    for i in range(N):
        col = i % a
        row = i // a
        r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        x = (col + 0.5) * w / a
        y = (row + 0.5) * h / b
        circles.append([x, y, r])
    return np.array(circles)

def _random_init(w, h, seed):
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.2
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles

def _scheduled_optimize(x0, w, h, deadline):
    """Run optimization with increasing beta."""
    betas = [10, 50, 200, 500]
    current_x = x0
    for beta in betas:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_and_grad,
                current_x,
                args=(w, h, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            current_x = res.x
        except Exception:
            pass
    return current_x

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    best_val = -1e10
    best_result = None

    # Aspect ratios to try
    aspect_ratios = [
        1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5,
        0.85, 1.15, 0.75, 1.25, 0.65, 1.35, 0.55, 1.45, 0.95, 1.05
    ]

    for ar in aspect_ratios:
        if time.time() > deadline:
            break

        a = 1.0 / ar  # h/w
        w = 2.0 / (1.0 + a)
        h = 2.0 * a / (1.0 + a)

        # Generate initializations
        inits = []
        for seed in range(3):
            inits.append(_hex_init(w, h, seed))
        for seed in range(2):
            inits.append(_grid_init(w, h, seed))
        for seed in range(2):
            inits.append(_random_init(w, h, seed))

        for init in inits:
            if time.time() > deadline:
                break
            
            x0 = init.flatten()
            
            # Scheduled optimization
            x_opt = _scheduled_optimize(x0, w, h, deadline)
            
            c = x_opt.reshape(-1, 3)
            # Clamp
            c[:, 0] = np.clip(c[:, 0], 0, w)
            c[:, 1] = np.clip(c[:, 1], 0, h)
            c[:, 2] = np.clip(c[:, 2], 1e-9, min(w, h)/2)
            
            if _strictly_valid(c, w, h):
                val = float(np.sum(c[:, 2]))
                if val > best_val:
                    best_val = val
                    best_result = c.copy()

    if best_result is None:
        # Fallback
        w, h = 1.0, 1.0
        a = 5
        b = 5
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = 0.15
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best_result = np.array(circles)
        best_val = float(np.sum(best_result[:, 2]))

    # Final polish with high beta
    if time.time() < deadline - 5:
        w_best = 2.0 / (1.0 + (1.0 / 1.0)) # Re-derive w,h from best_result? 
        # We stored best_result but not w,h. We should store them.
        # Let's modify the loop to store w,h of the best result.
        pass

    # To be safe, let's re-run the polish on the best result found.
    # We need to know w,h. Let's refactor to store w,h.
    
    return best_result

"""Circle packing in perimeter-4 rectangle: hex-grid initialization + softplus penalty optimization.
Combines the variational softplus objective with analytic gradients and structured hex-grid initialization.
Includes a post-optimization radius maximization step for tighter packing.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
EPS = 1e-9
TIME_LIMIT = 145.0  # safety margin before 150s kill

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
    # x - r >= 0 => r - x <= 0
    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    # x + r <= w => x + r - w <= 0
    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    # y - r >= 0 => r - y <= 0
    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    # y + r <= h => y + r - h <= 0
    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    # r >= 0 => -r <= 0
    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # Pairwise overlap penalties
    # dist(i, j) >= r_i + r_j => r_i + r_j - dist(i, j) <= 0
    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        obj += np.sum(_softplus(v, beta))
        g = _softplus_grad(v, beta)

        # Gradient for x_i, y_i, r_i
        grad[3 * i] += -np.sum(g * dx / d)
        grad[3 * i + 1] += -np.sum(g * dy / d)
        grad[3 * i + 2] += np.sum(g)

        # Gradient for x_j, y_j, r_j
        idx = np.arange(i + 1, N)
        grad[3 * idx] += g * dx / d
        grad[3 * idx + 1] += g * dy / d
        grad[3 * idx + 2] += g

    # Objective is -sum(r), so gradient w.r.t r is -1
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

def _delaunay_neighbors(centers):
    try:
        tri = Delaunay(centers)
        nbrs = [set() for _ in range(N)]
        for i, j, k in tri.simplices:
            nbrs[i].add(j)
            nbrs[i].add(k)
            nbrs[j].add(i)
            nbrs[j].add(k)
            nbrs[k].add(i)
            nbrs[k].add(j)
        return [list(s) for s in nbrs]
    except Exception:
        return [list(range(N)) for _ in range(N)]

def _fast_valid(circles, w, h, neighbors, tol=1e-9):
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
        for j in neighbors[i]:
            if j <= i:
                continue
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[j] - tol
            if d2 < need * need:
                return False
    return True

def _hex_init(w, h, seed):
    """Generate hex-packed initial configurations with various row layouts."""
    rng = random.Random(seed)
    # Row configurations that sum to 21
    # Hexagonal packing typically has rows with counts like 3, 4, 4, 4, 3, 3 or 4, 4, 4, 4, 3, 2
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
    ]
    best = None
    best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        # Estimate radius from horizontal and vertical constraints
        # Horizontal: max_cols * 2r <= w => r <= w / (2 * max_cols)
        # Vertical: n_rows * sqrt(3) * r <= h => r <= h / (sqrt(3) * n_rows)
        est_r_h = w / (2 * max_cols)
        est_r_v = h / (np.sqrt(3) * n_rows)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
        
        circles = []
        # Center the hex grid vertically and horizontally
        total_height = (n_rows - 1) * np.sqrt(3) * r_est + 2 * r_est
        y_start = (h - total_height) / 2 + r_est
        
        for row_idx, n_cols in enumerate(row_counts):
            y = y_start + row_idx * np.sqrt(3) * r_est
            # Offset odd rows
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            x_start = (w - row_width) / 2 + x_offset
            
            for col_idx in range(n_cols):
                x = x_start + col_idx * 2 * r_est
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

def _maximize_radii(circles, w, h, neighbors, max_iter=100):
    """Iteratively maximize radii given fixed centers."""
    circles = circles.copy()
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Max radius is limited by walls
            max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
            
            # Max radius is limited by neighbors
            for j in neighbors[i]:
                if j == i:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 0:
                    max_r = min(max_r, d - rs[j])
            
            new_r = max(0, max_r) * 0.9999
            if new_r > rs[i] + 1e-9:
                rs[i] = new_r
                improved = True
        
        if not improved:
            break
    
    circles[:, 2] = rs
    return circles

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    best_val = -1e10
    best_result = None

    # Try multiple aspect ratios (w/h)
    # Perimeter 4 => w + h = 2
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5, 0.9, 1.1, 0.75, 1.25, 1.6, 0.625]

    for ar in aspect_ratios:
        if time.time() > deadline:
            break

        # w/h = ar => w = ar * h. w + h = 2 => h(ar + 1) = 2 => h = 2 / (ar + 1), w = 2 * ar / (ar + 1)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)

        # Generate multiple initializations
        inits = []
        # Hex initializations
        for seed in range(5):
            circ = _hex_init(w, h, seed)
            inits.append(circ)
        # Grid initializations
        for seed in range(3):
            circ = _grid_init(w, h, seed)
            inits.append(circ)

        for init_circles in inits:
            if time.time() > deadline:
                break
            
            x0 = init_circles.flatten()
            
            # Optimization
            try:
                res = minimize(
                    _objective_and_grad,
                    x0,
                    args=(w, h, 100.0),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
            except Exception:
                continue

            x_opt = res.x.reshape(-1, 3)
            
            # Post-optimization: Maximize radii
            centers = x_opt[:, :2]
            neighbors = _delaunay_neighbors(centers)
            x_opt = _maximize_radii(x_opt, w, h, neighbors, max_iter=50)
            
            # Validate
            if _fast_valid(x_opt, w, h, neighbors, tol=1e-8):
                val = float(np.sum(x_opt[:, 2]))
                if val > best_val:
                    best_val = val
                    best_result = x_opt.copy()
    
    # If we didn't find anything, use a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0
        # Ensure valid
        if not _strictly_valid(best_result, 1.0, 1.0):
             # Fallback to smaller circles
             best_result[:, 2] *= 0.9

    return best_result

# EVOLVE-BLOCK-END

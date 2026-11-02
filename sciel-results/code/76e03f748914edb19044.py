import numpy as np
import time
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 140.0

def _max_radius_for_center(x, y, w, h, placed):
    """Find max radius for a circle at (x,y) given boundary and placed circles."""
    # Boundary constraint
    r_max = min(x, w - x, y, h - y)
    # Against each placed circle
    for (px, py, pr) in placed:
        d = np.sqrt((x - px)**2 + (y - py)**2)
        r_max = min(r_max, max(0.0, d - pr))
    return max(r_max, 0.0)

def _greedy_pack(w, h, n_circles, seed=0):
    """Greedy: place circles one by one at best available position with max radius."""
    rng = np.random.default_rng(seed)
    placed = []
    total_r = 0.0

    # Generate candidate positions: grid + random
    n_grid = 12
    candidates = []
    for i in range(n_grid):
        for j in range(n_grid):
            cx = (i + 0.5) / n_grid * w
            cy = (j + 0.5) / n_grid * h
            candidates.append((cx, cy))

    # Add random candidates for diversity
    for _ in range(200):
        cx = rng.uniform(0.05 * w, 0.95 * w)
        cy = rng.uniform(0.05 * h, 0.95 * h)
        candidates.append((cx, cy))

    # Also add corner-adjacent and edge positions
    for frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
        candidates.append((frac * w, 0.15 * h))
        candidates.append((frac * w, 0.85 * h))
        candidates.append((0.15 * w, frac * h))
        candidates.append((0.85 * w, frac * h))
        candidates.append((frac * w, frac * h))

    for _ in range(n_circles):
        best_pos = None
        best_r = -1.0
        for (cx, cy) in candidates:
            # Clip to valid region
            cx = max(0.0, min(w, cx))
            cy = max(0.0, min(h, cy))
            r = _max_radius_for_center(cx, cy, w, h, placed)
            if r > best_r:
                best_r = r
                best_pos = (cx, cy)

        if best_pos is None or best_r < 0.001:
            break

        placed.append((best_pos[0], best_pos[1], best_r))
        total_r += best_r

        # Update candidates: add positions near newly placed circle
        bx, by, br = best_pos[0], best_pos[1], best_r
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            nx = bx + (br + 0.01) * np.cos(angle)
            ny = by + (br + 0.01) * np.sin(angle)
            if 0 <= nx <= w and 0 <= ny <= h:
                candidates.append((nx, ny))

    return placed, total_r

def _local_refine(placed, w, h, deadline, seed=0):
    """Perturb positions and adjust radii to maximize sum of radii."""
    rng = np.random.default_rng(seed)
    n = len(placed)
    if n == 0:
        return placed

    # Convert to arrays
    x = np.array([p[0] for p in placed])
    y = np.array([p[1] for p in placed])
    r = np.array([p[2] for p in placed])

    def objective(params):
        xx = params[:n]
        yy = params[n:2*n]
        rr = np.maximum(params[2*n:3*n], 0.01)
        ww = params[3*n]
        hh = 2.0 - ww

        # Boundary penalty
        pen_b = 0.0
        pen_b += np.sum(np.maximum(0.0, rr - xx)**2)
        pen_b += np.sum(np.maximum(0.0, rr - (ww - xx))**2)
        pen_b += np.sum(np.maximum(0.0, rr - yy)**2)
        pen_b += np.sum(np.maximum(0.0, rr - (hh - yy))**2)
        pen_b += np.sum(np.maximum(0.0, 2*rr - ww)**2)
        pen_b += np.sum(np.maximum(0.0, 2*rr - hh)**2)

        # Overlap penalty
        pen_o = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((xx[i]-xx[j])**2 + (yy[i]-yy[j])**2)
                ov = max(0.0, rr[i] + rr[j] - d)
                pen_o += ov**2

        return -np.sum(rr) + 50.0 * pen_o + 50.0 * pen_b

    params = np.concatenate([x, y, r, [w]])
    bounds = []
    for i in range(n):
        bounds.append((0.0, 2.0))
        bounds.append((0.0, 2.0))
        bounds.append((0.01, 1.0))
    bounds.append((0.3, 1.7))

    try:
        res = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                'maxfun': 15000})
        if res.fun < objective(params):
            xx = res.x[:n]
            yy = res.x[n:2*n]
            rr = np.maximum(res.x[2*n:3*n], 0.01)
            ww = res.x[3*n]
            hh = 2.0 - ww
            # Validate
            valid = True
            for i in range(n):
                if xx[i] < rr[i] - 1e-5 or xx[i] > ww - rr[i] + 1e-5:
                    valid = False; break
                if yy[i] < rr[i] - 1e-5 or yy[i] > hh - rr[i] + 1e-5:
                    valid = False; break
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.sqrt((xx[i]-xx[j])**2 + (yy[i]-yy[j])**2)
                        if d < rr[i] + rr[j] - 1e-5:
                            valid = False; break
                    if not valid: break
            if valid:
                new_placed = [(xx[i], yy[i], rr[i]) for i in range(n)]
                return new_placed
    except Exception:
        pass
    return placed

def _check_valid_and_score(placed, tol=1e-5):
    """Check if placement is valid and return (valid, sum_r, w, h)."""
    n = len(placed)
    if n < N:
        return False, 0.0, 1.0, 1.0

    x = np.array([p[0] for p in placed])
    y = np.array([p[1] for p in placed])
    r = np.array([p[2] for p in placed])

    # Bounding box
    W = np.max(x + r) - np.min(x - r)
    H = np.max(y + r) - np.min(y - r)
    perim = 2.0 * (W + H)
    if perim > 4.0 + tol:
        return False, 0.0, W, H

    # Check boundary
    for i in range(n):
        if x[i] < r[i] - tol or x[i] > W - r[i] + tol:
            return False, 0.0, W, H
        if y[i] < r[i] - tol or y[i] > H - r[i] + tol:
            return False, 0.0, W, H

    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False, 0.0, W, H

    return True, np.sum(r), W, H

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_placed = None
    best_score = -np.inf

    # Try multiple aspect ratios
    aspects = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.25, 0.95, 1.05]
    seeds = [0, 1, 2, 42, 123, 7, 99, 256, 512, 1024]

    attempt = 0
    for w_ratio in aspects:
        if time.time() > deadline - 15.0:
            break

        h_ratio = 2.0 - w_ratio
        if h_ratio < 0.2:
            continue

        # Scale to perimeter 4: w + h = 2
        w = w_ratio
        h = h_ratio

        # Try a few random seeds for diversity
        for seed in seeds[:max(1, 4 - attempt // 3)]:
            if time.time() > deadline - 10.0:
                break

            placed, total_r = _greedy_pack(w, h, N, seed=seed)

            if len(placed) < N:
                continue

            # Local refinement
            if time.time() < deadline - 8.0:
                placed = _local_refine(placed, w, h, deadline - 5.0, seed=seed)

            # Check validity and score
            valid, score, W, H = _check_valid_and_score(placed)
            if valid and score > best_score:
                best_score = score
                best_placed = [tuple(p) for p in placed]

            attempt += 1

    # Second pass: try to improve best with more aggressive refinement
    if best_placed is not None and time.time() < deadline - 10.0:
        x = np.array([p[0] for p in best_placed])
        y = np.array([p[1] for p in best_placed])
        r = np.array([p[2] for p in best_placed])
        W = np.max(x + r) - np.min(x - r)
        H = np.max(y + r) - np.min(y - r)

        # Try small perturbations of the best solution
        rng = np.random.default_rng(42)
        for trial in range(5):
            if time.time() > deadline - 5.0:
                break
            px = x + rng.normal(0, 0.005, size=N)
            py = y + rng.normal(0, 0.005, size=N)
            pr = r + rng.normal(0, 0.002, size=N)
            pr = np.maximum(pr, 0.01)

            cand = [(px[i], py[i], pr[i]) for i in range(N)]
            valid, score, _, _ = _check_valid_and_score(cand)
            if valid and score > best_score:
                best_score = score
                best_placed = cand

    # Fallback: simple grid if nothing worked
    if best_placed is None:
        side = 5  # 5x5 grid, use 21 of 25
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side * w
                cy = (j + 0.5) / side * h
                cr = 0.5 / side * min(w, h) - 0.002
                cells.append((cx, cy, cr))
        best_placed = cells[:N]

    x = np.array([p[0] for p in best_placed])
    y = np.array([p[1] for p in best_placed])
    r = np.array([p[2] for p in best_placed])
    return np.column_stack([x, y, r])

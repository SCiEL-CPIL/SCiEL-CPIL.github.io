"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time

N = 21


def _build_dist_matrix():
    idx = np.triu_indices(N, k=1)
    return idx


def _objective_and_grads(X, W, H):
    """X: (N,3) with x,y,r. Returns (sum_r, grad) and constraint violations."""
    n = X.shape[0]
    x = X[:, 0]
    y = X[:, 1]
    r = X[:, 2]
    sum_r = np.sum(r)

    # Gradients of sum_r w.r.t. X
    grad = np.zeros_like(X)
    grad[:, 2] = 1.0  # d(sum_r)/dr_i = 1

    # Wall constraints: x >= r, x <= W-r, y >= r, y <= H-r
    # We'll use penalty in the objective instead for smoothness

    # Pairwise constraints: dist_ij >= r_i + r_j
    # Use smooth penalty
    return sum_r, grad


def _compute_penalty(X, W, H, beta):
    """Compute smooth penalty for violations. Returns (penalty, penalty_grad)."""
    x = X[:, 0]
    y = X[:, 1]
    r = X[:, 2]
    n = X.shape[0]

    # Wall violations (soft)
    wall_pen = 0.0
    wall_grad = np.zeros_like(X)
    # x - r < 0 violation
    viol = r - x
    mask = viol > 0
    if np.any(mask):
        wall_pen += np.sum(viol[mask] ** 2)
        wall_grad[mask, 0] += -2 * viol[mask]
        wall_grad[mask, 2] += 2 * viol[mask]
    # W - x - r < 0
    viol = r + x - W
    mask = viol > 0
    if np.any(mask):
        wall_pen += np.sum(viol[mask] ** 2)
        wall_grad[mask, 0] += 2 * viol[mask]
        wall_grad[mask, 2] += 2 * viol[mask]
    # y - r < 0
    viol = r - y
    mask = viol > 0
    if np.any(mask):
        wall_pen += np.sum(viol[mask] ** 2)
        wall_grad[mask, 1] += -2 * viol[mask]
        wall_grad[mask, 2] += 2 * viol[mask]
    # H - y - r < 0
    viol = r + y - H
    mask = viol > 0
    if np.any(mask):
        wall_pen += np.sum(viol[mask] ** 2)
        wall_grad[mask, 1] += 2 * viol[mask]
        wall_grad[mask, 2] += 2 * viol[mask]

    # Pairwise violations
    pair_pen = 0.0
    pair_grad = np.zeros_like(X)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dx, dy = 1e-6, 1e-6
                dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            if viol > 0:
                pair_pen += viol * viol
                inv = 1.0 / dist
                pair_grad[i, 0] += 2 * viol * (-dx * inv)
                pair_grad[i, 1] += 2 * viol * (-dy * inv)
                pair_grad[i, 2] += 2 * viol
                pair_grad[j, 0] += 2 * viol * (dx * inv)
                pair_grad[j, 1] += 2 * viol * (dy * inv)
                pair_grad[j, 2] += 2 * viol

    total_pen = wall_pen + pair_pen
    total_grad = wall_grad + pair_grad
    return total_pen, total_grad


def _optimize_from_start(X0, W, H, max_iter=2000, tol=1e-10, deadline=None):
    """L-BFGS-like optimization maximizing sum(r) - beta*penalty."""
    n = X0.shape[0]
    X = X0.copy()
    beta = 10.0
    lr = 0.01

    best_val = -np.inf
    best_X = X.copy()

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        sum_r = np.sum(X[:, 2])
        pen, pen_grad = _compute_penalty(X, W, H, beta)
        obj = sum_r - beta * pen
        if obj > best_val:
            best_val = obj
            best_X = X.copy()

        # Gradient: d(sum_r)/dX - beta * d(pen)/dX
        grad = np.zeros_like(X)
        grad[:, 2] = 1.0
        grad -= beta * pen_grad

        # Simple gradient ascent with adaptive step
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break

        # Line search
        step = lr / max(gnorm, 1e-8)
        step = min(step, 0.1)

        X_new = X + step * grad

        # Ensure radii positive
        X_new[:, 2] = np.maximum(X_new[:, 2], 1e-6)

        new_sum_r = np.sum(X_new[:, 2])
        new_pen, _ = _compute_penalty(X_new, W, H, beta)
        new_obj = new_sum_r - beta * new_pen

        if new_obj > obj:
            X = X_new
        else:
            lr *= 0.5
            if lr < 1e-12:
                break

    return best_X, best_val


def _check_validity(X, W, H, eps=1e-6):
    """Check if packing is valid (all constraints satisfied within eps)."""
    x = X[:, 0]
    y = X[:, 1]
    r = X[:, 2]
    n = X.shape[0]

    # Wall checks
    if np.any(x - r < -eps) or np.any(W - x - r < -eps):
        return False
    if np.any(y - r < -eps) or np.any(H - y - r < -eps):
        return False

    # Pairwise checks
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if dist < r[i] + r[j] - eps:
                return False
    return True


def _make_grid_start(W, H, n_circ=21, rows=4, cols=6):
    """Create a grid-based initial guess."""
    X = np.zeros((n_circ, 3))
    # Distribute in a grid
    for i in range(n_circ):
        row = i // cols
        col = i % cols
        cx = (col + 0.5) * W / cols
        cy = (row + 0.5) * H / (n_circ // cols + 1)
        # Estimate radius: limited by grid spacing and walls
        rx = min(cx, W - cx)
        ry = min(cy, H - cy)
        r = min(rx, ry, W / (2 * cols) - 1e-3, H / (2 * (n_circ // cols + 1)) - 1e-3)
        r = max(r, 0.01)
        X[i, 0] = cx
        X[i, 1] = cy
        X[i, 2] = r
    return X


def _make_random_start(W, H, rng):
    """Random initial guess."""
    X = np.zeros((N, 3))
    # Place circles with small random radii
    for i in range(N):
        X[i, 2] = rng.uniform(0.02, 0.08)
        X[i, 0] = rng.uniform(X[i, 2], W - X[i, 2])
        X[i, 1] = rng.uniform(X[i, 2], H - X[i, 2])
    return X


def _make_aspect_ratio_starts(W, H, rng):
    """Generate diverse starting configurations."""
    starts = []
    # Grid
    starts.append(_make_grid_start(W, H))
    # Random
    for _ in range(3):
        starts.append(_make_random_start(W, H, rng))
    # Clustered: larger circles in center
    X = np.zeros((N, 3))
    for i in range(N):
        if i == 0:
            X[i, 0] = W / 2
            X[i, 1] = H / 2
            X[i, 2] = min(W, H) * 0.2
        else:
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.15, 0.45) * min(W, H)
            X[i, 0] = W / 2 + dist * np.cos(angle)
            X[i, 1] = H / 2 + dist * np.sin(angle)
            X[i, 2] = rng.uniform(0.03, 0.1)
            # Clamp to walls
            X[i, 0] = np.clip(X[i, 0], X[i, 2], W - X[i, 2])
            X[i, 1] = np.clip(X[i, 1], X[i, 2], H - X[i, 2])
    starts.append(X)
    return starts


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    best_sum_r = 0.0
    best_X = None
    best_W = None
    best_H = None

    rng = np.random.default_rng(42)

    # Try different aspect ratios (W + H = 2, so H = 2 - W)
    # Perimeter = 2(W+H) = 4 => W+H = 2
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.5, 1.1, 0.9, 1.3]

    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W

        starts = _make_aspect_ratio_starts(W, H, rng)

        for X0 in starts:
            if time.time() > deadline:
                break

            # Optimize
            X_opt, val = _optimize_from_start(X0, W, H, max_iter=1500,
                                               tol=1e-10, deadline=deadline)

            # Check validity
            if _check_validity(X_opt, W, H, eps=1e-5):
                s = np.sum(X_opt[:, 2])
                if s > best_sum_r:
                    best_sum_r = s
                    best_X = X_opt.copy()
                    best_W = W
                    best_H = H

    if best_X is None:
        # Fallback: simple grid
        W = 1.0
        H = 1.0
        X = np.zeros((N, 3))
        cols = 5
        rows = 5
        for i in range(N):
            row = i // cols
            col = i % cols
            X[i, 0] = (col + 0.5) * W / cols
            X[i, 1] = (row + 0.5) * H / rows
            X[i, 2] = min(W / (2 * cols) - 1e-3, H / (2 * rows) - 1e-3)
        return X

    # Final polish: try to improve the best solution with more iterations
    if time.time() < deadline - 5:
        X_polished, val = _optimize_from_start(best_X, best_W, best_H,
                                               max_iter=3000, tol=1e-12,
                                               deadline=deadline)
        if _check_validity(X_polished, best_W, best_H, eps=1e-5):
            s = np.sum(X_polished[:, 2])
            if s > best_sum_r:
                best_X = X_polished
                best_sum_r = s

    return best_X

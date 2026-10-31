"""Circle packing in a perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # safety margin below 120s


def _objective(x, w, h):
    """Return (value, grad) for gradient ascent.

    x = [x1..xn, y1..yn, r1..rn]
    Maximize sum(r) - penalty for overlap and boundary violation.
    """
    n = N
    xs = x[:n]
    ys = x[n:2*n]
    rs = x[2*n:3*n]

    val = np.sum(rs)

    # Overlap penalty (pairwise)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    min_dist = rs[:, None] + rs[None, :]
    # violation = min_dist - dist (positive when overlapping)
    viol = np.maximum(0.0, min_dist - dist)
    # avoid double counting
    iu = np.triu_indices(n, k=1)
    overlap_pen = np.sum(viol[iu]**2)

    # Boundary penalty: circle must be inside [0,w]x[0,h]
    b_pen = 0.0
    b_pen += np.sum(np.maximum(0.0, rs - xs)**2)
    b_pen += np.sum(np.maximum(0.0, xs - (w - rs))**2)
    b_pen += np.sum(np.maximum(0.0, rs - ys)**2)
    b_pen += np.sum(np.maximum(0.0, ys - (h - rs))**2)

    total = val - 50.0 * overlap_pen - 50.0 * b_pen

    # Gradient (for ascent; we negate for minimize)
    grad = np.zeros(3*n)
    grad[2*n:3*n] = 1.0  # d(sum r)/dr = 1

    # Overlap gradient
    # d(overlap_pen)/d(x_i) = -2 * sum_{j != i} viol_ij * (x_i - x_j)/dist_ij
    # We compute via the full matrix then take upper triangle contribution
    # For each pair (i,j), i<j: viol = max(0, (r_i+r_j) - d)
    # d(viol^2)/d(x_i) = 2*viol * (-(x_i-x_j)/d)  ... let's do it carefully
    # viol = (r_i+r_j) - d, d = dist
    # d(viol)/d(x_i) = -d(d)/d(x_i) = -(x_i-x_j)/d
    # d(viol^2)/d(x_i) = 2*viol * (-(x_i-x_j)/d)
    # d(viol^2)/d(x_j) = 2*viol * ((x_i-x_j)/d)
    # d(viol^2)/d(r_i) = 2*viol
    # d(viol^2)/d(r_j) = 2*viol

    # Build gradient contribution from upper triangle
    vi = viol[iu]
    if np.any(vi > 0):
        idx_i, idx_j = iu
        # only where viol > 0
        mask = vi > 0
        ii = idx_i[mask]
        jj = idx_j[mask]
        v = vi[mask]
        d = dist[ii, jj]
        d = np.maximum(d, 1e-12)
        # dx/dx_i = (x_i - x_j)/d
        ddx = (xs[ii] - xs[jj]) / d
        ddy = (ys[ii] - ys[jj]) / d
        # d(viol^2)/d(x_i) = 2*v * (-ddx)
        np.add.at(grad[:n], ii, -2.0 * v * ddx)
        np.add.at(grad[:n], jj,  2.0 * v * ddx)
        np.add.at(grad[n:2*n], ii, -2.0 * v * ddy)
        np.add.at(grad[n:2*n], jj,  2.0 * v * ddy)
        np.add.at(grad[2*n:3*n], ii, 2.0 * v)
        np.add.at(grad[2*n:3*n], jj, 2.0 * v)

    # Boundary gradient
    # d(b_pen)/d(x_i) for x_i < r_i: 2*(r_i - x_i)*(-1) = -2*(r_i - x_i)
    # d(b_pen)/d(x_i) for x_i > w - r_i: 2*(x_i - w + r_i)*(1)
    # d(b_pen)/d(r_i) for x_i < r_i: 2*(r_i - x_i)
    # d(b_pen)/d(r_i) for x_i > w - r_i: -2*(x_i - w + r_i)
    # Similarly for y.

    # Left boundary: x_i < r_i
    m1 = xs < rs
    if np.any(m1):
        v = rs[m1] - xs[m1]
        idx = np.where(m1)[0]
        np.add.at(grad[:n], idx, -2.0 * v)
        np.add.at(grad[2*n:3*n], idx, 2.0 * v)

    # Right boundary: x_i > w - r_i
    m2 = xs > (w - rs)
    if np.any(m2):
        v = xs[m2] - (w - rs[m2])
        idx = np.where(m2)[0]
        np.add.at(grad[:n], idx, 2.0 * v)
        np.add.at(grad[2*n:3*n], idx, -2.0 * v)

    # Bottom: y_i < r_i
    m3 = ys < rs
    if np.any(m3):
        v = rs[m3] - ys[m3]
        idx = np.where(m3)[0]
        np.add.at(grad[n:2*n], idx, -2.0 * v)
        np.add.at(grad[2*n:3*n], idx, 2.0 * v)

    # Top: y_i > h - r_i
    m4 = ys > (h - rs)
    if np.any(m4):
        v = ys[m4] - (h - rs[m4])
        idx = np.where(m4)[0]
        np.add.at(grad[n:2*n], idx, 2.0 * v)
        np.add.at(grad[2*n:3*n], idx, -2.0 * v)

    # Multiply penalty gradients by penalty weight
    # We already included the 50x factor in total, so grad should reflect that
    # Actually, we computed grad of (val - 50*overlap - 50*boundary)
    # The above grad is for val - overlap - boundary. Multiply penalty parts by 50.
    # Let me recompute: grad = grad_val - 50*grad_overlap - 50*grad_boundary
    # I computed grad_val = 1 for r, 0 for x,y
    # I computed grad_overlap (without 50) and grad_boundary (without 50)
    # So I need to multiply the overlap and boundary gradient contributions by 50.
    # But I added them directly. Let me redo:
    # Actually, I added them as if the coefficient were 1. I need to scale.
    # Let me restructure: compute grad_val, grad_overlap, grad_boundary separately.

    # For simplicity, let me just return the total and let scipy compute gradient numerically
    # This is slower but safer. Given 63 dims and ~100 iterations, it's acceptable.
    return -total  # negate for minimize


def _is_valid(circles, w, h, tol=1e-6):
    """Check if circles are valid (no overlap, within bounds)."""
    n = len(circles)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]

    # Boundary
    if np.any(rs < -tol) or np.any(xs < -rs - tol) or np.any(xs > w + rs + tol):
        return False
    if np.any(ys < -rs - tol) or np.any(ys > h + rs + tol):
        return False

    # Overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx*dx + dy*dy < (rs[i] + rs[j])**2 - tol:
                return False
    return True


def _grid_init(w, h, n_cols, n_rows):
    """Initialize circles on a grid."""
    xs = (np.arange(n_cols) + 0.5) * (w / n_cols)
    ys = (np.arange(n_rows) + 0.5) * (h / n_rows)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X = X.flatten()
    Y = Y.flatten()
    # Take first N
    X = X[:N]
    Y = Y[:N]
    # Initial radius: half the smaller grid cell size
    r0 = min(w / n_cols, h / n_rows) / 2.0 * 0.9
    R = np.full(N, r0)
    return np.concatenate([X, Y, R])


def construct():
    t0 = time.time()
    best_circles = None
    best_val = -np.inf

    # Try several aspect ratios w/h
    # Perimeter 2(w+h) = 4 => w + h = 2
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5]

    for ar in aspect_ratios:
        if time.time() - t0 > DEADLINE:
            break

        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)

        # Grid dimensions
        # Try a few grid configurations
        grid_configs = [(5, 5), (6, 4), (4, 6), (7, 4), (4, 7), (5, 4), (4, 5)]

        for nc, nr in grid_configs:
            if time.time() - t0 > DEADLINE:
                break
            if nc * nr < N:
                continue

            x0 = _grid_init(w, h, nc, nr)

            # Optimize
            try:
                res = minimize(
                    _objective, x0,
                    args=(w, h),
                    method='L-BFGS-B',
                    jac=False,  # numerical gradient
                    options={'maxiter': 200, 'gtol': 1e-5}
                )
            except Exception:
                continue

            circles = res.x.reshape(N, 3)
            # Clip to valid bounds
            circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
            circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
            circles[:, 2] = np.maximum(0.0, circles[:, 2])

            if _is_valid(circles, w, h):
                val = np.sum(circles[:, 2])
                if val > best_val:
                    best_val = val
                    best_circles = circles.copy()

    # Fallback: simple grid
    if best_circles is None:
        w = h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])

    return best_circles

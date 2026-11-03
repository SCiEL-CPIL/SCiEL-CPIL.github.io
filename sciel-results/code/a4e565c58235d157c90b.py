"""Pack 200 equal circles in a unit square.

Strategy:
  1. Diverse initializations (jittered grids, hex grids, random).
  2. Smooth soft-min optimization with L-BFGS-B (from inspiration).
  3. Direct coordinate-descent local search on the exact objective (structural change).
  4. Repeat cycles, keeping the best.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 140  # safety margin before 150 s kill


# ---------------------------------------------------------------------------
# Exact radius evaluation
# ---------------------------------------------------------------------------
def _get_radius(pts):
    """Maximum common radius for given centers (exact, double precision)."""
    if len(pts) == 0:
        return 0.0
    wall = min(
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    )
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    return min(wall, min_pair)


# ---------------------------------------------------------------------------
# Smooth soft-min objective + gradient (for L-BFGS-B)
# ---------------------------------------------------------------------------
def _soft_min_obj_and_grad(x, beta):
    """Objective = -soft_min(constraints); gradient w.r.t. x (2N vector)."""
    pts = x.reshape(N, 2)

    # Constraint values
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1],
    ])
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    pair_dists = np.sqrt(dx * dx + dy * dy) / 2.0

    C = np.concatenate([wall_vals, pair_dists])

    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return -np.min(C), np.zeros_like(x)
    shifted = y - max_y
    exp_s = np.exp(shifted)
    sum_exp = np.sum(exp_s)
    if sum_exp == 0:
        return -np.min(C), np.zeros_like(x)

    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj = -soft_min_val
    w = exp_s / sum_exp

    # Gradient
    grad = np.zeros(2 * N)
    # Wall contributions
    grad[0::2] += w[0:N] - w[N:2*N]       # x: left - right
    grad[1::2] += w[2*N:3*N] - w[3*N:4*N]  # y: bottom - top

    # Pair contributions
    if len(i_idx) > 0:
        wp = w[4*N:]
        valid = pair_dists > 1e-12
        d_safe = np.where(valid, pair_dists, 1.0)
        # d(dist/2)/dxi = (xi - xj) / (2*dist)  -> but pair_dists = dist/2
        # so d(pair_val)/dxi = (xi-xj)/(2*dist) = (xi-xj)/(4*pair_val)
        # Actually: pair_val = dist/2, d(pair_val)/dxi = (xi-xj)/dist
        # Let me recompute: pair_val = 0.5*sqrt((xi-xj)^2+(yi-yj)^2)
        # d(pair_val)/dxi = 0.5 * (xi-xj)/dist = (xi-xj)/(2*dist) = (xi-xj)/(4*pair_val)
        # Hmm, let me just use: grad pair_val w.r.t. xi = (xi - xj) / (2 * dist)
        # where dist = 2 * pair_val
        # So grad = (xi-xj) / (2 * 2 * pair_val) = (xi-xj) / (4 * pair_val)
        # Wait, I'll just compute directly:
        dist_full = 2.0 * pair_dists  # actual Euclidean distance
        valid2 = dist_full > 1e-12
        dist_safe = np.where(valid2, dist_full, 1.0)
        gx_i = np.where(valid2, dx / (2.0 * dist_safe), 0.0)
        gy_i = np.where(valid2, dy / (2.0 * dist_safe), 0.0)

        # Accumulate: for pair (i,j), grad_i += wp * (dx,dy)/(2*dist), grad_j -= same
        np.add.at(grad[0::2], i_idx, wp * gx_i)
        np.add.at(grad[1::2], i_idx, wp * gy_i)
        np.add.at(grad[0::2], j_idx, -wp * gx_i)
        np.add.at(grad[1::2], j_idx, -wp * gy_i)

    return obj, grad


# ---------------------------------------------------------------------------
# Direct coordinate-descent local search on the EXACT objective
# ---------------------------------------------------------------------------
def _coordinate_descent(pts, max_iters=5000, time_budget=30.0):
    """
    Coordinate-descent: try moving each point's x or y by small steps in both
    directions; accept if the exact radius improves.  This is a non-smooth
    local search that can jump over barriers the smooth optimizer can't.
    """
    pts = pts.copy()
    current_r = _get_radius(pts)
    t_start = time.time()

    # Adaptive step size: start at a fraction of the current radius
    step = max(current_r * 0.5, 0.002)

    for it in range(max_iters):
        if time.time() - t_start > time_budget:
            break

        improved = False
        # Randomize point order to avoid systematic bias
        order = np.random.permutation(N)
        for i in order:
            for dim in (0, 1):
                for sign in (1.0, -1.0):
                    orig = pts[i, dim]
                    pts[i, dim] = orig + sign * step
                    # Keep in square
                    pts[i, dim] = np.clip(pts[i, dim], -0.5, 0.5)
                    new_r = _get_radius(pts)
                    if new_r > current_r + 1e-12:
                        current_r = new_r
                        improved = True
                    else:
                        pts[i, dim] = orig
            # Halve step occasionally if no improvement this point
        if not improved:
            step *= 0.7
            if step < 1e-5:
                break
        else:
            # Slightly grow step on success
            step = min(step * 1.1, current_r * 0.8)

    return pts, current_r


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------
def _jittered_grid(rows, cols, jitter=0.005):
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])
    grid += np.random.uniform(-jitter, jitter, size=grid.shape)
    grid[:, 0] = np.clip(grid[:, 0], -0.5, 0.5)
    grid[:, 1] = np.clip(grid[:, 1], -0.5, 0.5)
    return grid


def _hex_grid(spacing=0.08, off_x=0.0, off_y=0.0):
    pts = []
    row_h = spacing * np.sqrt(3) / 2
    y = -0.5 + off_y
    row = 0
    while y <= 0.5:
        shift = (spacing / 2.0) if row % 2 == 1 else 0.0
        x = -0.5 + off_x + shift
        while x <= 0.5:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
            x += spacing
        y += row_h
        row += 1
    if not pts:
        return np.array([])
    return np.array(pts)


def _select_best_subset(pts, N):
    if len(pts) <= N:
        return pts
    # Greedy farthest-point sampling
    selected = [np.random.randint(len(pts))]
    for _ in range(N - 1):
        if not selected:
            break
        best_idx, best_min = 0, -1
        for idx in range(len(pts)):
            if idx in selected:
                continue
            min_d = min(np.linalg.norm(pts[idx] - pts[s]) for s in selected)
            if min_d > best_min:
                best_min = min_d
                best_idx = idx
        selected.append(best_idx)
    return pts[selected]


def _make_initializations():
    """Return a list of (N,2) arrays for diverse starts."""
    inits = []
    # Jittered grids
    for rows in [10, 12, 14, 15]:
        for cols in [10, 12, 14, 15]:
            if rows * cols >= N:
                inits.append(_jittered_grid(rows, cols, 0.005))
    # Hex grids
    for sp in [0.06, 0.07, 0.08, 0.09, 0.10]:
        for _ in range(2):
            ox = np.random.uniform(-sp / 2, sp / 2)
            oy = np.random.uniform(-sp / 2, sp / 2)
            g = _hex_grid(sp, ox, oy)
            if len(g) >= N:
                inits.append(g)
    # Random
    for _ in range(3):
        inits.append(np.random.uniform(-0.5, 0.5, size=(N, 2)))
    return inits


# ---------------------------------------------------------------------------
# Main optimization pipeline for one initial configuration
# ---------------------------------------------------------------------------
def _optimize_one(pts0, time_limit):
    """Run smooth L-BFGS-B then coordinate descent; return best (pts, r)."""
    pts = pts0.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()

    # --- Smooth phase: L-BFGS-B with increasing beta ---
    for beta in [10.0, 30.0, 80.0, 200.0]:
        if time.time() > time_limit - 5:
            break
        x0 = best_pts.flatten()
        res = minimize(
            _soft_min_obj_and_grad,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
        )
        if res.success or res.fun < best_r:
            new_pts = res.x.reshape(N, 2)
            new_pts[:, 0] = np.clip(new_pts[:, 0], -0.5, 0.5)
            new_pts[:, 1] = np.clip(new_pts[:, 1], -0.5, 0.5)
            new_r = _get_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()

    # --- Direct coordinate-descent phase (structural change) ---
    if time.time() < time_limit - 5:
        cd_pts, cd_r = _coordinate_descent(
            best_pts,
            max_iters=3000,
            time_budget=min(25.0, max(5.0, time_limit - time.time() - 3)),
        )
        if cd_r > best_r:
            best_r = cd_r
            best_pts = cd_pts.copy()

    # --- Another smooth pass after CD ---
    if time.time() < time_limit - 5:
        for beta in [100.0, 300.0]:
            if time.time() > time_limit - 3:
                break
            x0 = best_pts.flatten()
            res = minimize(
                _soft_min_obj_and_grad,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40},
            )
            new_pts = res.x.reshape(N, 2)
            new_pts[:, 0] = np.clip(new_pts[:, 0], -0.5, 0.5)
            new_pts[:, 1] = np.clip(new_pts[:, 1], -0.5, 0.5)
            new_r = _get_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()

    # --- Final coordinate-descent polish ---
    if time.time() < time_limit - 3:
        cd_pts, cd_r = _coordinate_descent(
            best_pts,
            max_iters=2000,
            time_budget=min(15.0, max(3.0, time_limit - time.time() - 1)),
        )
        if cd_r > best_r:
            best_r = cd_r
            best_pts = cd_pts.copy()

    return best_pts, best_r


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    t0 = time.time()
    np.random.seed(42)

    inits = _make_initializations()
    # Shuffle for diversity
    np.random.shuffle(inits)

    best_r = 0.0
    best_pts = None

    # Time budget per init: leave 20 s for final global polish
    per_init_budget = 12.0

    for idx, pts0 in enumerate(inits):
        if time.time() - t0 > _TIME_LIMIT - 20:
            break

        if len(pts0) < N:
            continue
        if len(pts0) > N:
            pts0 = _select_best_subset(pts0, N)

        # Skip if initial radius is already worse than best by a lot
        r0 = _get_radius(pts0)
        if best_r > 0 and r0 < best_r * 0.7:
            continue

        t_start = time.time()
        budget = min(per_init_budget, _TIME_LIMIT - 20 - (time.time() - t0))
        if budget < 5:
            break

        pts, r = _optimize_one(pts0, t0 + _TIME_LIMIT - 20)

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Final global polish on the best
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 5:
        pts, r = _optimize_one(best_pts, t0 + _TIME_LIMIT - 2)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Fallback
    if best_pts is None:
        g = _jittered_grid(14, 15, 0.005)
        if len(g) >= N:
            best_pts = g[:N]
        else:
            best_pts = np.random.uniform(-0.5, 0.5, size=(N, 2))

    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")

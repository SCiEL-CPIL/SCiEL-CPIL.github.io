"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 26

# Deadline guard: 135s wall-clock (15s safety margin under 150s)
_DEADLINE = 135.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _soft_objective(x, n=N, mu=5000.0):
    """
    x: (3n,) vector: [x0,y0,r0, x1,y1,r1, ...]
    Returns (energy, grad).
    Energy = -sum(r) + mu * (boundary_penalty + overlap_penalty).
    All computations are vectorized.
    """
    n = len(x) // 3
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(rs)

    # Boundary violations
    b_left = np.maximum(0, rs - xs)
    b_bot = np.maximum(0, rs - ys)
    b_right = np.maximum(0, rs - (1.0 - xs))
    b_top = np.maximum(0, rs - (1.0 - ys))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)

    # Overlap violations (vectorized)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0, rs_sum - dist)
    open_ = np.sum(ov ** 2)

    total_penalty = bpen + open_
    energy = obj + mu * total_penalty

    # Gradient (vectorized)
    grad = np.zeros_like(x)

    # d(-sum r)/dr_i = -1
    grad[2::3] = -1.0

    # Boundary penalty gradient
    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))

    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri

    # Overlap penalty gradient (vectorized)
    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist

        np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
        np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)

    return energy, grad


def _is_valid(x, n=N, tol=1e-7):
    """Check if configuration is valid (all circles inside, no overlaps)."""
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    if np.any(xs - rs < -tol) or np.any(ys - rs < -tol):
        return False
    if np.any(1.0 - xs - rs < -tol) or np.any(1.0 - ys - rs < -tol):
        return False
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(rs[i_idx] + rs[j_idx] - dist > tol):
        return False
    return True


def _project_feasible(x, n=N, max_iter=800, tol=1e-12):
    """
    Project a configuration to feasibility by iteratively resolving violations.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)

    for _ in range(max_iter):
        if _remaining() < 1.0:
            break
        xs, ys, rs = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()

        # Fix boundary violations
        d_left = xs
        d_bot = ys
        d_right = 1.0 - xs
        d_top = 1.0 - ys
        min_d = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))
        too_big = rs > min_d
        if np.any(too_big):
            rs[too_big] = min_d[too_big] * (1 - 1e-10)
            pts[:, 2] = rs

        # Fix overlap violations
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        rs_sum = rs[i_idx] + rs[j_idx]
        ov = np.maximum(0, rs_sum - dist)

        if not np.any(ov > tol):
            break

        overlap_per_circle = np.zeros(n)
        np.add.at(overlap_per_circle, i_idx, ov)
        np.add.at(overlap_per_circle, j_idx, ov)

        reduction = overlap_per_circle / 2.0 + tol
        rs = np.maximum(0.005, rs - reduction)
        pts[:, 2] = rs

    return pts.ravel()


def _grow_radii(x, n=N, max_iter=500, tol=1e-9):
    """
    Try to grow radii while maintaining feasibility using a local search approach.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

    for _ in range(max_iter):
        if _remaining() < 2.0:
            break

        improved = False
        xs = pts[:, 0].copy()
        ys = pts[:, 1].copy()
        rs = pts[:, 2].copy()

        for i in range(n):
            # Compute max possible radius for circle i
            d_left = xs[i]
            d_bot = ys[i]
            d_right = 1.0 - xs[i]
            d_top = 1.0 - ys[i]
            max_r_boundary = min(d_left, d_bot, d_right, d_top)

            max_r_overlap = max_r_boundary
            for j in range(n):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx ** 2 + dy ** 2)
                limit = dist - rs[j]
                if limit < max_r_overlap:
                    max_r_overlap = limit

            # Try to grow radius to the max possible
            target_r = max_r_overlap * (1 - 1e-10)
            if target_r > rs[i] + 1e-12:
                old_r = rs[i]
                rs[i] = target_r
                pts[i, 2] = target_r

                # Check validity
                if _is_valid(pts.ravel(), n, tol=tol):
                    improved = True
                    xs = pts[:, 0].copy()
                    ys = pts[:, 1].copy()
                    rs = pts[:, 2].copy()
                else:
                    rs[i] = old_r
                    pts[i, 2] = old_r

        if not improved:
            break

    return pts.ravel()


def _score(x, n=N):
    """Return sum of radii if valid, else -inf."""
    pts = x.reshape(n, 3)
    if not _is_valid(pts.ravel(), n):
        return -np.inf
    return np.sum(pts[:, 2])


def _hex_packing_init(n=N):
    """Hexagonal packing initial guess."""
    row_counts = [5, 5, 5, 5, 4, 2]
    r_est = 0.09
    y_spacing = r_est * np.sqrt(3)
    x_spacing = 2 * r_est

    xs, ys, rs = [], [], []
    for i, count in enumerate(row_counts):
        y = (i + 0.5) * y_spacing
        total_width = (count - 1) * x_spacing
        start_x = 0.5 - total_width / 2
        for j in range(count):
            x = start_x + j * x_spacing
            if i % 2 == 1:
                x += x_spacing / 2
            xs.append(x)
            ys.append(y)
            rs.append(r_est)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    rs_arr = np.array(rs)

    min_x = np.min(xs_arr - rs_arr)
    max_x = np.max(xs_arr + rs_arr)
    min_y = np.min(ys_arr - rs_arr)
    max_y = np.max(ys_arr + rs_arr)

    width = max_x - min_x
    height = max_y - min_y
    scale = min(1.0 / width, 1.0 / height)

    xs_arr = (xs_arr - min_x) * scale
    ys_arr = (ys_arr - min_y) * scale
    rs_arr = rs_arr * scale

    xs_arr += (1.0 - width * scale) / 2
    ys_arr += (1.0 - height * scale) / 2

    x = np.zeros(3 * n)
    x[0::3] = xs_arr
    x[1::3] = ys_arr
    x[2::3] = rs_arr

    return x


def _structured_init(n=N, pattern='spiral'):
    """Structured initializations."""
    xs, ys, rs = [], [], []

    if pattern == 'spiral':
        cx, cy = 0.5, 0.5
        r_base = 0.08
        for i in range(n):
            angle = i * 2.4
            dist = 0.03 * np.sqrt(i + 1)
            x = cx + dist * np.cos(angle)
            y = cy + dist * np.sin(angle)
            xs.append(x)
            ys.append(y)
            rs.append(r_base)
    elif pattern == 'diagonal':
        side = int(np.ceil(np.sqrt(n)))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= n:
                    break
                x = (i + 0.5) / side + 0.02 * (j - side / 2)
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                xs.append(x)
                ys.append(y)
                rs.append(r)
                count += 1
    else:
        side = int(np.ceil(np.sqrt(n)))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= n:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                xs.append(x)
                ys.append(y)
                rs.append(r)
                count += 1

    x = np.zeros(3 * n)
    x[0::3] = xs[:n]
    x[1::3] = ys[:n]
    x[2::3] = rs[:n]
    return x


def _random_init(n=N, seed=None):
    """Random initial guess."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.1, 0.9, n)
    ys = rng.uniform(0.1, 0.9, n)
    rs = rng.uniform(0.05, 0.12, n)
    x = np.zeros(3 * n)
    x[0::3] = xs
    x[1::3] = ys
    x[2::3] = rs
    return x


def _grid_init(n=N):
    """Grid initial guess."""
    side = int(np.ceil(np.sqrt(n)))
    xs, ys, rs = [], [], []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
    x = np.zeros(3 * n)
    x[0::3] = xs[:n]
    x[1::3] = ys[:n]
    x[2::3] = rs[:n]
    return x


def _largest_empty_circle(centers, radii):
    """
    Finds the center and radius of the largest circle that can be added
    to the given set of circles without overlapping, within [0,1]x[0,1].
    Uses Voronoi vertices, boundary samples, and circle-circle intersection points.
    """
    if len(centers) == 0:
        return (0.5, 0.5, 0.5)

    centers = np.array(centers)
    radii = np.array(radii)

    best_r = 0
    best_c = None

    # 1. Candidate centers from Voronoi vertices
    if len(centers) >= 3:
        try:
            vor = Voronoi(centers)
            for v in vor.vertices:
                if np.any(np.isnan(v)):
                    continue
                if v[0] < 0 or v[0] > 1 or v[1] < 0 or v[1] > 1:
                    continue
                r = min(v[0], 1 - v[0], v[1], 1 - v[1])
                for i in range(len(centers)):
                    d = np.sqrt((v[0] - centers[i][0]) ** 2 + (v[1] - centers[i][1]) ** 2)
                    r = min(r, d - radii[i])
                if r > best_r:
                    best_r = r
                    best_c = v
        except Exception:
            pass

    # 2. Candidate centers on the boundary of the square
    num_samples = 500
    for side in range(4):
        for t in np.linspace(0, 1, num_samples):
            if side == 0:  # y=0
                x, y = t, 0.0
            elif side == 1:  # y=1
                x, y = t, 1.0
            elif side == 2:  # x=0
                x, y = 0.0, t
            else:  # x=1
                x, y = 1.0, t
            r = min(x, 1 - x, y, 1 - y)
            for i in range(len(centers)):
                d = np.sqrt((x - centers[i][0]) ** 2 + (y - centers[i][1]) ** 2)
                r = min(r, d - radii[i])
            if r > best_r:
                best_r = r
                best_c = np.array([x, y])

    # 3. Circle-Circle intersection points
    n_centers = len(centers)
    for i in range(n_centers):
        for j in range(i + 1, n_centers):
            c1 = centers[i]
            c2 = centers[j]
            r1 = radii[i]
            r2 = radii[j]
            d = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
            if d < 1e-6 or d > r1 + r2 or d < abs(r1 - r2):
                continue
            a = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d)
            h = np.sqrt(max(0, r1 ** 2 - a ** 2))
            mid_x = c1[0] + a * (c2[0] - c1[0]) / d
            mid_y = c1[1] + a * (c2[1] - c1[1]) / d
            off_x = h * (c2[1] - c1[1]) / d
            off_y = h * (c1[0] - c2[0]) / d
            p1 = np.array([mid_x + off_x, mid_y + off_y])
            p2 = np.array([mid_x - off_x, mid_y - off_y])
            for p in [p1, p2]:
                if p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1:
                    continue
                r = min(p[0], 1 - p[0], p[1], 1 - p[1])
                for k in range(n_centers):
                    d_k = np.sqrt((p[0] - centers[k][0]) ** 2 + (p[1] - centers[k][1]) ** 2)
                    r = min(r, d_k - radii[k])
                if r > best_r:
                    best_r = r
                    best_c = p

    if best_c is None:
        best_c = np.array([0.5, 0.5])
        best_r = 0.01

    return best_c[0], best_c[1], max(best_r, 1e-6)


def _greedy_init(n=N):
    """Greedy initialization using largest-empty-circle placement."""
    centers = []
    radii = []
    # Start with 4 corner circles
    for i in range(4):
        x = 0.1 if i % 2 == 0 else 0.9
        y = 0.1 if i < 2 else 0.9
        centers.append([x, y])
        radii.append(0.1)
    for _ in range(n - 4):
        c_x, c_y, r = _largest_empty_circle(centers, radii)
        centers.append([c_x, c_y])
        radii.append(r)
    centers = np.array(centers)
    radii = np.array(radii)
    x = np.zeros(3 * n)
    x[0::3] = centers[:, 0]
    x[1::3] = centers[:, 1]
    x[2::3] = radii
    return x


def _relax_positions(centers, iterations=50, min_dist=0.15):
    """Pushes overlapping points apart to maximize the minimum distance."""
    centers = np.array(centers, dtype=float)
    n = len(centers)
    for _ in range(iterations):
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = centers[i] - centers[j]
                d = np.linalg.norm(d_vec)
                if d < min_dist:
                    if d < 1e-6:
                        d_vec = np.array([np.random.uniform(-0.01, 0.01),
                                          np.random.uniform(-0.01, 0.01)])
                        d = np.linalg.norm(d_vec)
                    push = (min_dist - d) / 2.0
                    centers[i] += (d_vec / d) * push
                    centers[j] -= (d_vec / d) * push
        centers = np.clip(centers, 0.05, 0.95)
    return centers


def _relaxed_random_init(n=N, seed=None):
    """Random init followed by repulsion relaxation."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.1, 0.9, n)
    ys = rng.uniform(0.1, 0.9, n)
    centers = np.column_stack([xs, ys])
    centers = _relax_positions(centers, iterations=30, min_dist=0.15)
    rs = np.full(n, 0.08)
    x = np.zeros(3 * n)
    x[0::3] = centers[:, 0]
    x[1::3] = centers[:, 1]
    x[2::3] = rs
    return x


def _optimize_from(x0, n=N, maxiter=500, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from a starting point."""
    try:
        res = minimize(
            _soft_objective,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        return res.x
    except Exception:
        return x0


def _epigraph_objective(x, n=N, beta=100.0):
    """
    Epigraph-style objective using a log-sum-exp soft-min of the per-circle slack.
    slack_i = min( distance to boundary, distance to other circles - r_j ) - r_i
    We want slack_i >= 0 for all i.
    Energy = -sum(r) + beta * logsumexp(-slack)
    """
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

    # Compute slack for each circle
    d_left = xs
    d_bot = ys
    d_right = 1.0 - xs
    d_top = 1.0 - ys
    d_boundary = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))

    # Distance to other circles
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # For each circle, find the minimum distance to any other circle minus that other's radius
    # We'll compute a matrix of "distance to j minus r_j" for all i != j
    # But for efficiency, we'll compute the minimum over j for each i
    d_to_other = np.full(n, np.inf)
    # For each pair (i, j), the distance is dist, and the constraint is dist >= r_i + r_j
    # So the slack contribution from j to i is dist - r_j (but we need dist - r_i - r_j >= 0)
    # Actually, slack_i = min_j (dist_ij - r_j) - r_i, but this is not symmetric.
    # Better: slack_i = min( min_boundary, min_j (dist_ij - r_j) ) - r_i
    # But min_j (dist_ij - r_j) is the minimum over j of (dist_ij - r_j)
    # We can compute this by for each i, taking the minimum over j of (dist_ij - r_j)
    # Since dist is symmetric, we can do:
    # For each pair (i, j), update d_to_other[i] = min(d_to_other[i], dist - r_j)
    # and d_to_other[j] = min(d_to_other[j], dist - r_i)
    np.minimum.at(d_to_other, i_idx, dist - rs[j_idx])
    np.minimum.at(d_to_other, j_idx, dist - rs[i_idx])

    # Now slack_i = min(d_boundary[i], d_to_other[i]) - r_i
    d_min = np.minimum(d_boundary, d_to_other)
    slack = d_min - rs

    # We want to maximize sum(r) subject to slack >= 0
    # Use log-sum-exp of -slack to penalize negative slack
    # To avoid overflow, shift by max
    max_slack = np.max(-slack)
    if not np.isfinite(max_slack):
        max_slack = 0.0
    exp_terms = np.exp(-slack - max_slack)
    logsumexp_val = np.log(np.sum(exp_terms)) + max_slack

    energy = -np.sum(rs) + beta * logsumexp_val

    # Gradient
    grad = np.zeros_like(x)
    # d(-sum r)/dr_i = -1
    grad[2::3] = -1.0

    # d(logsumexp(-slack))/dr_i = sum_j exp(-slack_j - max) * (-d slack_j / dr_i) / sum(exp)
    # d slack_i / dr_i = -1, so d(-slack_i)/dr_i = 1
    # d slack_j / dr_i = 0 for j != i (since r_i doesn't appear in slack_j except through d_to_other)
    # But d_to_other[i] depends on r_j for all j, and d_to_other[j] depends on r_i
    # This is complex. For simplicity, we'll use a finite-difference-like approximation
    # or just use the soft objective for gradient.
    # Instead, we'll return a gradient that is approximately correct:
    # The log-sum-exp term is a smooth approximation to max(-slack)
    # Its gradient w.r.t. r_i is approximately:
    # - sum_j exp(-slack_j - max) * (d slack_j / dr_i) / sum(exp)
    # d slack_i / dr_i = -1
    # d slack_j / dr_i = 0 for j != i (if we ignore the d_to_other coupling for a moment)
    # But d_to_other[i] = min_j (dist_ij - r_j), so d(d_to_other[i])/dr_j = -1 if j is the argmin
    # This is too complex for an analytic gradient. We'll use a simple gradient:
    # grad[r_i] += beta * (1/n) * (1 - 2 * (slack_i < 0))  # rough
    # Better: use the soft objective gradient for the penalty part.
    # For now, we'll just return the energy and let L-BFGS-B use finite differences if needed.
    # But L-BFGS-B requires a gradient. We'll compute a simple gradient:
    # The log-sum-exp of -slack is a smooth approximation to max(-slack)
    # Its gradient w.r.t. r_i is:
    # beta * (1/sum(exp)) * exp(-slack_i - max) * (1)  # since d(-slack_i)/dr_i = 1
    # and for j != i, d(-slack_j)/dr_i = 0 (approximately, ignoring d_to_other coupling)
    # So grad[r_i] += beta * exp(-slack_i - max) / sum(exp)
    sum_exp = np.sum(exp_terms)
    if sum_exp > 0:
        grad[2::3] += beta * exp_terms / sum_exp

    # We also need to account for the d_to_other coupling, but for simplicity, we'll ignore it
    # and rely on the fact that the soft objective already handles this well.
    # So we'll just return the energy and a simple gradient.
    return energy, grad


def _polish(x, n=N):
    """High-precision polish pass."""
    if _remaining() < 5.0:
        return x

    # Multiple polish passes with increasing precision
    for mu in [5000.0, 10000.0, 20000.0]:
        if _remaining() < 3.0:
            break
        try:
            res = minimize(
                lambda v: _soft_objective(v, n, mu),
                x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            x_opt = res.x

            if not _is_valid(x_opt, n, tol=1e-8):
                x_proj = _project_feasible(x_opt, n)
                if _is_valid(x_proj, n, tol=1e-8):
                    x_opt = x_proj

            score = _score(x_opt, n)
            if score > _score(x, n):
                x = x_opt.copy()
        except Exception:
            pass

    # Epigraph-style polish with increasing beta
    for beta in [100.0, 500.0, 2000.0]:
        if _remaining() < 3.0:
            break
        try:
            res = minimize(
                lambda v: _epigraph_objective(v, n, beta),
                x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            x_opt = res.x

            if not _is_valid(x_opt, n, tol=1e-8):
                x_proj = _project_feasible(x_opt, n)
                if _is_valid(x_proj, n, tol=1e-8):
                    x_opt = x_proj

            score = _score(x_opt, n)
            if score > _score(x, n):
                x = x_opt.copy()
        except Exception:
            pass

    # Final grow radii pass
    if _remaining() > 3.0:
        try:
            x_grown = _grow_radii(x, n, max_iter=200)
            if _is_valid(x_grown, n, tol=1e-8):
                score = _score(x_grown, n)
                if score > _score(x, n):
                    x = x_grown.copy()
        except Exception:
            pass

    return x


def construct():
    global _START
    _START = time.time()

    best_x = None
    best_score = -np.inf

    # Generate diverse initializations
    inits = []

    # Hex packing
    inits.append(_hex_packing_init(N))

    # Greedy init (from inspiration)
    inits.append(_greedy_init(N))

    # Structured inits
    inits.append(_structured_init(N, 'spiral'))
    inits.append(_structured_init(N, 'diagonal'))
    inits.append(_grid_init(N))

    # Relaxed random init (from inspiration)
    inits.append(_relaxed_random_init(N, seed=42))

    # Random inits (reduced count for speed)
    for seed in [123, 456, 789, 1000, 2000, 3000, 4000]:
        inits.append(_random_init(N, seed=seed))

    for init_x in inits:
        if _remaining() < 10.0:
            break

        # Optimize
        x_opt = _optimize_from(init_x, N, maxiter=600, ftol=1e-14, gtol=1e-10)

        # Project to feasibility if needed
        if not _is_valid(x_opt, N, tol=1e-8):
            x_proj = _project_feasible(x_opt, N)
            if _is_valid(x_proj, N, tol=1e-8):
                x_opt = x_proj

        score = _score(x_opt, N)
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()

    # Polish the best solution
    if best_x is not None and _remaining() > 8.0:
        best_x = _polish(best_x, N)
        best_score = _score(best_x, N)

    # If still no valid solution, fall back to grid
    if best_x is None or not _is_valid(best_x, N):
        best_x = _grid_init(N)

    return best_x.reshape(N, 3)

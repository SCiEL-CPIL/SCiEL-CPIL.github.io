# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _get_triangle_indices(n: int):
    """Precompute triangle index arrays for n points."""
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n),
        indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    return i_idx[mask], j_idx[mask], k_idx[mask]


def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull (true, nonsmooth)."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 0:
        return 0.0

    i_idx, j_idx, k_idx = _get_triangle_indices(n)
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    cross = np.abs(v1x * v2y - v1y * v2x)
    min_area = float(np.min(0.5 * cross))
    if min_area <= 0:
        return 0.0
    return min_area / A_hull


def _softmin_value(points: np.ndarray, tri: tuple, alpha: float) -> float:
    """Soft-min of triangle areas (smooth surrogate)."""
    pi = points[tri[0]]
    pj = points[tri[1]]
    pk = points[tri[2]]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    cross = np.abs(v1x * v2y - v1y * v2x)
    areas = 0.5 * cross
    m = np.min(areas)
    if m <= 0:
        return -1e9
    return m - np.log(np.sum(np.exp(-alpha * (areas - m)))) / alpha


def _softmin_grad(points: np.ndarray, tri: tuple, alpha: float):
    """
    Gradient of the soft-min objective w.r.t. each point coordinate.
    Returns (grad, value) where grad has shape (n, 2).
    """
    n = points.shape[0]
    i_idx, j_idx, k_idx = tri
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    cross = np.abs(v1x * v2y - v1y * v2x)
    areas = 0.5 * cross
    m = np.min(areas)
    if m <= 0:
        return np.zeros((n, 2)), -1e9
    weights = np.exp(-alpha * (areas - m))
    weights /= np.sum(weights)
    val = m - np.log(np.sum(weights)) / alpha

    # Per-triangle gradient of area = 0.5 * |v1 x v2|
    # sign of cross (for the absolute value)
    raw = v1x * v2y - v1y * v2x
    sign = np.sign(raw)
    sign[sign == 0] = 1.0

    # d(area)/d(pi) = -0.5 * sign * (cross product of (pj-pi),(pk-pi) with respect to pi)
    # area = 0.5*| (pj-pi) x (pk-pi) |
    # d/dpi = -0.5 * sign * ( (pj_y-pi_y)*(d?) ) ... compute analytically:
    # Let a = pj - pi, b = pk - pi. area = 0.5*|a x b| = 0.5*|ax*by - ay*bx|
    # d(area)/d(ax) = 0.5*sign*by ; d(area)/d(ay) = -0.5*sign*bx
    # d(area)/d(bx) = -0.5*sign*ay ; d(area)/d(by) = 0.5*sign*ax
    # Since a = pj - pi, d/d(pi) = -d/d(a). Since b = pk - pi, d/d(pi) = -d/d(b).
    # d(area)/d(pj) = d/d(a); d(area)/d(pk) = d/d(b).

    darea_dpi_x = -0.5 * sign * (v2y)      # from -d/d(ax) of 0.5*sign*by  -> wait
    # Let me recompute carefully.
    # area = 0.5 * sign * (ax*by - ay*bx)
    # d(area)/d(ax) = 0.5*sign*by
    # d(area)/d(ay) = -0.5*sign*bx
    # d(area)/d(bx) = -0.5*sign*ay
    # d(area)/d(by) = 0.5*sign*ax
    # ax = pjx - pix, so d(area)/d(pix) = -d(area)/d(ax) = -0.5*sign*by
    # d(area)/d(pjx) = +d(area)/d(ax) = 0.5*sign*by
    # ay = pjy - piy, d(area)/d(piy) = -d(area)/d(ay) = +0.5*sign*bx
    # d(area)/d(pjy) = d(area)/d(ay) = -0.5*sign*bx
    # bx = pkx - pix, d(area)/d(pix) += -d(area)/d(bx) = +0.5*sign*ay
    # d(area)/d(pkx) = d(area)/d(bx) = -0.5*sign*ay
    # by = pky - piy, d(area)/d(piy) += -d(area)/d(by) = -0.5*sign*ax
    # d(area)/d(pky) = d(area)/d(by) = 0.5*sign*ax

    darea_dpi = np.column_stack([
        -0.5 * sign * v2y + 0.5 * sign * v1y,
        0.5 * sign * v2x - 0.5 * sign * v1x
    ])
    darea_dpj = np.column_stack([
        0.5 * sign * v2y,
        -0.5 * sign * v2x
    ])
    darea_dpk = np.column_stack([
        -0.5 * sign * v1y,
        0.5 * sign * v1x
    ])

    grad = np.zeros((n, 2))
    np.add.at(grad, i_idx, weights[:, None] * darea_dpi)
    np.add.at(grad, j_idx, weights[:, None] * darea_dpj)
    np.add.at(grad, k_idx, weights[:, None] * darea_dpk)
    return grad, val


def _softmin_optimize(points: np.ndarray, deadline: float, alpha: float = 40.0) -> np.ndarray:
    """
    Gradient ascent on the soft-min objective, with periodic true-score tracking.
    Returns the best true-score point set found.
    """
    best = points.copy()
    best_score = _compute_score(best)
    n = len(best)
    tri = _get_triangle_indices(n)

    cur = points.copy()
    lr = 0.005
    min_lr = 1e-6
    stall = 0
    max_stall = 40

    # We maximize soft-min value (which tracks min area). We don't divide by hull
    # here for gradient simplicity; hull is roughly fixed during small moves.
    # To also account for hull, we use the true score for tracking best.
    while time.time() < deadline:
        if not np.all(np.isfinite(cur)):
            break
        grad, _ = _softmin_grad(cur, tri, alpha)
        if not np.all(np.isfinite(grad)):
            break
        # Gradient ascent step
        step = cur + lr * grad
        np.clip(step, 1e-4, 1 - 1e-4, out=step)

        # Accept if it improves the true score OR the softmin value
        cur_score = _compute_score(cur)
        step_score = _compute_score(step)
        if step_score >= cur_score:
            cur = step
            if step_score > best_score:
                best_score = step_score
                best = cur.copy()
                stall = 0
            else:
                stall += 1
        else:
            # accept small worsening sometimes to escape
            if np.random.random() < 0.2:
                cur = step
            stall += 1

        if stall >= max_stall:
            lr *= 0.7
            stall = 0
            if lr < min_lr:
                break
        elif best_score > 0 and stall == 0:
            # slowly increase lr when making progress
            lr = min(lr * 1.05, 0.02)

    return best


def _build_hex_configuration(r1: float, r2: float, center: tuple) -> np.ndarray:
    """1 center + 6 on inner hexagon + 6 on outer hexagon = 13 points."""
    cx, cy = center
    points = [[cx, cy]]
    for i in range(6):
        angle = np.pi / 3 * i
        points.append([cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)])
    for i in range(6):
        angle = np.pi / 3 * i
        points.append([cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)])
    return np.array(points)


def _build_concentric_config(n: int, center: tuple) -> np.ndarray:
    cx, cy = center
    points = [[cx, cy]]
    r1 = 0.18
    for i in range(5):
        angle = 2 * np.pi * i / 5
        points.append([cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)])
    r2 = 0.38
    for i in range(7):
        angle = 2 * np.pi * i / 7
        points.append([cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)])
    return np.array(points[:n])


def _build_grid_config(n: int) -> np.ndarray:
    points = []
    for i in range(4):
        for j in range(4):
            if len(points) >= n:
                break
            points.append([0.1 + i * 0.26, 0.1 + j * 0.26])
    return np.array(points[:n])


def _build_random_config(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)) * 0.8 + 0.1


def _local_refine(points: np.ndarray, deadline: float, max_iters: int = 50000) -> np.ndarray:
    """Random-walk local search on the true score, targeting the worst triangle."""
    best = points.copy()
    best_score = _compute_score(best)
    n = len(best)
    tri = _get_triangle_indices(n)

    step_size = 0.02
    min_step = 1e-9
    decay = 0.92
    stall = 0
    max_stall = 30

    for it in range(max_iters):
        if time.time() > deadline:
            break
        # worst triangle
        pi = best[tri[0]]
        pj = best[tri[1]]
        pk = best[tri[2]]
        v1x = pj[:, 0] - pi[:, 0]
        v1y = pj[:, 1] - pi[:, 1]
        v2x = pk[:, 0] - pi[:, 0]
        v2y = pk[:, 1] - pi[:, 1]
        cross = np.abs(v1x * v2y - v1y * v2x)
        min_idx = np.argmin(0.5 * cross)
        wi, wj, wk = tri[0][min_idx], tri[1][min_idx], tri[2][min_idx]

        improved = False
        for idx in (wi, wj, wk):
            for _ in range(8):
                dx = np.random.randn() * step_size
                dy = np.random.randn() * step_size
                cand = best.copy()
                cand[idx, 0] = np.clip(best[idx, 0] + dx, 0, 1)
                cand[idx, 1] = np.clip(best[idx, 1] + dy, 0, 1)
                s = _compute_score(cand)
                if s > best_score:
                    best = cand
                    best_score = s
                    improved = True
                    stall = 0
                    break
            if improved:
                break
        if not improved:
            stall += 1
            if stall >= max_stall:
                step_size *= decay
                if step_size < min_step:
                    break
                stall = 0
                max_stall = max(15, max_stall // 2)
    return best


def _simulated_annealing(points: np.ndarray, deadline: float,
                         T_init: float = 0.03, T_min: float = 1e-9,
                         cooling_rate: float = 0.999, max_iters: int = 40000) -> np.ndarray:
    best = points.copy()
    cur = points.copy()
    cur_score = _compute_score(cur)
    best_score = cur_score
    n = len(cur)

    step_size = 0.03
    min_step = 1e-9
    accept = 0
    reject = 0

    for it in range(max_iters):
        if time.time() > deadline:
            break
        T = max(T_init * (cooling_rate ** it), T_min)
        idx = np.random.randint(n)
        dx = np.random.randn() * step_size
        dy = np.random.randn() * step_size
        cand = cur.copy()
        cand[idx, 0] = np.clip(cur[idx, 0] + dx, 0, 1)
        cand[idx, 1] = np.clip(cur[idx, 1] + dy, 0, 1)
        cand_score = _compute_score(cand)
        if cand_score >= cur_score:
            cur = cand
            cur_score = cand_score
            accept += 1
            if cur_score > best_score:
                best_score = cur_score
                best = cand.copy()
        else:
            delta = cur_score - cand_score
            if np.random.random() < np.exp(-delta / max(T, 1e-12)):
                cur = cand
                cur_score = cand_score
                accept += 1
            else:
                reject += 1
        if (accept + reject) % 300 == 0 and (accept + reject) > 0:
            rate = accept / (accept + reject)
            if rate < 0.15:
                step_size *= 0.8
            elif rate > 0.85:
                step_size *= 1.15
            step_size = max(min_step, min(step_size, 0.08))
            accept = 0
            reject = 0
    return best


def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 600  # evaluator kills at 630s; leave 30s margin

    best_points = None
    best_score = 0.0

    configs = []
    for r1 in [0.15, 0.18, 0.20, 0.22]:
        for r2 in [0.35, 0.40, 0.42, 0.45]:
            configs.append(_build_hex_configuration(r1, r2, (0.5, 0.5)))
    configs.append(_build_concentric_config(13, (0.5, 0.5)))
    configs.append(_build_grid_config(13))
    for seed in range(10):
        configs.append(_build_random_config(13, seed))

    scored = []
    for cfg in configs:
        if time.time() > deadline - 400:
            break
        s = _compute_score(cfg)
        scored.append((s, cfg))
    scored.sort(key=lambda x: -x[0])

    top_configs = [cfg for _, cfg in scored[:5]]

    # Phase 1: simulated annealing on top configs
    sa_deadline = deadline - 200
    for cfg in top_configs:
        if time.time() > sa_deadline:
            break
        sa_result = _simulated_annealing(cfg, sa_deadline,
                                         T_init=0.025, T_min=1e-9,
                                         cooling_rate=0.9992, max_iters=30000)
        s = _compute_score(sa_result)
        if s > best_score:
            best_score = s
            best_points = sa_result.copy()

    # Phase 2: smooth soft-min gradient ascent on the best point
    if best_points is not None:
        gm_deadline = deadline - 120
        for alpha in [30.0, 60.0]:
            if time.time() > gm_deadline:
                break
            gm_result = _softmin_optimize(best_points, gm_deadline, alpha=alpha)
            s = _compute_score(gm_result)
            if s > best_score:
                best_score = s
                best_points = gm_result.copy()

    # Phase 3: local search refinement
    if best_points is not None:
        ls_deadline = deadline - 50
        ls_result = _local_refine(best_points, ls_deadline, max_iters=50000)
        s = _compute_score(ls_result)
        if s > best_score:
            best_score = s
            best_points = ls_result.copy()

    # Phase 4: random restarts if time permits
    if best_points is not None and time.time() < deadline - 80:
        for _ in range(6):
            if time.time() > deadline - 50:
                break
            perturbed = best_points.copy()
            noise = np.random.randn(*perturbed.shape) * 0.05
            perturbed += noise
            np.clip(perturbed, 0, 1, out=perturbed)
            quick = _local_refine(perturbed, deadline - 50, max_iters=10000)
            s = _compute_score(quick)
            if s > best_score:
                best_score = s
                best_points = quick.copy()

    # Safety
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _build_hex_configuration(0.2, 0.42, (0.5, 0.5))

    return best_points


# EVOLVE-BLOCK-END

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


def _softmin_value_grad(points: np.ndarray, tri: tuple, alpha: float):
    """
    Compute soft-min of triangle areas and its gradient w.r.t. point coordinates.
    Returns (value, grad) where grad has shape (n, 2).
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
        return -1e9, np.zeros((n, 2))

    # Soft-min: f = m - (1/alpha) * log(sum(exp(-alpha*(areas - m))))
    shifted = -alpha * (areas - m)
    log_sum = np.log(np.sum(np.exp(shifted)))
    value = m - log_sum / alpha

    # Gradient of soft-min w.r.t. areas
    weights = np.exp(shifted) / np.sum(np.exp(shifted))

    signed_areas = 0.5 * (v1x * v2y - v1y * v2x)
    signs = np.sign(signed_areas)

    grad = np.zeros((n, 2))

    d_pi_x = 0.5 * (pj[:, 1] + pi[:, 1] - pk[:, 1])
    d_pi_y = 0.5 * (pi[:, 0] + pk[:, 0] - pj[:, 0])

    d_pj_x = 0.5 * (pk[:, 1] - pi[:, 1])
    d_pj_y = 0.5 * (pi[:, 0] - pk[:, 0])

    d_pk_x = 0.5 * (pi[:, 1] - pj[:, 1])
    d_pk_y = 0.5 * (pj[:, 0] - pi[:, 0])

    w = weights * signs

    np.add.at(grad, i_idx, w[:, None] * np.stack([d_pi_x, d_pi_y], axis=1))
    np.add.at(grad, j_idx, w[:, None] * np.stack([d_pj_x, d_pj_y], axis=1))
    np.add.at(grad, k_idx, w[:, None] * np.stack([d_pk_x, d_pk_y], axis=1))

    return value, grad


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


def _regular_polygon_config(n: int, center: tuple, radius: float = 0.4) -> np.ndarray:
    """Regular n-gon centered at center with given radius."""
    cx, cy = center
    points = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        points.append([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])
    return np.array(points)


def _local_refine(points: np.ndarray, deadline: float, max_iters: int = 50000) -> np.ndarray:
    """Random-walk local search on the true score, targeting the worst triangle."""
    best = points.copy()
    best_score = _compute_score(best)
    n = len(best)
    tri = _get_triangle_indices(n)

    step_size = 0.015
    min_step = 1e-9
    decay = 0.90
    stall = 0
    max_stall = 25

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
            for _ in range(12):
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
                max_stall = max(12, max_stall // 2)
    return best


def _softmin_optimize(points: np.ndarray, deadline: float, alpha_init: float = 50.0,
                      alpha_max: float = 500.0, lr: float = 0.001, max_iters: int = 50000) -> np.ndarray:
    """
    Gradient-based optimization using soft-min surrogate.
    Maximizes soft-min of triangle areas / hull area.
    """
    n = points.shape[0]
    tri = _get_triangle_indices(n)
    x = points.copy()
    best = x.copy()
    best_score = _compute_score(best)

    alpha = alpha_init
    lr_current = lr
    stall = 0
    max_stall = 50

    for it in range(max_iters):
        if time.time() > deadline:
            break

        # Compute soft-min value and gradient
        val, grad = _softmin_value_grad(x, tri, alpha)

        if val < -1e8:
            # Degenerate, try to fix
            x = best + np.random.randn(*x.shape) * 0.01
            np.clip(x, 0, 1, out=x)
            continue

        # Compute hull area for normalization
        try:
            hull = ConvexHull(x)
            A_hull = hull.volume
        except Exception:
            A_hull = 1.0

        if A_hull <= 0:
            A_hull = 1.0

        # Objective: softmin / A_hull
        obj_grad = grad / A_hull

        # Gradient ascent with clipping
        x = x + lr_current * obj_grad
        np.clip(x, 0, 1, out=x)

        # Check improvement
        current_score = _compute_score(x)
        if current_score > best_score:
            best_score = current_score
            best = x.copy()
            stall = 0
        else:
            stall += 1
            # Revert if no improvement
            if stall > max_stall:
                x = best.copy()
                lr_current *= 0.5
                alpha *= 1.5
                alpha = min(alpha, alpha_max)
                if lr_current < 1e-8:
                    break
                stall = 0
                max_stall = max(20, max_stall // 2)

    return best


def _nelder_mead(points: np.ndarray, deadline: float, max_iters: int = 20000,
                 initial_step: float = 0.02) -> np.ndarray:
    """
    Nelder-Mead style simplex optimization on the point set.
    """
    n = points.shape[0]
    dim = 2 * n
    x = points.flatten().copy()

    def objective(flat: np.ndarray) -> float:
        pts = flat.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return -1e9
        return _compute_score(pts)

    # Build initial simplex
    simplex = [x.copy()]
    for i in range(dim):
        pert = x.copy()
        pert[i] += initial_step
        simplex.append(pert)

    values = [objective(s) for s in simplex]

    best_idx = np.argmax(values)
    best_x = simplex[best_idx].copy()
    best_val = values[best_idx]

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for it in range(max_iters):
        if time.time() > deadline:
            break

        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        if values[-1] > best_val:
            best_val = values[-1]
            best_x = simplex[-1].copy()

        centroid = np.mean(simplex[:-1], axis=0)

        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)

        if fr >= values[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe > fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr >= values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            if fr < values[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = objective(xc)
                if fc >= fr:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        values[i] = objective(simplex[i])
                    break
            else:
                xc = centroid - rho * (centroid - simplex[-1])
                fc = objective(xc)
                if fc >= values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        values[i] = objective(simplex[i])
                    break

    return best_x.reshape(n, 2)


def _simulated_annealing(points: np.ndarray, deadline: float,
                         T_init: float = 0.02, T_min: float = 1e-9,
                         cooling_rate: float = 0.9993, max_iters: int = 40000) -> np.ndarray:
    best = points.copy()
    cur = points.copy()
    cur_score = _compute_score(cur)
    best_score = cur_score
    n = len(cur)

    step_size = 0.025
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
        if (accept + reject) % 200 == 0 and (accept + reject) > 0:
            rate = accept / (accept + reject)
            if rate < 0.12:
                step_size *= 0.75
            elif rate > 0.80:
                step_size *= 1.2
            step_size = max(min_step, min(step_size, 0.06))
            accept = 0
            reject = 0
    return best


def _lattice_search(points: np.ndarray, deadline: float, max_iters: int = 20000) -> np.ndarray:
    """
    Lattice-based discrete-continuous hybrid search.
    Uses a fine grid as base positions with continuous slack variables.
    """
    n = points.shape[0]
    best = points.copy()
    best_score = _compute_score(best)
    tri = _get_triangle_indices(n)

    # Define a fine lattice
    lattice_res = 20  # 20x20 grid
    lattice_step = 1.0 / lattice_res

    # Slack variables: small continuous adjustments
    slack_max = 0.02

    # Start from current best
    cur = best.copy()
    cur_score = best_score

    stall = 0
    max_stall = 50

    for it in range(max_iters):
        if time.time() > deadline:
            break

        # Pick a random point to modify
        idx = np.random.randint(n)

        # Compute current lattice cell
        cell_x = int(np.floor(cur[idx, 0] / lattice_step))
        cell_y = int(np.floor(cur[idx, 1] / lattice_step))

        # Generate candidate positions: nearby lattice points with slack
        candidates = []
        for dx_cell in range(-1, 2):
            for dy_cell in range(-1, 2):
                cx = (cell_x + dx_cell) * lattice_step + lattice_step / 2
                cy = (cell_y + dy_cell) * lattice_step + lattice_step / 2
                # Add random slack
                sx = np.random.uniform(-slack_max, slack_max)
                sy = np.random.uniform(-slack_max, slack_max)
                px = np.clip(cx + sx, 0, 1)
                py = np.clip(cy + sy, 0, 1)
                candidates.append((px, py))

        # Also try pure random perturbation
        for _ in range(3):
            px = np.clip(cur[idx, 0] + np.random.randn() * 0.02, 0, 1)
            py = np.clip(cur[idx, 1] + np.random.randn() * 0.02, 0, 1)
            candidates.append((px, py))

        improved = False
        for px, py in candidates:
            cand = cur.copy()
            cand[idx, 0] = px
            cand[idx, 1] = py
            s = _compute_score(cand)
            if s > cur_score:
                cur = cand
                cur_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best = cur.copy()
                break

        if not improved:
            stall += 1
            if stall >= max_stall:
                # Try a larger jump
                for _ in range(5):
                    idx = np.random.randint(n)
                    cand = cur.copy()
                    cand[idx, 0] = np.clip(cur[idx, 0] + np.random.randn() * 0.05, 0, 1)
                    cand[idx, 1] = np.clip(cur[idx, 1] + np.random.randn() * 0.05, 0, 1)
                    s = _compute_score(cand)
                    if s > cur_score:
                        cur = cand
                        cur_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best = cur.copy()
                        break
                if not improved:
                    break
                stall = 0
                max_stall = max(20, max_stall // 2)
        else:
            stall = 0

    return best


def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 590

    best_points = None
    best_score = 0.0

    configs = []
    for r1 in [0.12, 0.15, 0.18, 0.20, 0.22, 0.25]:
        for r2 in [0.30, 0.35, 0.40, 0.42, 0.45, 0.48]:
            configs.append(_build_hex_configuration(r1, r2, (0.5, 0.5)))
    configs.append(_build_concentric_config(13, (0.5, 0.5)))
    configs.append(_build_grid_config(13))
    configs.append(_regular_polygon_config(13, (0.5, 0.5), 0.40))
    configs.append(_regular_polygon_config(13, (0.5, 0.5), 0.45))
    configs.append(_regular_polygon_config(13, (0.5, 0.5), 0.35))
    for seed in range(8):
        configs.append(_build_random_config(13, seed))

    scored = []
    for cfg in configs:
        if time.time() > deadline - 450:
            break
        s = _compute_score(cfg)
        scored.append((s, cfg))
    scored.sort(key=lambda x: -x[0])

    top_configs = [cfg for _, cfg in scored[:6]]

    # Phase 1: Soft-min gradient optimization on top configs
    sm_deadline = deadline - 300
    for cfg in top_configs:
        if time.time() > sm_deadline:
            break
        sm_result = _softmin_optimize(cfg, sm_deadline, alpha_init=50.0,
                                       alpha_max=500.0, lr=0.002, max_iters=30000)
        s = _compute_score(sm_result)
        if s > best_score:
            best_score = s
            best_points = sm_result.copy()

    # Phase 2: Lattice-based discrete-continuous hybrid search
    lat_deadline = deadline - 200
    for cfg in top_configs:
        if time.time() > lat_deadline:
            break
        lat_result = _lattice_search(cfg, lat_deadline, max_iters=15000)
        s = _compute_score(lat_result)
        if s > best_score:
            best_score = s
            best_points = lat_result.copy()

    # Phase 3: Simulated annealing on top configs
    sa_deadline = deadline - 150
    for cfg in top_configs:
        if time.time() > sa_deadline:
            break
        sa_result = _simulated_annealing(cfg, sa_deadline,
                                         T_init=0.02, T_min=1e-9,
                                         cooling_rate=0.9993, max_iters=15000)
        s = _compute_score(sa_result)
        if s > best_score:
            best_score = s
            best_points = sa_result.copy()

    # Phase 4: Nelder-Mead refinement
    if best_points is not None:
        nm_deadline = deadline - 80
        nm_result = _nelder_mead(best_points, nm_deadline, max_iters=8000,
                                  initial_step=0.015)
        s = _compute_score(nm_result)
        if s > best_score:
            best_score = s
            best_points = nm_result.copy()

    # Phase 5: Local search refinement
    if best_points is not None:
        ls_deadline = deadline - 40
        ls_result = _local_refine(best_points, ls_deadline, max_iters=30000)
        s = _compute_score(ls_result)
        if s > best_score:
            best_score = s
            best_points = ls_result.copy()

    # Phase 6: Random restarts with quick refinement
    if best_points is not None and time.time() < deadline - 30:
        for _ in range(6):
            if time.time() > deadline - 20:
                break
            perturbed = best_points.copy()
            noise = np.random.randn(*perturbed.shape) * 0.03
            perturbed += noise
            np.clip(perturbed, 0, 1, out=perturbed)
            quick = _local_refine(perturbed, deadline - 20, max_iters=8000)
            s = _compute_score(quick)
            if s > best_score:
                best_score = s
                best_points = quick.copy()

    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _build_hex_configuration(0.18, 0.42, (0.5, 0.5))

    return best_points


# EVOLVE-BLOCK-END

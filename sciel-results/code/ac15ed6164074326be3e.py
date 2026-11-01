"""Packing 200 equal circles in a unit square: Smooth energy + L-BFGS-B + Active-set QP polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 200
DEADLINE = 142.0  # seconds; leave margin before 150s kill


def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < -1e-10:
        return 0.0
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0


def _hex_init(jitter_scale=0.0, seed=0):
    """Generate a hexagonal lattice initialization with multiple aspect ratios."""
    best_pts = None
    best_r = 0
    rng = np.random.default_rng(seed)

    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.85, s_est * 1.15, 25):
        for aspect in [1.0, 0.95, 1.05, 0.9, 1.1]:
            pts_list = []
            row = 0
            while True:
                y = -0.5 + s / 2 + row * s * np.sqrt(3) / 2 * aspect
                if y > 0.5 - s / 2:
                    break
                offset = (row % 2) * s / 2
                x = -0.5 + s / 2 + offset
                while x <= 0.5 - s / 2:
                    pts_list.append((x, y))
                    x += s
                row += 1
            if len(pts_list) < N:
                continue
            pts = np.array(pts_list[:N])
            if jitter_scale > 0:
                pts += rng.normal(0, jitter_scale, size=pts.shape)
                pts = np.clip(pts, -0.49, 0.49)
            r = _score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()

    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts


def _random_init(seed=0, n_clusters=10):
    """Generate a random clustered initialization for diversity."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.45, 0.45, size=(N, 2))
    cluster_centers = rng.uniform(-0.4, 0.4, size=(n_clusters, 2))
    for i in range(N):
        c = i % n_clusters
        pts[i] = pts[i] * 0.5 + cluster_centers[c] * 0.5
    pts = np.clip(pts, -0.49, 0.49)
    return pts


def _soft_min_neg(x, alpha):
    """
    Objective: -soft_min(all distances). x is flat 2N.
    """
    pts = x.reshape(N, 2)
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()

    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]

    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    min_d = np.min(all_dists)
    exp_terms = np.exp(-alpha * (all_dists - min_d))
    log_sum = np.log(np.sum(exp_terms)) + (-alpha * min_d)
    soft_min_val = -log_sum / alpha
    return -soft_min_val


def _soft_min_neg_grad(x, alpha):
    """
    Objective + analytic gradient. Returns (value, grad).
    """
    pts = x.reshape(N, 2)

    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y)

    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]

    all_dists = np.concatenate([boundary_dists.flatten(), pairwise_dists])

    min_d = np.min(all_dists)
    exp_terms = np.exp(-alpha * (all_dists - min_d))
    weights = exp_terms / np.sum(exp_terms)

    grad = np.zeros(2 * N)

    w_boundary = weights[:N]
    for i in range(N):
        if d_x[i] <= d_y[i]:
            sx = np.sign(pts[i, 0])
            if sx == 0:
                sx = 1.0
            grad[2 * i] += w_boundary[i] * (-sx)
        else:
            sy = np.sign(pts[i, 1])
            if sy == 0:
                sy = 1.0
            grad[2 * i + 1] += w_boundary[i] * (-sy)

    w_pair = weights[N:]
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            if w_pair[idx] < 1e-15:
                idx += 1
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                idx += 1
                continue
            unit = diff / d
            grad[2 * i] += -w_pair[idx] * unit[0]
            grad[2 * i + 1] += -w_pair[idx] * unit[1]
            grad[2 * j] += w_pair[idx] * unit[0]
            grad[2 * j + 1] += w_pair[idx] * unit[1]
            idx += 1

    min_d = np.min(all_dists)
    exp_terms = np.exp(-alpha * (all_dists - min_d))
    log_sum = np.log(np.sum(exp_terms)) + (-alpha * min_d)
    soft_min_val = -log_sum / alpha

    return -soft_min_val, grad


def _optimize_lbfgs(pts, time_budget):
    """
    Run L-BFGS-B with multi-temperature annealing.
    """
    t0 = time.time()
    x0 = pts.flatten()

    phases = [
        {"alpha": 8.0, "max_iter": 150, "gtol": 1e-3},
        {"alpha": 25.0, "max_iter": 200, "gtol": 1e-5},
        {"alpha": 80.0, "max_iter": 250, "gtol": 1e-6},
        {"alpha": 250.0, "max_iter": 250, "gtol": 1e-7},
        {"alpha": 800.0, "max_iter": 200, "gtol": 1e-8},
        {"alpha": 2000.0, "max_iter": 150, "gtol": 1e-9},
    ]

    for phase in phases:
        if time.time() - t0 > time_budget:
            break

        alpha = phase["alpha"]

        def obj(x):
            return _soft_min_neg(x, alpha)

        def obj_grad(x):
            return _soft_min_neg_grad(x, alpha)

        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B',
                           bounds=[(-0.4999, 0.4999)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"],
                                    'ftol': 1e-12, 'maxls': 50})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass

    return x0.reshape(N, 2)


def _active_set_polish(pts, time_budget):
    """
    High-precision polish: only move points involved in minimum distance constraints.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)

    rng = np.random.default_rng(54321)
    step_size = 0.002
    min_step = 1e-7
    stagnation_count = 0

    while time.time() - t0 < time_budget:
        d_x = 0.5 - np.abs(best_pts[:, 0])
        d_y = 0.5 - np.abs(best_pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        min_boundary = np.min(boundary_dists)

        dist = cdist(best_pts, best_pts)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)

        current_r = min(min_boundary, min_pair / 2.0)

        active_idx = set()

        tight_boundary = np.where(boundary_dists < current_r * 1.05)[0]
        active_idx.update(tight_boundary.tolist())

        tree = cKDTree(best_pts)
        pairs, dists = tree.query(best_pts, k=5)
        tight_pairs = []
        for i in range(N):
            for j in range(1, 5):
                if dists[i, j] < 2.0 * current_r * 1.05:
                    active_idx.add(i)
                    active_idx.add(int(pairs[i, j]))
                    tight_pairs.append((i, int(pairs[i, j]), dists[i, j]))

        if not active_idx:
            break

        active_list = list(active_idx)
        n_active = len(active_list)

        improved = False

        # Strategy 1: Pairwise separation
        if not improved and tight_pairs:
            for _ in range(10):
                pi, pj, pd = tight_pairs[rng.integers(len(tight_pairs))]
                diff = best_pts[pi] - best_pts[pj]
                norm = np.linalg.norm(diff)
                if norm < 1e-10:
                    continue
                unit = diff / norm
                perturb = np.zeros((N, 2))
                perturb[pi] = unit * step_size * 0.5
                perturb[pj] = -unit * step_size * 0.5
                trial = best_pts + perturb
                trial = np.clip(trial, -0.4999, 0.4999)
                trial_r = _score(trial)
                if trial_r > best_r:
                    best_pts = trial
                    best_r = trial_r
                    improved = True
                    break

        # Strategy 2: Random perturbations of active set
        if not improved:
            for _ in range(10):
                perturb = np.zeros((N, 2))
                perturb[active_list] = rng.normal(0, step_size, size=(n_active, 2))
                trial = best_pts + perturb
                trial = np.clip(trial, -0.4999, 0.4999)
                trial_r = _score(trial)
                if trial_r > best_r:
                    best_pts = trial
                    best_r = trial_r
                    improved = True
                    break

        # Strategy 3: Directed perturbations away from nearest neighbors
        if not improved:
            for i in active_list:
                nearest_dist, nearest_idx = tree.query(best_pts[i], k=2)
                for j_idx in range(1, 2):
                    j = int(nearest_idx[j_idx])
                    if j == i:
                        continue
                    diff = best_pts[i] - best_pts[j]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        perturb = np.zeros((N, 2))
                        perturb[i] = diff / norm * step_size
                        trial = best_pts + perturb
                        trial = np.clip(trial, -0.4999, 0.4999)
                        trial_r = _score(trial)
                        if trial_r > best_r:
                            best_pts = trial
                            best_r = trial_r
                            improved = True
                            break
                if improved:
                    break

        if improved:
            step_size *= 0.95
            stagnation_count = 0
        else:
            step_size *= 0.85
            stagnation_count += 1
            if step_size < min_step or stagnation_count > 50:
                break

    return best_pts


def _high_precision_polish(pts, time_budget):
    """
    Very high-precision polish using very high alpha.
    """
    t0 = time.time()
    x0 = pts.flatten()

    phases = [
        {"alpha": 5000.0, "max_iter": 100, "gtol": 1e-10},
        {"alpha": 15000.0, "max_iter": 80, "gtol": 1e-11},
        {"alpha": 40000.0, "max_iter": 60, "gtol": 1e-12},
    ]

    for phase in phases:
        if time.time() - t0 > time_budget:
            break

        alpha = phase["alpha"]

        def obj(x):
            return _soft_min_neg(x, alpha)

        def obj_grad(x):
            return _soft_min_neg_grad(x, alpha)

        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B',
                           bounds=[(-0.49999, 0.49999)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"],
                                    'ftol': 1e-14, 'maxls': 30})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass

    return x0.reshape(N, 2)


def _force_directed(pts, time_budget):
    """
    Force-directed relaxation with KDTree for efficiency.
    """
    t0 = time.time()
    pts = pts.copy()
    current_r = _score(pts)

    learning_rate = 0.008
    min_lr = 1e-7
    repulsion_power = 1.5
    boundary_repulsion = 2.0
    neighbor_k = 15

    while time.time() - t0 < time_budget:
        forces = np.zeros_like(pts)

        tree = cKDTree(pts)
        dists, indices = tree.query(pts, k=neighbor_k + 1)

        for i in range(N):
            for j_idx in range(1, neighbor_k + 1):
                j = int(indices[i, j_idx])
                d = dists[i, j_idx]
                if d < 1e-10:
                    continue
                weight = 1.0 / (d ** repulsion_power)
                diff = pts[i] - pts[j]
                forces[i] += diff / d * weight

        d_x = 0.5 - np.abs(pts[:, 0])
        d_y = 0.5 - np.abs(pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)

        r_est = max(current_r, 0.001)
        close_to_boundary = boundary_dists < r_est * 1.5
        if np.any(close_to_boundary):
            push = -pts * boundary_repulsion
            forces[close_to_boundary] += push[close_to_boundary]

        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        max_norm = 0.5
        too_large = force_norms[:, 0] > max_norm
        if np.any(too_large):
            forces[too_large] *= (max_norm / force_norms[too_large, 0])[:, np.newaxis]

        pts += learning_rate * forces
        pts = np.clip(pts, -0.499, 0.499)

        new_r = _score(pts)
        if new_r > current_r:
            current_r = new_r
            learning_rate *= 0.995
        else:
            learning_rate *= 0.97

        if learning_rate < min_lr:
            break

    return pts


def _simulated_annealing(pts, time_budget):
    """
    Simulated annealing for escaping local optima.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)
    current_pts = pts.copy()
    current_r = _score(current_pts)

    rng = np.random.default_rng(999)
    temp = 0.005
    min_temp = 1e-7
    cooling = 0.995
    max_iter = 50000

    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        if temp < min_temp:
            break

        # Random perturbation
        idx = rng.integers(N)
        perturb = rng.normal(0, 0.01, size=2)
        new_pts = current_pts.copy()
        new_pts[idx] += perturb
        new_pts = np.clip(new_pts, -0.4999, 0.4999)

        new_r = _score(new_pts)
        delta = new_r - current_r

        if delta > 0 or rng.random() < np.exp(delta / temp):
            current_pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = new_pts.copy()

        temp *= cooling

    return best_pts


def construct():
    t_start = time.time()

    # 1. Initialization - try diverse starting points
    best_pts = None
    best_r = 0.0

    inits = [
        _hex_init(jitter_scale=0.0, seed=0),
        _hex_init(jitter_scale=0.003, seed=1),
        _hex_init(jitter_scale=0.008, seed=2),
        _hex_init(jitter_scale=0.015, seed=3),
        _random_init(seed=42),
        _random_init(seed=123),
    ]

    for pts in inits:
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # 2. Force-Directed Relaxation
    remaining = DEADLINE - time.time()
    if remaining > 12:
        fd_budget = min(remaining * 0.12, 10.0)
        fd_pts = _force_directed(best_pts, fd_budget)
        fd_r = _score(fd_pts)
        if fd_r > best_r:
            best_r = fd_r
            best_pts = fd_pts.copy()

    # 3. L-BFGS-B Polish with multi-temperature annealing
    remaining = DEADLINE - time.time()
    if remaining > 20:
        lbfgs_budget = min(remaining * 0.45, 55.0)
        opt_pts = _optimize_lbfgs(best_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.4999, 0.4999)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    # 4. Active Set Polish
    remaining = DEADLINE - time.time()
    if remaining > 8:
        polish_budget = min(remaining * 0.2, 15.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    # 5. Simulated Annealing to escape local optima
    remaining = DEADLINE - time.time()
    if remaining > 15:
        sa_budget = min(remaining * 0.25, 20.0)
        sa_pts = _simulated_annealing(best_pts, sa_budget)
        sa_r = _score(sa_pts)
        if sa_r > best_r:
            best_r = sa_r
            best_pts = sa_pts.copy()

    # 6. Second L-BFGS-B pass
    remaining = DEADLINE - time.time()
    if remaining > 18:
        lbfgs_budget = min(remaining * 0.4, 35.0)
        opt_pts = _optimize_lbfgs(best_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.4999, 0.4999)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    # 7. Active Set Polish again
    remaining = DEADLINE - time.time()
    if remaining > 8:
        polish_budget = min(remaining * 0.25, 12.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    # 8. High-precision polish with very high alpha
    remaining = DEADLINE - time.time()
    if remaining > 10:
        hp_budget = min(remaining * 0.45, 28.0)
        hp_pts = _high_precision_polish(best_pts, hp_budget)
        hp_pts = np.clip(hp_pts, -0.49999, 0.49999)
        hp_r = _score(hp_pts)
        if hp_r > best_r:
            best_r = hp_r
            best_pts = hp_pts.copy()

    # 9. Final active set polish
    remaining = DEADLINE - time.time()
    if remaining > 5:
        polish_budget = min(remaining * 0.5, 10.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    # 10. Final safety check
    if best_pts is None:
        best_pts = _hex_init()

    best_pts = np.clip(best_pts, -0.49999, 0.49999)

    return best_pts

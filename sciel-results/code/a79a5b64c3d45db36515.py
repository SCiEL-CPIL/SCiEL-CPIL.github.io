"""Pack 300 equal circles in the unit disk.

Strategy: combine a robust multi-rotation hexagonal initialization with a
soft-min (log-sum-exp) gradient-based optimizer, plus a final force-directed
polish. A deadline guard returns the best valid configuration found.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

N = 300
DEADLINE = 150.0
MARGIN = 5.0
RNG_SEED = 12345


def _clearance(x):
    """Return (min_clearance, feasible_flag) for a configuration."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (N, 2) or not np.all(np.isfinite(x)):
        return 0.0, False
    d = np.linalg.norm(x, axis=1)
    bound = 1.0 - d
    if np.any(bound <= 0.0):
        return 0.0, False
    tree = cKDTree(x)
    dd, _ = tree.query(x, k=2)
    min_pair = dd[:, 1].min()
    if min_pair <= 0.0:
        return 0.0, False
    return min(0.5 * min_pair, bound.min()), True


def _soft_min_logsumexp(values, T):
    """Stable soft minimum: -T * logsumexp(-values/T)"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum


def _objective_and_grad(x, T):
    """
    Objective: minimize -softmin(constraints)
    Returns (objective_value, gradient_vector)
    """
    pts = x.reshape(-1, 2)
    n = len(pts)

    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms

    # Pairwise distances - use cdist for efficiency
    d = cdist(pts, pts)
    iu = np.triu_indices(n, k=1)
    pair_half = d[iu] / 2.0

    all_vals = np.concatenate([pair_half, boundary])

    obj = -_soft_min_logsumexp(all_vals, T)

    # Gradient
    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_all = -weights

    grad_pts = np.zeros_like(pts)

    # Boundary gradient
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[len(pair_half):][:, np.newaxis]
    grad_pts += grad_boundary

    # Pairwise gradient
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]

    W = np.zeros((n, n))
    n_pairs = len(pair_half)
    W[iu] = grad_all[:n_pairs]
    W[iu[1], iu[0]] = grad_all[:n_pairs]

    weighted_unit = W[:, :, np.newaxis] * unit_vecs
    grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)
    grad_pts += grad_pairs

    return obj, grad_pts.ravel()


def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    for attempt in range(14):
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.95 + 0.015 * attempt)

        k = int(1.0 / spacing) + 5
        rows = []
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                r = np.hypot(x_rot, y_rot)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))

        if len(rows) < n_points:
            continue

        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]

        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.04, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]

        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]

    return best_pts


def _soft_min_optimize(x, t0, deadline, T_start, T_end, n_steps, lr_start, lr_end):
    """Gradient-based optimization using soft-min objective."""
    x = x.copy()
    best_x = x.copy()
    best_c = _clearance(x)[0]

    for it in range(n_steps):
        if time.time() > deadline:
            break

        frac = it / max(n_steps - 1, 1)
        T = T_start + (T_end - T_start) * frac
        lr = lr_start + (lr_end - lr_start) * frac

        obj, grad = _objective_and_grad(x, T)
        x = x - lr * grad

        # Project back inside disk
        r = np.linalg.norm(x, axis=1)
        mask = r > 1.0
        if np.any(mask):
            x[mask] = x[mask] / r[mask, None] * 0.999

        # Check improvement every 10 steps
        if it % 10 == 0:
            c, ok = _clearance(x)
            if ok and c > best_c:
                best_c = c
                best_x = x.copy()

    return best_x, best_c


def _force_relax(pts, n_steps, deadline, lr_start=0.02, cooldown=0.998):
    """Fast force-directed relaxation with overlap-based repulsion."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _clearance(best_pos)[0]
    n = len(pos)

    r_est = best_r
    lr = lr_start

    for step in range(n_steps):
        if time.time() > deadline:
            break

        forces = np.zeros_like(pos)

        # Pairwise repulsion
        tree = cKDTree(pos)
        dd, idx = tree.query(pos, k=2)

        target = 2.0 * r_est
        for i in range(n):
            j = idx[i, 1]
            diff = pos[i] - pos[j]
            dist = dd[i, 1]
            if dist < 1e-12:
                diff = np.array([1e-6, 0.0])
                dist = 1e-6

            if dist < target:
                excess = target - dist
                force_mag = 4.0 * excess**3 / dist
                unit_vec = diff / dist
                forces[i] += force_mag * unit_vec

        # Boundary force
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.03
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess_b = norms[i] - (1.0 - wall_margin)
                if excess_b > 0:
                    f_mag = 50.0 * excess_b**3
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction

        pos = pos + lr * forces

        # Project back inside disk
        r = np.linalg.norm(pos, axis=1)
        mask = r > 1.0
        if np.any(mask):
            pos[mask] = pos[mask] / r[mask, None] * 0.999

        lr *= cooldown

        c, ok = _clearance(pos)
        if ok and c > best_r:
            best_r = c
            best_pos = pos.copy()

    return best_pos, best_r


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN

    # Start from best hexagonal initialization
    x = _hex_init(N, seed=RNG_SEED)
    if x.shape[0] < N:
        # Fallback to random
        rng = np.random.default_rng(RNG_SEED)
        pts = []
        while len(pts) < N:
            a = rng.uniform(0, 2 * np.pi)
            r = np.sqrt(rng.uniform(0, 1))
            pts.append((r * np.cos(a), r * np.sin(a)))
        x = np.array(pts)

    best = _clearance(x)[0]
    best_x = x.copy()

    # Phase 1: Soft-min gradient optimization (coarse to fine)
    # T controls the "softness" of the minimum - start high, end low
    phases = [
        (5.0, 2.0, 400, 0.05, 0.01),   # T: 5->2, 400 steps, lr: 0.05->0.01
        (2.0, 0.5, 500, 0.02, 0.005),  # T: 2->0.5, 500 steps, lr: 0.02->0.005
        (0.5, 0.1, 600, 0.01, 0.002),  # T: 0.5->0.1, 600 steps, lr: 0.01->0.002
    ]

    for T_start, T_end, n_steps, lr_start, lr_end in phases:
        if time.time() > deadline:
            break
        x, c = _soft_min_optimize(x, t0, deadline, T_start, T_end, n_steps, lr_start, lr_end)
        if c > best:
            best = c
            best_x = x.copy()

    # Phase 2: Force-directed polish
    if time.time() < deadline:
        x, c = _force_relax(best_x, 500, deadline, lr_start=0.01, cooldown=0.997)
        if c > best:
            best = c
            best_x = x.copy()

    # Phase 3: Final fine soft-min polish
    if time.time() < deadline:
        x, c = _soft_min_optimize(best_x, t0, deadline, 0.1, 0.02, 300, 0.003, 0.0005)
        if c > best:
            best = c
            best_x = x.copy()

    return best_x


if __name__ == "__main__":
    pts = construct()
    r, ok = _clearance(pts)
    print("radius:", r, "feasible:", ok)

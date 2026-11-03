"""Improved packing: soft-min epigraph objective with true gradients + force-directed phase."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, cdist

N = 200

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _compute_radius(centers):
    """Compute the maximum valid radius for given centers (exact)."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    return min(min_wall, min_pair / 2.0)


def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    best_centers = None
    best_r = 0.0
    for n_rows in range(10, 26):
        n_cols = int(np.ceil(N / n_rows))
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = -0.5 + r + j * s + (i % 2) * s / 2
                y = -0.5 + r + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        centers = np.array(centers[:N])
        # Center the grid
        cx = (centers[:, 0].max() + centers[:, 0].min()) / 2
        cy = (centers[:, 1].max() + centers[:, 1].min()) / 2
        centers[:, 0] -= cx
        centers[:, 1] -= cy
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            # scale to fit
            margin = 1e-9
            xs = centers[:, 0]
            ys = centers[:, 1]
            span_x = xs.max() - xs.min()
            span_y = ys.max() - ys.min()
            scale = min((1.0 - 2 * margin) / max(span_x, 1e-12),
                        (1.0 - 2 * margin) / max(span_y, 1e-12), 1.0)
            centers *= scale
        r_actual = _compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    return best_centers


def _objective_and_grad(centers, t, beta):
    """
    Epigraph smooth objective. We want to maximize t subject to:
      pairwise: d_ij >= 2 t  ->  margin_ij = d_ij - 2 t >= 0
      wall:     w_i  >= t    ->  margin_i  = w_i  - t   >= 0
    We minimize  F = soft_min_all(margins)  (want F >= 0).
    Return (F, grad_centers) where grad_centers is dF/dcenters (to ascend t,
    we move centers to increase F).
    """
    # Pairwise margins
    d = pdist(centers)  # shape (n_pairs,)
    m_pair = d - 2.0 * t  # (n_pairs,)
    # Wall margins
    wall = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)  # (N,)
    m_wall = wall - t  # (N,)

    # Combined soft min over all margins
    all_m = np.concatenate([m_pair, m_wall])
    mn = np.min(all_m)
    # weights for soft min: softmax of -beta*m
    z = -beta * (all_m - mn)
    # numerical stability
    z -= np.max(z)
    w = np.exp(z)
    w /= np.sum(w)

    F = mn - np.log(np.sum(w)) / beta

    # Gradient of F w.r.t. centers
    # dF/d m_pair[p] = w_pair[p] / beta
    # dF/d m_wall[i] = w_wall[i] / beta
    n_pairs = len(m_pair)
    w_pair = w[:n_pairs] / beta
    w_wall = w[n_pairs:] / beta

    grad = np.zeros_like(centers)

    # Pairwise contribution: m_pair[p] = d_ij - 2t, d_ij = |c_i - c_j|
    # d d_ij / d c_i = (c_i - c_j)/d_ij ; d / d c_j = (c_j - c_i)/d_ij
    # Build index pairs
    ii, jj = np.triu_indices(N, k=1)
    diff = centers[ii] - centers[jj]  # (n_pairs, 2)
    dist = np.maximum(d, 1e-12)
    unit = diff / dist[:, None]  # (n_pairs, 2)
    # grad contribution: sum_p w_pair[p] * (unit[p] for i, -unit[p] for j)
    np.add.at(grad, ii, w_pair[:, None] * unit)
    np.add.at(grad, jj, -w_pair[:, None] * unit)

    # Wall contribution: m_wall[i] = w_i - t where w_i = min(0.5 - c_i, c_i + 0.5)
    # The min is over the 4 walls. We approximate by using the actual closest wall.
    # d w_i / d c_i,k = +1 if closest wall is right/top (0.5 - c), -1 if left/bottom (c+0.5)
    for i in range(N):
        if w_wall[i] == 0.0:
            continue
        for k in range(2):
            dist_left = 0.5 + centers[i, k]
            dist_right = 0.5 - centers[i, k]
            if dist_left <= dist_right:
                grad[i, k] += w_wall[i]  # d(c_i+0.5)/dc_i = 1
            else:
                grad[i, k] -= w_wall[i]  # d(0.5-c_i)/dc_i = -1

    return F, grad


def _run_optimization(centers, t, max_iter, lr, beta):
    """Gradient ascent on t, gradient descent-ascent on centers to raise F."""
    centers = centers.copy()
    best_t = t
    best_centers = centers.copy()
    no_improve = 0
    cur_lr = lr

    for it in range(max_iter):
        if _time_left() < 15.0:
            break
        F, grad = _objective_and_grad(centers, t, beta)
        # Move centers in +grad direction (increase F)
        centers = centers + cur_lr * grad
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        # Increase t a little (we are trying to push the radius up)
        t = t + cur_lr * 0.5
        # Check actual radius
        r_actual = _compute_radius(centers)
        if r_actual > best_t + 1e-13:
            best_t = r_actual
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            # If we can't improve, ease t back toward feasibility
            if no_improve > 20:
                t = max(t - cur_lr, best_t)
                cur_lr *= 0.97
                no_improve = 0
                if cur_lr < 1e-8:
                    break
    return best_centers, best_t


def _force_directed_phase(centers, max_iter, beta):
    """
    Force-directed relaxation: repulsion between all pairs (soft) + wall repulsion.
    This is a different search move that helps escape local minima.
    """
    centers = centers.copy()
    best_t = _compute_radius(centers)
    best_centers = centers.copy()
    lr = 0.002

    for it in range(max_iter):
        if _time_left() < 20.0:
            break
        # Pairwise repulsion (inverse distance, softened)
        d = pdist(centers)
        # Force magnitude ~ 1/(d+eps) for d < 2r_target
        # Use soft repulsion: f = 1/(d + 0.05)
        ii, jj = np.triu_indices(N, k=1)
        diff = centers[ii] - centers[jj]
        dist = np.maximum(d, 1e-12)
        # Only repel when close (within ~ 2.5 * current radius)
        r_cur = best_t
        mask = dist < 2.5 * max(r_cur, 0.03)
        force_mag = np.zeros(len(d))
        force_mag[mask] = 1.0 / (dist[mask] + 0.03)
        unit = diff / dist[:, None]
        grad = np.zeros_like(centers)
        np.add.at(grad, ii, force_mag[:, None] * unit)
        np.add.at(grad, jj, -force_mag[:, None] * unit)
        # Wall repulsion
        for i in range(N):
            for k in range(2):
                dl = 0.5 + centers[i, k]
                dr = 0.5 - centers[i, k]
                if dl < 0.1:
                    grad[i, k] += 1.0 / (dl + 0.02)
                if dr < 0.1:
                    grad[i, k] -= 1.0 / (dr + 0.02)
        gn = np.linalg.norm(grad)
        if gn > 1e-12:
            centers += lr * grad / gn * 0.01
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        r_actual = _compute_radius(centers)
        if r_actual > best_t + 1e-13:
            best_t = r_actual
            best_centers = centers.copy()
        lr *= 0.995
    return best_centers, best_t


def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Start from multiple initializations, keep the best
    best_centers = None
    best_r = 0.0

    inits = [_generate_hex_grid()]
    # A couple of perturbed hex starts for diversity
    rng = np.random.default_rng(12345)
    for _ in range(3):
        c = _generate_hex_grid().copy()
        c += rng.normal(0, 0.003, size=c.shape)
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
        inits.append(c)

    for init in inits:
        if _time_left() < 25.0:
            break
        r0 = _compute_radius(init)
        # Phase A: gradient epigraph optimization
        centers, r = _run_optimization(init, r0, max_iter=2500, lr=0.0015, beta=40.0)
        # Phase B: force-directed escape
        if _time_left() > 20.0:
            centers, r = _force_directed_phase(centers, max_iter=1500, beta=40.0)
        # Phase C: high-precision polish (tighter, more iterations, higher beta)
        if _time_left() > 18.0:
            centers, r = _run_optimization(centers, r, max_iter=3000, lr=0.0004, beta=120.0)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    if best_centers is None:
        best_centers = _generate_hex_grid()

    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

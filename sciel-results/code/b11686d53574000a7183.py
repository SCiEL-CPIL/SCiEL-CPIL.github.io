"""Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Entry point: construct() -> (N,3) array of sphere centres."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_MARGIN = 15.0  # safety margin before 150s kill


def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    return min(min_pair / 2.0, min_wall)


def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    ss = np.sum(c * c, axis=1)
    d2 = ss[:, None] + ss[None, :] - 2.0 * c @ c.T
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, np.inf)
    return d


def _softmin_radius_and_grad(c, t):
    """Smooth approximation of the true radius via log-sum-exp, with gradient.
    Returns (smooth_radius, gradient of -smooth_radius w.r.t. c)."""
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    # All constraints: pair distances/2 and wall distances
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    m = np.min(vals)
    exp_vals = np.exp(-(vals - m) / t)
    s = np.sum(exp_vals)
    r = m - t * np.log(s)
    
    # Gradient computation
    # For pair (i,j): contribution to c_i is -0.5 * (c_i - c_j)/d_ij * exp_val, etc.
    # For wall: contribution depends on sign
    grad = np.zeros((N, 3))
    exp_pair = np.exp(-(pair - m) / t)
    exp_wall = np.exp(-(wall - m) / t)
    
    # Pair contributions
    diff = c[:, None, :] - c[None, :, :]
    # For each pair (i,j), weight = exp_pair[i,j] / s
    # d(pair_ij)/d(c_i) = (c_i - c_j) / d_ij
    weights = exp_pair / s
    weights = np.where(np.isfinite(weights), weights, 0)
    # grad from pairs: sum_j weights[i,j] * (c_i - c_j)/d_ij
    for k in range(3):
        # Use broadcasting
        d_safe = np.maximum(d, 1e-10)
        contrib = np.sum(weights * diff[:, :, k] / d_safe[:, :, None] * np.ones((1, 1, 3)), axis=1)
        # Actually simpler:
    # Let me redo this more carefully with vectorized ops
    grad = np.zeros((N, 3))
    d_safe = np.maximum(d, 1e-10)
    # For pair (i,j): d/d(c_i) [1/2 * d_ij] = (c_i - c_j) / (2 * d_ij)
    # Weight for pair (i,j) in softmin: exp(-(pair_ij - m)/t) / s
    w_pair = exp_pair / s
    w_pair = np.where(np.isfinite(w_pair), w_pair, 0)
    np.fill_diagonal(w_pair, 0)
    
    for k in range(3):
        # sum_j w_pair[i,j] * (c_i[k] - c_j[k]) / d_ij
        contrib = np.sum(w_pair * diff[:, :, k] / d_safe, axis=1)
        grad[:, k] -= contrib  # because we're minimizing -r, and dr/dc_i = -sum(...)
    
    # Wall contributions
    # d/d(c_i[k]) [0.5 - |c_i[k]|] = -sign(c_i[k]) if c_i[k] != 0
    w_wall = exp_wall / s
    w_wall = np.where(np.isfinite(w_wall), w_wall, 0)
    for k in range(3):
        signs = np.sign(c[:, k])
        signs[signs == 0] = 1.0
        grad[:, k] -= w_wall * signs  # dr/dc = -sign, so d(-r)/dc = sign * w_wall
    
    return r, grad


def _epigraph_objective(x):
    """Epigraph formulation: minimize t subject to all constraints <= t.
    x = [c_0, c_1, ..., c_{N-1}, t] where c_i are (3,) and t is scalar."""
    c = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    # Penalty for violations
    d = _pair_dist_matrix(c)
    pair_viol = np.maximum(0, t - d/2.0)
    wall = 0.5 - np.abs(c)
    wall_viol = np.maximum(0, t - wall)
    
    penalty = 1e4 * (np.sum(pair_viol**2) + np.sum(wall_viol**2))
    return t + penalty


def _epigraph_grad(x):
    """Gradient of epigraph objective."""
    c = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    grad = np.zeros_like(x)
    grad[N*3] = 1.0  # dt/dt = 1
    
    d = _pair_dist_matrix(c)
    pair_half = d / 2.0
    wall = 0.5 - np.abs(c)
    
    # Pair violations
    pair_viol = np.maximum(0, t - pair_half)
    for i in range(N):
        for j in range(i+1, N):
            if pair_viol[i, j] > 0:
                diff = c[i] - c[j]
                d_ij = max(d[i, j], 1e-10)
                # d/d(c_i) [ (t - d_ij/2)^2 ] = 2*(t - d_ij/2) * (-1/2) * (c_i - c_j)/d_ij
                factor = 2.0 * pair_viol[i, j] * (-0.5) / d_ij
                grad[i*3:(i+1)*3] += 1e4 * factor * diff
                grad[j*3:(j+1)*3] -= 1e4 * factor * diff
    
    # Wall violations
    for i in range(N):
        for k in range(3):
            if wall[i, k] < t:
                sign = np.sign(c[i, k])
                if sign == 0:
                    sign = 1.0
                # d/d(c_i[k]) [ (t - (0.5 - |c_i[k]|))^2 ] = 2*(t - wall) * sign
                factor = 2.0 * (t - wall[i, k]) * sign
                grad[i*3+k] += 1e4 * factor
    
    return grad


def _generate_initial_positions():
    """Generate a good initial configuration using FCC lattice points closest to origin."""
    points = []
    max_idx = 6
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])

    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    pts = points[idx[:N]].copy()
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        scale = 0.42 / max_coord
        pts *= scale
    return pts


def _generate_grid_initial():
    """Generate initial positions on a perturbed grid."""
    n_per_side = 4
    grid = np.array([[i, j, k] for i in range(n_per_side)
                     for j in range(n_per_side)
                     for k in range(n_per_side)], dtype=float)
    grid = (grid / (n_per_side - 1) - 0.5) * 0.85
    rng = np.random.default_rng(123)
    grid += rng.uniform(-0.02, 0.02, size=grid.shape)
    return grid[:N].copy()


def _generate_random_initial(seed=42):
    """Generate random initial positions."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.4, 0.4, size=(N, 3))
    return pts


def _lloyd_optimize(c, deadline):
    """Voronoi-Lloyd equalization: move each center toward its Voronoi cell centroid."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)

    for damp in (0.5, 0.3, 0.15, 0.08, 0.04, 0.02):
        if time.time() > deadline:
            break
        steps = max(20, int(100 * damp))
        for _ in range(steps):
            if time.time() > deadline:
                break
            try:
                from scipy.spatial import Delaunay
                tri = Delaunay(c)
                simplices = tri.simplices
                cell_centroids = np.zeros((N, 3))
                cell_counts = np.zeros(N)

                for si in simplices:
                    v0, v1, v2, v3 = c[si[0]], c[si[1]], c[si[2]], c[si[3]]
                    A = np.array([v1 - v0, v2 - v0, v3 - v0])
                    b = np.array([
                        np.dot(v1, v1) - np.dot(v0, v0),
                        np.dot(v2, v2) - np.dot(v0, v0),
                        np.dot(v3, v3) - np.dot(v0, v0)
                    ])
                    try:
                        cc = np.linalg.solve(2.0 * A, b)
                    except np.linalg.LinAlgError:
                        cc = np.mean([v0, v1, v2, v3])

                    for vi in si:
                        cell_centroids[vi] += cc
                        cell_counts[vi] += 1

                valid = cell_counts > 0
                if np.any(valid):
                    targets = cell_centroids[valid] / cell_counts[valid, None]
                    c[valid] = c[valid] + damp * (targets - c[valid])

                c = np.clip(c, -0.49, 0.49)
            except Exception:
                # Fallback: simple repulsion
                d = _pair_dist_matrix(c)
                inv = 1.0 / np.maximum(d, 1e-6) ** 2
                diff = c[:, None, :] - c[None, :, :]
                F = np.sum(diff * inv[:, :, None], axis=1)
                c = np.clip(c + 0.005 * F, -0.49, 0.49)
                continue

        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c.copy()

    return best, best_r


def _repulsion_anneal(c, deadline, steps=2000):
    """Vectorized repulsion + wall-attraction gradient descent."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)

    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv[:, :, None], axis=1)

        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))

        frac = it / max(steps - 1, 1)
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        c = c + lr * (rep_scale * F + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)

    return c


def _smooth_polish(c, deadline, fine=False):
    """Polish with smooth softmin max-min via L-BFGS-B, annealing temperature."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)

    if fine:
        temps = (0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001)
        maxiter = 5000
    else:
        temps = (0.05, 0.02, 0.008, 0.003, 0.001, 0.0005, 0.0001)
        maxiter = 1500

    for t in temps:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                lambda x: -_softmin_radius_and_grad(x.reshape(N, 3), t)[0],
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-14,
                         'maxls': 80},
            )
            c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c.copy()
            x0 = c.flatten()
        except Exception:
            pass
    return best, best_r


def _local_search(c, deadline, iters=3000):
    """Local search: perturb multiple binding constraints simultaneously, accept if radius improves."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(42)

    for it in range(iters):
        if time.time() > deadline:
            break

        d = _pair_dist_matrix(c)
        wall = 0.5 - np.abs(c)

        min_pair_dist = np.min(d)
        min_wall_val = np.min(wall)
        current_r = min(min_pair_dist / 2.0, min_wall_val)

        base_step = 0.002 * max(current_r, 0.01)
        base_step = np.clip(base_step, 0.0005, 0.004)

        pair_threshold = min_pair_dist * 1.10
        pair_mask = (d < pair_threshold) & (d > 0)
        pair_indices = np.argwhere(pair_mask)
        pair_indices = pair_indices[pair_indices[:, 0] < pair_indices[:, 1]]

        wall_threshold = min_wall_val * 1.10
        wall_mask = wall < wall_threshold
        wall_indices = np.argwhere(wall_mask)

        if min_pair_dist / 2.0 < min_wall_val:
            if len(pair_indices) > 0:
                n_move = min(3, len(pair_indices))
                chosen = rng.choice(len(pair_indices), size=n_move, replace=False)
                for ci in chosen:
                    i, j = pair_indices[ci]
                    direction = c[i] - c[j]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-12:
                        direction /= norm
                        step = base_step * rng.uniform(0.5, 1.5)
                        c[i] += step * direction
                        c[j] -= step * direction
        else:
            if len(wall_indices) > 0:
                n_move = min(3, len(wall_indices))
                chosen = rng.choice(len(wall_indices), size=n_move, replace=False)
                for ci in chosen:
                    idx, axis = wall_indices[ci]
                    step = base_step * rng.uniform(0.5, 1.5)
                    if c[idx, axis] > 0:
                        c[idx, axis] -= step
                    else:
                        c[idx, axis] += step

        c = np.clip(c, -0.499, 0.499)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    return best_c, best_r


def _metropolis_mc(c, deadline, steps=5000, temp0=0.01, temp_min=0.001):
    """Metropolis Monte Carlo: random perturbations with acceptance criterion."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(77)

    for it in range(steps):
        if time.time() > deadline:
            break

        frac = it / max(steps - 1, 1)
        temp = temp0 * (temp_min / temp0) ** frac

        i = rng.integers(0, N)
        pert_scale = 0.005 * (1.0 + best_r)
        delta = rng.normal(0, pert_scale, size=3)
        c_new = c.copy()
        c_new[i] += delta
        c_new = np.clip(c_new, -0.49, 0.49)

        r_new = _compute_radius(c_new)
        r_cur = _compute_radius(c)

        if r_new >= r_cur:
            c = c_new
        else:
            prob = np.exp((r_new - r_cur) / max(temp, 1e-10))
            if rng.random() < prob:
                c = c_new

        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()

    return best_c, best_r


def _epigraph_optimize(c, deadline, maxiter=3000):
    """Optimize using epigraph formulation with L-BFGS-B."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    
    # Initial t estimate
    t0 = _compute_radius(c) + 0.01
    x0 = np.concatenate([c.flatten(), [t0]])
    
    bounds = [(-0.49, 0.49)] * (N * 3) + [(0.0, 0.5)]
    
    try:
        res = minimize(
            _epigraph_objective,
            x0,
            jac=_epigraph_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        c_new = np.clip(res.x[:N*3].reshape(N, 3), -0.49, 0.49)
        r = _compute_radius(c_new)
        if r > best_r:
            best_r = r
            best = c_new.copy()
    except Exception:
        pass
    
    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin

    # 1. Initialize with multiple strategies
    best_pts = _generate_initial_positions()
    best_r = _compute_radius(best_pts)

    grid_pts = _generate_grid_initial()
    grid_r = _compute_radius(grid_pts)
    if grid_r > best_r:
        best_r = grid_r
        best_pts = grid_pts

    # 2. Voronoi-Lloyd equalization
    if time.time() < deadline:
        pts, r = _lloyd_optimize(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 3. Repulsion anneal
    if time.time() < deadline:
        pts = _repulsion_anneal(best_pts, deadline, steps=1500)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 4. Epigraph optimization
    if time.time() < deadline:
        pts, r = _epigraph_optimize(best_pts, deadline, maxiter=2000)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 5. Smooth max-min polish
    if time.time() < deadline:
        pts, r = _smooth_polish(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 6. Metropolis MC to escape local optima
    if time.time() < deadline:
        pts, r = _metropolis_mc(best_pts, deadline, steps=2000)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 7. Local search on binding constraints
    if time.time() < deadline:
        pts, r = _local_search(best_pts, deadline, iters=2000)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 8. Fine smooth polish if time remains
    if time.time() < deadline:
        pts, r = _smooth_polish(best_pts, deadline, fine=True)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 9. Final local search with tighter steps
    if time.time() < deadline:
        pts, r = _local_search(best_pts, deadline, iters=1000)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 10. Final fine polish
    if time.time() < deadline:
        pts, r = _smooth_polish(best_pts, deadline, fine=True)
        if r > best_r:
            best_r = r
            best_pts = pts

    # Ensure final result is valid
    final_r = _compute_radius(best_pts)
    if final_r > best_r:
        best_r = final_r

    return best_pts
# EVOLVE-BLOCK-END

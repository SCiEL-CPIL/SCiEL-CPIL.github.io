"""Combined solver: FCC initialization + Voronoi-Lloyd equalization + smooth max-min polish + local search.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5,0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


def _generate_initial_positions():
    """Generate a good initial configuration using FCC lattice points closest to origin."""
    points = []
    max_idx = 6
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC basis
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])

    points = np.array(points)

    # Center at origin
    points -= np.mean(points, axis=0)

    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)

    # Take N closest
    pts = points[idx[:N]].copy()

    # Scale to fit inside cube with margin
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        scale = 0.42 / max_coord
        pts *= scale

    return pts


def _generate_grid_initial():
    """Generate initial positions on a perturbed grid for diversity."""
    # Use a 4x4x4 grid (64 points), take 50
    n_per_side = 4
    total = n_per_side ** 3
    grid = np.array([[i, j, k] for i in range(n_per_side)
                     for j in range(n_per_side)
                     for k in range(n_per_side)], dtype=float)
    grid = (grid / (n_per_side - 1) - 0.5) * 0.85  # map to [-0.425, 0.425]

    rng = np.random.default_rng(123)
    # Add small random perturbation
    grid += rng.uniform(-0.02, 0.02, size=grid.shape)

    # Take first N points
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

    for damp in (0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06, 0.03):
        if time.time() > deadline:
            break
        steps = max(30, int(200 * damp))
        for _ in range(steps):
            if time.time() > deadline:
                break
            try:
                tri = Delaunay(c)
            except Exception:
                # Fallback: simple repulsion
                d = _pair_dist_matrix(c)
                inv = 1.0 / np.maximum(d, 1e-6) ** 2
                diff = c[:, None, :] - c[None, :, :]
                F = np.sum(diff * inv[:, :, None], axis=1)
                c = np.clip(c + 0.005 * F, -0.49, 0.49)
                continue

            simplices = tri.simplices
            cell_centroids = np.zeros((N, 3))
            cell_counts = np.zeros(N)

            for si in simplices:
                v0, v1, v2, v3 = c[si[0]], c[si[1]], c[si[2]], c[si[3]]
                # Circumcenter
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

        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c.copy()

    return best, best_r


def _softmin_radius(c, t):
    """Smooth approximation of the true radius via log-sum-exp."""
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    m = np.min(vals)
    s = np.sum(np.exp(-(vals - m) / t))
    return m - t * np.log(s)


def _smooth_obj(x, t):
    """Objective to MINIMIZE: negative smooth radius plus boundary penalty."""
    c = x.reshape(N, 3)
    r = _softmin_radius(c, t)
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    return -r + 1e3 * pen


def _smooth_polish(c, deadline, fine=False):
    """Polish with smooth softmin max-min via L-BFGS-B, annealing temperature."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)

    if fine:
        temps = (0.003, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002)
        maxiter = 3000
    else:
        temps = (0.05, 0.02, 0.008, 0.003, 0.001, 0.0005, 0.0001)
        maxiter = 1500

    for t in temps:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12,
                         'maxls': 60},
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


def _local_search(c, deadline, iters=3000):
    """Local search: perturb binding constraints, accept if radius improves."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(42)

    for it in range(iters):
        if time.time() > deadline:
            break

        d = _pair_dist_matrix(c)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]

        wall = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        min_wall_val = wall[min_wall_idx]

        # Adaptive step size based on current radius
        base_step = 0.003 * (best_r / 0.135)
        base_step = np.clip(base_step, 0.001, 0.005)

        if min_pair_dist / 2.0 < min_wall_val:
            i, j = min_pair_idx
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
                # Push apart
                step = base_step * rng.uniform(0.5, 1.5)
                c[i] += step * direction
                c[j] -= step * direction
        else:
            idx = min_wall_idx
            axis = idx[1]
            # Push away from wall
            step = base_step * rng.uniform(0.5, 1.5)
            if c[idx[0], axis] > 0:
                c[idx[0], axis] -= step
            else:
                c[idx[0], axis] += step

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

        # Anneal temperature
        frac = it / max(steps - 1, 1)
        temp = temp0 * (temp_min / temp0) ** frac

        # Pick a random sphere and perturb it
        i = rng.integers(0, N)
        delta = rng.normal(0, 0.01, size=3)
        c_new = c.copy()
        c_new[i] += delta
        c_new = np.clip(c_new, -0.49, 0.49)

        r_new = _compute_radius(c_new)
        r_cur = _compute_radius(c)

        # Accept if better or with probability exp((r_new - r_cur) / temp)
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


def _jellium_energy(c, k_rep=1.0, k_wall=1.0, r_c=0.25):
    """Electrostatics-like energy: Riesz repulsion + harmonic confinement."""
    d = _pair_dist_matrix(c)
    # Soft Riesz kernel: 1/d^2 softened
    rep = np.sum(1.0 / (d ** 2 + r_c ** 2))
    # Harmonic wall confinement
    wall = np.sum((np.abs(c) - 0.5) ** 2)
    return k_rep * rep + k_wall * wall


def _jellium_gradient(c, k_rep=1.0, k_wall=1.0, r_c=0.25):
    """Gradient of jellium energy."""
    d = _pair_dist_matrix(c)
    diff = c[:, None, :] - c[None, :, :]
    # dE/dx_i = sum_j -2*(c_i - c_j) / (d_ij^2 + r_c^2)^2
    denom = (d ** 2 + r_c ** 2) ** 2
    denom = np.where(np.isfinite(denom), denom, 1e10)
    grad_rep = -2.0 * np.sum(diff / denom[:, :, None], axis=1)

    grad_wall = 2.0 * (np.abs(c) - 0.5) * np.sign(c)
    grad_wall = np.where(c == 0, 0, grad_wall)

    return k_rep * grad_rep + k_wall * grad_wall


def _jellium_optimize(c, deadline, steps=5000):
    """Optimize jellium energy via gradient descent with annealing."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)

    for it in range(steps):
        if time.time() > deadline:
            break

        frac = it / max(steps - 1, 1)
        k_rep = 1.0 * (1.0 - 0.5 * frac)
        k_wall = 1.0 * (0.5 + 1.0 * frac)
        lr = 1e-4 * (1.0 - 0.5 * frac)

        grad = _jellium_gradient(c, k_rep, k_wall)
        c = c - lr * grad
        c = np.clip(c, -0.49, 0.49)

        if it % 50 == 0:
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c.copy()

    r = _compute_radius(c)
    if r > best_r:
        best_r = r
        best = c.copy()

    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin

    # 1. Initialize with FCC-like cluster
    best_pts = _generate_initial_positions()
    best_r = _compute_radius(best_pts)

    # Also try grid initialization
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

    # 3. Jellium electrostatics optimization
    if time.time() < deadline:
        pts, r = _jellium_optimize(best_pts, deadline, steps=3000)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 4. Repulsion anneal
    if time.time() < deadline:
        pts = _repulsion_anneal(best_pts, deadline, steps=1500)
        r = _compute_radius(pts)
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
        pts, r = _metropolis_mc(best_pts, deadline, steps=3000)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 7. Local search on binding constraints
    if time.time() < deadline:
        pts, r = _local_search(best_pts, deadline, iters=2500)
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
        pts, r = _local_search(best_pts, deadline, iters=1500)
        if r > best_r:
            best_r = r
            best_pts = pts

    # Ensure final result is valid
    final_r = _compute_radius(best_pts)
    if final_r > best_r:
        best_r = final_r

    return best_pts
# EVOLVE-BLOCK-END

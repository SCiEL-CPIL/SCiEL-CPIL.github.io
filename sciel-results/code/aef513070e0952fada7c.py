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
    n_per_side = 4
    grid = np.array([[i, j, k] for i in range(n_per_side)
                     for j in range(n_per_side)
                     for k in range(n_per_side)], dtype=float)
    grid = (grid / (n_per_side - 1) - 0.5) * 0.85

    rng = np.random.default_rng(123)
    grid += rng.uniform(-0.02, 0.02, size=grid.shape)

    return grid[:N].copy()


def _generate_bcc_initial():
    """Generate initial positions from BCC lattice for diversity."""
    points = []
    max_idx = 4
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k + 0.5])

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


def _lloyd_optimize(c, deadline):
    """Voronoi-Lloyd equalization: move each center toward its Voronoi cell centroid."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)

    for damp in (0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06, 0.03):
        if time.time() > deadline:
            break
        steps = max(20, int(150 * damp))
        for _ in range(steps):
            if time.time() > deadline:
                break
            try:
                tri = Delaunay(c)
            except Exception:
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


def _softmin_radius_and_grad(c, t):
    """Smooth approximation of radius via log-sum-exp with analytical gradient.
    
    Uses unique pairs (i<j) to avoid double-counting.
    Returns (radius, gradient of radius w.r.t. c).
    """
    n = c.shape[0]
    # Pair distances (upper triangle only)
    ii, jj = np.triu_indices(n, 1)
    diff = c[ii] - c[jj]
    dists = np.sqrt(np.sum(diff * diff, axis=1) + 1e-16)
    pair_vals = dists / 2.0
    
    # Wall distances
    wall = 0.5 - np.abs(c)  # (n, 3)
    wall_vals = wall.ravel()
    
    # Combine all constraints
    all_vals = np.concatenate([pair_vals, wall_vals])
    n_constraints = len(all_vals)
    
    # Log-sum-exp for softmin: r = m - t * log(sum(exp(-(v-m)/t)))
    m = np.min(all_vals)
    exp_terms = np.exp(-(all_vals - m) / t)
    s = np.sum(exp_terms)
    r = m - t * np.log(s)
    
    # Gradient: dr/dv_k = -exp(-(v_k - m)/t) / sum(exp(-(v-m)/t))
    weights = exp_terms / s  # (n_constraints,)
    
    # Gradient w.r.t. c
    grad = np.zeros_like(c)
    
    # Pair contributions
    # For pair (i,j) with weight w: 
    #   d(pair_val)/d(c_i) = -(c_i - c_j) / (2 * dist)
    #   d(pair_val)/d(c_j) = +(c_i - c_j) / (2 * dist)
    #   dr/d(c_i) = -w * d(pair_val)/d(c_i) = w * (c_i - c_j) / (2 * dist)
    #   dr/d(c_j) = -w * d(pair_val)/d(c_j) = -w * (c_i - c_j) / (2 * dist)
    for k in range(len(ii)):
        i, j = ii[k], jj[k]
        w = weights[k]
        d = dists[k]
        if d > 1e-12:
            factor = w / (2.0 * d)
            grad[i] += factor * diff[k]
            grad[j] -= factor * diff[k]
    
    # Wall contributions
    # wall_val[i,k] = 0.5 - |c[i,k]|
    # d(wall_val)/d(c[i,k]) = -sign(c[i,k])
    # dr/d(c[i,k]) = -w * d(wall_val)/d(c[i,k]) = w * sign(c[i,k])
    wall_weights = weights[len(pair_vals):].reshape(n, 3)
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 1.0  # arbitrary sign at zero
    grad += wall_weights * signs
    
    return r, grad


def _smooth_obj_and_grad(x, t):
    """Objective to MINIMIZE: negative smooth radius plus boundary penalty.
    Returns (value, gradient).
    """
    c = x.reshape(N, 3)
    r, grad_r = _softmin_radius_and_grad(c, t)
    
    # Boundary penalty
    pen = 0.0
    grad_pen = np.zeros_like(c)
    outside = np.abs(c) - 0.5 + 1e-6
    mask = outside > 0
    if np.any(mask):
        pen = np.sum(outside[mask] ** 2)
        grad_pen[mask] = 2.0 * outside[mask] * np.sign(c[mask])
    
    value = -r + 1e3 * pen
    grad = -grad_r + 1e3 * grad_pen
    return value, grad


def _smooth_polish(c, deadline, fine=False):
    """Polish with smooth softmin max-min via L-BFGS-B with analytical gradients."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)

    if fine:
        temps = (0.003, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002)
        maxiter = 2000
    else:
        temps = (0.05, 0.02, 0.008, 0.003, 0.001, 0.0005, 0.0001)
        maxiter = 1000

    for t in temps:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_obj_and_grad,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                jac=True,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12,
                         'maxls': 40},
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
    """Local search: perturb binding constraints, accept if radius improves.
    Handles multiple near-binding constraints simultaneously.
    """
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

        current_r = min(min_pair_dist / 2.0, min_wall_val)
        
        # Find all near-binding constraints (within 10% of current radius)
        threshold = current_r * 1.1
        base_step = 0.002 * (best_r / 0.135)
        base_step = np.clip(base_step, 0.0008, 0.004)

        # Collect binding constraints
        moves = []
        
        # Binding pairs
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] / 2.0 <= threshold:
                    direction = c[i] - c[j]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-12:
                        direction /= norm
                        moves.append(('pair', i, j, direction))
        
        # Binding walls
        for i in range(N):
            for k in range(3):
                if wall[i, k] <= threshold:
                    axis = k
                    if c[i, axis] > 0:
                        direction = np.zeros(3)
                        direction[axis] = -1.0
                    else:
                        direction = np.zeros(3)
                        direction[axis] = 1.0
                    moves.append(('wall', i, -1, direction))

        if not moves:
            # No binding constraints; random small move
            i = rng.integers(0, N)
            delta = rng.normal(0, base_step, size=3)
            c[i] += delta
        else:
            # Choose a random binding constraint and move
            mtype, i, j, direction = moves[rng.integers(0, len(moves))]
            step = base_step * rng.uniform(0.5, 1.5)
            if mtype == 'pair':
                c[i] += step * direction
                c[j] -= step * direction
            else:
                c[i] += step * direction

        c = np.clip(c, -0.499, 0.499)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        elif rng.random() < 0.1:
            # Occasionally accept slightly worse to escape local optima
            pass

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
        delta = rng.normal(0, 0.01, size=3)
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

    bcc_pts = _generate_bcc_initial()
    bcc_r = _compute_radius(bcc_pts)
    if bcc_r > best_r:
        best_r = bcc_r
        best_pts = bcc_pts

    # 2. Voronoi-Lloyd equalization
    if time.time() < deadline:
        pts, r = _lloyd_optimize(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 3. Repulsion anneal
    if time.time() < deadline:
        pts = _repulsion_anneal(best_pts, deadline, steps=1000)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 4. Smooth max-min polish (with analytical gradients)
    if time.time() < deadline:
        pts, r = _smooth_polish(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 5. Local search on binding constraints
    if time.time() < deadline:
        pts, r = _local_search(best_pts, deadline, iters=2500)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 6. Second round: smooth polish again
    if time.time() < deadline:
        pts, r = _smooth_polish(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 7. Metropolis MC to escape local optima
    if time.time() < deadline:
        pts, r = _metropolis_mc(best_pts, deadline, steps=2000)
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

"""
Combined solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Combines:
- Multi-stage optimization (FCC-inspired, grid, random seeds)
- Repulsion-based simulated annealing (vectorized)
- Metropolis Monte Carlo phase
- Softmin gradient polishing with order-statistic targeting
- Local search on binding constraints
"""

import numpy as np
import time
from scipy.spatial.distance import cdist

N = 50
R_BEST = 0.1359545126417

# Precompute upper triangle indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)


def _compute_radius(c):
    """Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3."""
    if len(c) == 0:
        return 0.0
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _seed_fcc():
    """Generate an FCC-inspired initial configuration for 50 spheres."""
    m = 4
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    px = x + dx * spacing
                    py = y + dy * spacing
                    pz = z + dz * spacing
                    if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5 and -0.5 <= pz <= 0.5:
                        points.append([px, py, pz])
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    return points


def _seed_grid():
    """Generate a grid-based initial configuration."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]


def _seed_random():
    """Generate a random initial configuration."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))


def _seed_bcc():
    """Generate a BCC-inspired initial configuration."""
    m = 3
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                points.append([x, y, z])
                points.append([x + 0.5 * spacing, y + 0.5 * spacing, z + 0.5 * spacing])
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    return points


def _repulsion_force(c):
    """Compute repulsive forces on all centers (vectorized)."""
    forces = np.zeros_like(c)
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    with np.errstate(divide='ignore', invalid='ignore'):
        force_mag = 1.0 / (dists * dists)
    force_mag = np.where(force_mag > 1e6, 1e6, force_mag)
    unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    pair_forces = force_mag[:, np.newaxis] * unit_vecs
    for idx in range(_NUM_PAIRS):
        i, j = _PAIR_IDX[idx]
        forces[i] += pair_forces[idx]
        forces[j] -= pair_forces[idx]
    wall_dists = 0.5 - np.abs(c)
    with np.errstate(divide='ignore', invalid='ignore'):
        wall_force_mag = 1.0 / (wall_dists * wall_dists)
    wall_force_mag = np.where(wall_force_mag > 1e6, 1e6, wall_force_mag)
    wall_force_mag = np.where(np.isnan(wall_force_mag), 0, wall_force_mag)
    signs = np.sign(c)
    forces += wall_force_mag * signs
    return forces


def _physics_anneal(c_init, deadline, t_start=2.0, t_end=0.01, iters_per_temp=30):
    """Repulsion-based simulated annealing (vectorized)."""
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 12)
    for t in temps:
        if time.time() > deadline:
            break
        for it in range(iters_per_temp):
            if time.time() > deadline:
                break
            forces = _repulsion_force(c)
            noise = np.random.normal(0, t * 0.05, size=c.shape)
            c_new = c + forces * 0.005 + noise
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            delta_r = r_new - _compute_radius(c)
            if delta_r > 0 or np.random.random() < np.exp(-delta_r / max(t, 1e-8)):
                c = c_new
    return best_c, best_r


def _metropolis_mc(c_init, deadline, n_steps=2000, step_size=0.02):
    """Metropolis Monte Carlo to escape local minima."""
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    t = 0.01
    for step in range(n_steps):
        if time.time() > deadline:
            break
        idx = np.random.randint(0, N)
        c_new = c.copy()
        c_new[idx] += np.random.normal(0, step_size, size=3)
        c_new[idx] = np.clip(c_new[idx], -0.499999, 0.499999)
        r_new = _compute_radius(c_new)
        r_cur = _compute_radius(c)
        delta_r = r_new - r_cur
        if delta_r > 0 or np.random.random() < np.exp(-delta_r / max(t, 1e-8)):
            c = c_new
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        t = max(t * 0.999, 0.001)
    return best_c, best_r


def _softmin_radius_and_grad(c, t):
    """Compute softmin radius and its gradient."""
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0
    wall_vals = 0.5 - np.abs(c)
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    m_val = np.min(all_vals)
    pair_weights = np.exp(-pair_vals / t)
    wall_weights = np.exp(-wall_vals / t)
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    if total_sum < 1e-300:
        return m_val, np.zeros_like(c)
    softmin_val = -t * np.log(total_sum)
    grad_c = np.zeros_like(c)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    pair_grad = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
    for idx in range(_NUM_PAIRS):
        i, j = _PAIR_IDX[idx]
        grad_c[i] -= pair_grad[idx]
        grad_c[j] += pair_grad[idx]
    signs = np.sign(c)
    grad_c += wall_weights * signs
    grad_c /= total_sum
    return softmin_val, grad_c


def _gradient_polish(c, deadline, max_iter=200):
    """Gradient-based polishing using softmin objective."""
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    t = 0.01
    lr = 0.005
    for it in range(max_iter):
        if time.time() > deadline:
            break
        softmin_val, grad = _softmin_radius_and_grad(c, t)
        c_new = c + lr * grad
        c_new = np.clip(c_new, -0.499999, 0.499999)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new
        else:
            lr *= 0.9
            if lr < 1e-6:
                break
        t = max(t * 0.95, 0.001)
    return best_c, best_r


def _local_search(c, deadline, max_iter=100):
    """Local search focusing on binding constraints."""
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    for it in range(max_iter):
        if time.time() > deadline:
            break
        r = _compute_radius(c)
        if r >= R_BEST:
            break
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_idx = np.argmin(dists)
        i, j = _PAIR_IDX[min_pair_idx]
        wall_dists = 0.5 - np.abs(c)
        min_wall_idx = np.argmin(wall_dists)
        if dists[min_pair_idx] / 2.0 < wall_dists[min_wall_idx]:
            # Pair constraint is binding
            d = c[i] - c[j]
            dist = np.linalg.norm(d)
            if dist > 1e-12:
                direction = d / dist
                push = 0.001
                c[i] += push * direction
                c[j] -= push * direction
        else:
            # Wall constraint is binding
            idx = min_wall_idx
            coord = np.argmin(0.5 - np.abs(c[idx]))
            if c[idx, coord] > 0:
                c[idx, coord] -= 0.001
            else:
                c[idx, coord] += 0.001
        c = np.clip(c, -0.499999, 0.499999)
        r_new = _compute_radius(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    np.random.seed(42)

    # Generate multiple seed configurations
    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_random(),
        _seed_bcc(),
    ]

    best_c = None
    best_r = 0.0

    for seed in seeds:
        if time.time() > deadline - 30:
            break

        # Phase 1: Physics annealing
        c1, r1 = _physics_anneal(seed, deadline - 40)
        if r1 > best_r:
            best_r = r1
            best_c = c1.copy()

        # Phase 2: Metropolis MC
        if time.time() > deadline - 25:
            break
        c2, r2 = _metropolis_mc(c1, deadline - 20, n_steps=1500)
        if r2 > best_r:
            best_r = r2
            best_c = c2.copy()

        # Phase 3: Gradient polish
        if time.time() > deadline - 10:
            break
        c3, r3 = _gradient_polish(c2, deadline - 5, max_iter=150)
        if r3 > best_r:
            best_r = r3
            best_c = c3.copy()

        # Phase 4: Local search on binding constraints
        if time.time() > deadline - 3:
            break
        c4, r4 = _local_search(c3, deadline - 1, max_iter=80)
        if r4 > best_r:
            best_r = r4
            best_c = c4.copy()

    # Final validity check
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = _seed_grid()
    best_c = np.clip(best_c, -0.5, 0.5)
    final_r = _compute_radius(best_c)

    if final_r <= 0:
        return _seed_grid()

    return best_c

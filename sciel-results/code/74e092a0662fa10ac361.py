"""Optimized packing of 50 equal spheres in a unit cube using Voronoi-based Lloyd + smooth gradient polish."""
import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist

N = 50
CUBE_BOUND = 0.5
DEADLINE = 150.0
MARGIN = 5.0


def _compute_radius(c):
    """Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3."""
    if len(c) == 0:
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    if c.shape != (N, 3):
        return 0.0
    if np.any(np.abs(c) > CUBE_BOUND + 1e-12):
        return 0.0

    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)

    wall_dists = CUBE_BOUND - np.abs(c)
    min_wall = np.min(wall_dists)

    r = min(min_pair / 2.0, min_wall)
    return max(0.0, r)


def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient using log-sum-exp.
    Uses only unique pairs (i < j) to avoid double-counting.
    Returns (radius, gradient) where gradient is (N, 3).
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)

    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_vals = pair_dists / 2.0

    wall_vals = CUBE_BOUND - np.abs(c)
    wall_flat = wall_vals.ravel()

    all_vals = np.concatenate([pair_vals, wall_flat])

    if len(all_vals) == 0:
        return 0.0, np.zeros_like(c)

    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / t)
    s = np.sum(exp_vals)
    radius = m - t * np.log(s)

    weights = exp_vals / s

    grad = np.zeros_like(c)

    n_pairs = len(pair_vals)
    pair_w = weights[:n_pairs]

    pi, pj = iu
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)

    diff_pair = c[pi] - c[pj]

    factor = pair_w / (2.0 * safe_dists)

    np.add.at(grad, pi, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad, pj, factor[:, np.newaxis] * diff_pair)

    wall_w = weights[n_pairs:]
    wall_w = wall_w.reshape(N, 3)

    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad += wall_w * signs

    return radius, grad


def _lloyds_algorithm(c_init, deadline, max_iters=100):
    """
    Lloyd's algorithm: move each center toward the centroid of its Voronoi cell
    (clipped to the cube) to equalize clearance.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)

    for it in range(max_iters):
        if time.time() > deadline:
            break

        try:
            vor = Voronoi(c)
        except Exception:
            break

        new_c = c.copy()
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue
            verts = vor.vertices[region]
            if len(verts) == 0:
                continue
            centroid = np.mean(verts, axis=0)
            centroid = np.clip(centroid, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
            new_c[i] = centroid

        new_r = _compute_radius(new_c)
        if new_r > best_r:
            best_r = new_r
            best_c = new_c.copy()

        if np.max(np.abs(new_c - c)) < 1e-10:
            c = new_c
            break
        c = new_c

    return best_c, best_r


def _gradient_polish(c_init, deadline, max_iters=200):
    """
    Gradient-based polishing using soft-min surrogate.
    Multi-scale beta annealing: start with larger temperature, refine.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)

    temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    lr = 0.002

    for t in temperatures:
        if time.time() > deadline:
            break
        for it in range(max_iters // len(temperatures)):
            if time.time() > deadline:
                break

            radius, grad = _softmin_radius_and_grad(c, t)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break

            new_c = c + lr * grad
            new_c = np.clip(new_c, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)

            new_r = _compute_radius(new_c)
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
                c = new_c
            else:
                lr *= 0.7
                if lr < 1e-8:
                    break

    return best_c, best_r


def _repulsion_anneal(c_init, deadline, max_iters=3000):
    """
    Simulated annealing with repulsion to escape local minima.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()

    r_target = best_r + 0.005
    temp = 0.05
    decay = 0.998
    pert_scale = 0.008

    for it in range(max_iters):
        if time.time() > deadline:
            break

        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)

        overlap_pair = np.maximum(0.0, 2.0 * r_target - dists)
        wall_dists = CUBE_BOUND - np.abs(c)
        overlap_wall = np.maximum(0.0, r_target - wall_dists)

        E = np.sum(overlap_pair ** 2) + np.sum(overlap_wall ** 2)

        move = np.random.normal(0.0, pert_scale, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.49, 0.49)

        diff_new = c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]
        dists_new = np.sqrt(np.sum(diff_new * diff_new, axis=-1))
        np.fill_diagonal(dists_new, np.inf)

        overlap_pair_new = np.maximum(0.0, 2.0 * r_target - dists_new)
        wall_dists_new = CUBE_BOUND - np.abs(c_new)
        overlap_wall_new = np.maximum(0.0, r_target - wall_dists_new)

        E_new = np.sum(overlap_pair_new ** 2) + np.sum(overlap_wall_new ** 2)

        dE = E_new - E
        if dE < 0 or np.random.random() < np.exp(-dE / max(temp, 1e-6)):
            c = c_new
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
                pert_scale *= 0.95
            else:
                pert_scale = min(pert_scale * 1.02, 0.02)
        else:
            pert_scale = min(pert_scale * 1.03, 0.02)

        temp *= decay

        if it % 500 == 0 and E < 1e-4:
            r_target += 0.002

    return best_c, best_r


def _cubic_grid_init():
    """Create a cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def _fcc_init():
    """FCC-like lattice initialization for 50 spheres."""
    # FCC structure in a cube: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) type positions
    # Scale to fit in [-0.5, 0.5]^3
    points = []
    # Generate FCC lattice points
    a = 1.0  # lattice constant
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Face-centered positions
                points.append(np.array([i * a, j * a, k * a]) - 1.0)
                points.append(np.array([i * a + 0.5 * a, j * a + 0.5 * a, k * a]) - 1.0)
                points.append(np.array([i * a + 0.5 * a, j * a, k * a + 0.5 * a]) - 1.0)
                points.append(np.array([i * a, j * a + 0.5 * a, k * a + 0.5 * a]) - 1.0)

    points = np.array(points)
    # Keep only points inside the cube
    mask = np.all(np.abs(points) <= 0.5 + 1e-6, axis=1)
    points = points[mask]

    # If we have more than N, take the first N; if fewer, pad with random
    if len(points) >= N:
        return points[:N]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, (remaining, 3))
        return np.vstack([points, random_pts])


def _jittered_grid_init():
    """Grid initialization with small random jitter."""
    base = _cubic_grid_init()
    jitter = np.random.randn(N, 3) * 0.005
    centers = base + jitter
    return np.clip(centers, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)


def _random_init():
    """Random initialization within the cube."""
    return np.random.uniform(-CUBE_BOUND + 0.02, CUBE_BOUND - 0.02, (N, 3))


def construct():
    """Construct the optimal packing of 50 spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + DEADLINE - MARGIN

    best_centers = None
    best_score = 0.0

    def try_update(centers):
        nonlocal best_centers, best_score
        s = _compute_radius(centers)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()

    # Stage 1: Multiple initializations
    inits = [
        _cubic_grid_init(),
        _fcc_init(),
        _jittered_grid_init(),
        _random_init(),
        _random_init(),
    ]

    for init in inits:
        if time.time() > deadline - 60:
            break

        # Lloyd's algorithm
        lloyd_c, lloyd_r = _lloyds_algorithm(init.copy(), deadline - 30)
        try_update(lloyd_c)

        # Gradient polish
        if time.time() < deadline - 20:
            grad_c, grad_r = _gradient_polish(lloyd_c.copy(), deadline - 10)
            try_update(grad_c)

    # Stage 2: Repulsion annealing from best so far
    if time.time() < deadline - 20 and best_centers is not None:
        anneal_c, anneal_r = _repulsion_anneal(best_centers.copy(), deadline - 10)
        try_update(anneal_c)

        # Polish after annealing
        if time.time() < deadline - 5:
            final_c, final_r = _gradient_polish(anneal_c.copy(), deadline)
            try_update(final_c)

    # Final high-precision polish
    if time.time() < deadline and best_centers is not None:
        # Very tight gradient polish with small temperature
        c = best_centers.copy()
        for t in [0.0001, 0.00005, 0.00002]:
            if time.time() > deadline:
                break
            lr = 0.001
            for _ in range(100):
                if time.time() > deadline:
                    break
                radius, grad = _softmin_radius_and_grad(c, t)
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-14:
                    break
                new_c = c + lr * grad
                new_c = np.clip(new_c, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
                new_r = _compute_radius(new_c)
                if new_r > _compute_radius(c):
                    c = new_c
                else:
                    lr *= 0.8
                    if lr < 1e-10:
                        break
        try_update(c)

    if best_centers is None:
        best_centers = _cubic_grid_init()

    best_centers = np.clip(best_centers, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
    return best_centers

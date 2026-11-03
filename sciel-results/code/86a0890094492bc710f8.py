"""Rewritten program: BCC lattice starts + multi-restart local optimization with deadline guard."""
import numpy as np
from scipy.spatial.distance import cdist
import time

N = 50
CUBE_HALF = 0.5
DEADLINE_FRAC = 0.90  # finish by 90% of wall-clock to be safe


def _compute_radius(centres):
    """Return the largest common radius for given centres."""
    # Wall distance
    wall = np.min(np.abs(centres), axis=1)
    r_wall = CUBE_HALF - np.max(wall, axis=0)  # wait, need per-sphere min over axes
    # Actually: for each sphere i, distance to wall = min over k of (CUBE_HALF - |c_ik|)
    dist_to_wall = CUBE_HALF - np.abs(centres)  # shape (N,3)
    r_wall_per_sphere = np.min(dist_to_wall, axis=1)  # shape (N,)
    r_wall = np.min(r_wall_per_sphere)

    # Pairwise distances
    d = cdist(centres, centres)
    np.fill_diagonal(d, np.inf)
    r_pair = np.min(d) / 2.0

    return min(r_wall, r_pair)


def _bcc_lattice(n_spheres):
    """Generate a BCC-like lattice configuration for n_spheres spheres in unit cube."""
    # BCC: simple cubic lattice + body centers
    # For 50 spheres, use a 4x4x4 grid = 64 positions, pick 50
    m = 4  # 4x4x4 = 64 cells
    positions = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Simple cubic position
                x = (ix + 0.5) / m - 0.5
                y = (iy + 0.5) / m - 0.5
                z = (iz + 0.5) / m - 0.5
                positions.append([x, y, z])
                # Body center of this cell
                positions.append([x, y, z])  # same for now; we'll offset later
    # That's 128 positions; instead let's do proper BCC
    # BCC unit cell has 2 atoms: corners + body center
    # In a supercell of m x m x m BCC cells, we have 2*m^3 positions
    # For m=3: 54 positions, pick 50
    m = 3
    positions = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner-like position (centered in subcell)
                x0 = (ix + 0.5) / m - 0.5
                y0 = (iy + 0.5) / m - 0.5
                z0 = (iz + 0.5) / m - 0.5
                positions.append([x0, y0, z0])
                # Body center offset
                dx = 1.0 / (2 * m)
                positions.append([x0 + dx, y0 + dx, z0 + dx])
    positions = np.array(positions[:n_spheres])
    return positions


def _random_perturbation(base, sigma=0.02, rng=None):
    """Add Gaussian perturbation and clip to cube."""
    if rng is None:
        rng = np.random.default_rng()
    perturbed = base + rng.normal(0, sigma, size=base.shape)
    perturbed = np.clip(perturbed, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
    return perturbed


def _local_optimize(centres, max_iter=2000, lr_init=0.01, deadline=None):
    """
    Local optimization to maximize the minimum radius.
    Uses gradient-free approach: move points apart when they are close.
    Objective: maximize r = min(pairwise_dist/2, wall_dist)
    We use a repulsive force model.
    """
    centres = centres.copy()
    n = len(centres)
    best_r = _compute_radius(centres)
    best_centres = centres.copy()

    lr = lr_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Compute pairwise distances
        d = cdist(centres, centres)
        np.fill_diagonal(d, np.inf)

        # Find the closest pair
        min_d = np.min(d)
        if min_d < 1e-15:
            # Coincident points; add small random kick
            i, j = np.unravel_index(np.argmin(d), d.shape)
            centres[i] += np.random.normal(0, 0.01, 3)
            centres[j] -= np.random.normal(0, 0.01, 3)
            continue

        # Wall distances
        dist_to_wall = CUBE_HALF - np.abs(centres)
        r_wall_per_sphere = np.min(dist_to_wall, axis=1)
        r_wall = np.min(r_wall_per_sphere)

        r_pair = min_d / 2.0
        r = min(r_wall, r_pair)

        if r > best_r + 1e-12:
            best_r = r
            best_centres = centres.copy()

        # Compute forces
        forces = np.zeros_like(centres)

        # Pairwise repulsion: for pairs closer than some threshold, push apart
        threshold = 0.3  # only consider close pairs for efficiency
        for i in range(n):
            for j in range(i + 1, n):
                dij = d[i, j]
                if dij < threshold and dij > 1e-12:
                    # Force magnitude inversely proportional to distance
                    f = (threshold - dij) / (dij * dij) * 0.01
                    diff = centres[i] - centres[j]
                    forces[i] += f * diff / dij
                    forces[j] -= f * diff / dij

        # Wall repulsion: push away from walls
        for i in range(n):
            for k in range(3):
                if centres[i, k] > 0:
                    # Near +x or +y or +z wall
                    dist = CUBE_HALF - centres[i, k]
                    if dist < threshold:
                        f = (threshold - dist) / max(dist, 1e-6) * 0.01
                        forces[i, k] -= f
                else:
                    dist = CUBE_HALF + centres[i, k]
                    if dist < threshold:
                        f = (threshold - dist) / max(dist, 1e-6) * 0.01
                        forces[i, k] += f

        # Update positions
        centres += lr * forces
        centres = np.clip(centres, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)

        # Anneal learning rate
        lr = max(lr_init * (0.995 ** it), 1e-5)

    return best_centres, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRAC

    rng = np.random.default_rng(42)

    # Generate multiple initial configurations
    configs = []

    # BCC lattice
    bcc = _bcc_lattice(N)
    configs.append(bcc)

    # Perturbed BCC variants
    for sigma in [0.01, 0.02, 0.03, 0.05]:
        perturbed = _random_perturbation(bcc, sigma=sigma, rng=rng)
        configs.append(perturbed)

    # Simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    configs.append(grid[:N])

    # Random uniform
    random_cfg = rng.uniform(-CUBE_HALF + 0.02, CUBE_HALF - 0.02, size=(N, 3))
    configs.append(random_cfg)

    best_centres = configs[0]
    best_r = _compute_radius(best_centres)

    for idx, cfg in enumerate(configs):
        if time.time() > deadline - 5:
            break

        # Local optimization
        opt_centres, opt_r = _local_optimize(
            cfg,
            max_iter=1500,
            lr_init=0.005,
            deadline=deadline
        )

        if opt_r > best_r:
            best_r = opt_r
            best_centres = opt_centres

    # Final polish: one more pass with tighter settings
    if time.time() < deadline - 10:
        polished, polished_r = _local_optimize(
            best_centres,
            max_iter=500,
            lr_init=0.001,
            deadline=deadline
        )
        if polished_r > best_r:
            best_r = polished_r
            best_centres = polished

    return best_centres

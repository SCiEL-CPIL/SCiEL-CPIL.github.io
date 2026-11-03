"""Combined program: Hybrid lattice initialization + hard-violation energy with adaptive radius search."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100
HALF = 0.5
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 12.0


def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)


def _compute_radius(C):
    """Compute the maximum possible radius for a set of centers C."""
    if len(C) < 2:
        return HALF - np.max(np.abs(C))
    dist_to_bound = HALF - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    tree = cKDTree(C)
    distances, _ = tree.query(C, k=2)
    min_dist = np.min(distances[:, 1])
    r_pair = min_dist / 2.0
    return min(r_bound, r_pair)


def _energy_and_gradient(r, flat_pos, beta=50.0):
    """
    Compute energy and gradient for fixed radius r using hard squared-violation.
    Energy is sum of squared violations for overlaps and boundary.
    """
    pos = flat_pos.reshape(N, 3)

    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]

    # Boundary distances
    d_walls = HALF - np.abs(pos)
    wall_dists = d_walls.flatten()

    # Penalties: hard squared violation
    # Overlap: violation = max(0, 2r - d)
    # Boundary: violation = max(0, -d_wall)

    overlap_viol = np.maximum(0, 2.0 * r - pair_dists)
    wall_viol = np.maximum(0, -wall_dists)

    energy = np.sum(overlap_viol ** 2) + np.sum(wall_viol ** 2)

    # Gradients
    grad_pos = np.zeros_like(pos)

    # Pairwise gradient
    # d(overlap_viol^2)/dpos_i = 2 * overlap_viol * d(2r-d)/dpos_i = 2*overlap_viol * (pos_i - pos_j)/d
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = pair_dists[idx]
        if d < 1e-12:
            # Coincident: push apart
            grad_pos[i] += np.array([1.0, 0.5, 0.2])
            grad_pos[j] -= np.array([1.0, 0.5, 0.2])
            continue
        coeff = 2.0 * overlap_viol[idx] / d
        vec = pos[i] - pos[j]
        grad_pos[i] += coeff * vec
        grad_pos[j] -= coeff * vec

    # Boundary gradient
    for idx in range(N * 3):
        i = idx // 3
        k = idx % 3
        if wall_viol[idx] > 0:
            s = np.sign(pos[i, k])
            if s == 0:
                s = 1.0
            grad_pos[i, k] += 2.0 * wall_viol[idx] * s

    return energy, grad_pos.flatten()


def _minimize_energy(r, pos, max_iter=100, ftol=1e-12, gtol=1e-10):
    """Minimize energy for fixed radius r."""
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)
    try:
        res = minimize(
            lambda x: _energy_and_gradient(r, x)[0],
            pos.flatten(),
            jac=lambda x: _energy_and_gradient(r, x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if np.isfinite(res.fun):
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos


def _generate_hybrid_initial_points(seed):
    """
    Generate hybrid initialization: 64 spheres in 4x4x4 simple cubic lattice
    and 36 spheres in centered cubic arrangement within interstitial voids.
    """
    rng = np.random.default_rng(seed)

    # 4x4x4 simple cubic lattice = 64 points
    # Place them in the cube [-0.5, 0.5]^3 with some margin
    margin = 0.05
    size = 4
    spacing = (2 * (HALF - margin)) / (size - 1)
    coords = np.linspace(-HALF + margin, HALF - margin, size)
    lattice_points = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                lattice_points.append([coords[i], coords[j], coords[k]])
    lattice_points = np.array(lattice_points)

    # 36 interstitial points: centered cubic arrangement
    # In a simple cubic lattice, there are body-centered positions
    # For a 4x4x4 grid, we can add points at cell centers
    # But we only need 36 more. Let's use the body-centered positions of a 3x3x4 grid
    # or place them at the interstitial sites

    # Alternative: place 36 points at the centers of the 3x3x3 interior cells
    # of the 4x4x4 lattice, plus some additional
    interior_points = []
    for i in range(size - 1):
        for j in range(size - 1):
            for k in range(size - 1):
                # Cell center
                cx = (coords[i] + coords[i + 1]) / 2
                cy = (coords[j] + coords[j + 1]) / 2
                cz = (coords[k] + coords[k + 1]) / 2
                interior_points.append([cx, cy, cz])
    interior_points = np.array(interior_points)

    # We have 27 interior points, need 36 total for the second group
    # Add 9 more at face centers or edge midpoints
    extra_points = []
    # Add points at the center of each face of the sub-cubes
    for i in range(size - 1):
        for j in range(size - 1):
            cx = (coords[i] + coords[i + 1]) / 2
            cy = (coords[j] + coords[j + 1]) / 2
            # Top and bottom faces
            extra_points.append([cx, cy, HALF - margin])
            extra_points.append([cx, cy, -HALF + margin])

    extra_points = np.array(extra_points[:9])

    # Combine: 64 lattice + 27 interior + 9 extra = 100
    all_points = np.vstack([lattice_points, interior_points, extra_points])

    # Clip to valid range
    all_points = np.clip(all_points, -HALF + 1e-9, HALF - 1e-9)

    # Shuffle and take first N
    idx = rng.permutation(len(all_points))
    return all_points[idx[:N]].copy()


def _generate_bcc_initial_points(seed):
    """Generate BCC-like initial points."""
    rng = np.random.default_rng(seed)
    size = 4
    raw = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                raw.append([i, j, k])
                raw.append([i + 0.5, j + 0.5, k + 0.5])
    raw = np.array(raw)
    center = np.array([size / 2 - 0.25, size / 2 - 0.25, size / 2 - 0.25])
    raw = raw - center
    max_coord = np.max(np.abs(raw))
    scale = 0.45 / max_coord
    raw *= scale
    idx = rng.permutation(len(raw))
    return raw[idx[:N]].copy()


def _generate_fcc_initial_points(seed):
    """Generate FCC-like initial points."""
    rng = np.random.default_rng(seed)
    size = 4
    raw = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                raw.append([i, j, k])
                raw.append([i + 0.5, j + 0.5, k])
                raw.append([i + 0.5, j, k + 0.5])
                raw.append([i, j + 0.5, k + 0.5])
    raw = np.array(raw)
    center = np.array([size / 2 - 0.25, size / 2 - 0.25, size / 2 - 0.25])
    raw = raw - center
    max_coord = np.max(np.abs(raw))
    scale = 0.45 / max_coord
    raw *= scale
    idx = rng.permutation(len(raw))
    return raw[idx[:N]].copy()


def _generate_random_initial_points(seed):
    """Generate random initial points."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, (N, 3))


def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)

    best_radius = 0.0
    best_centers = None

    # Start with a reasonable radius estimate
    r_current = 0.10

    # Try multiple initialization strategies
    generators = [
        _generate_hybrid_initial_points,
        _generate_bcc_initial_points,
        _generate_fcc_initial_points,
        _generate_random_initial_points,
    ]
    seeds = [42, 123, 456, 789, 101112, 202021, 303031, 404041, 505051, 606061]

    for gen_idx, gen_func in enumerate(generators):
        for seed in seeds:
            if _check_time():
                break

            centers = gen_func(seed)
            centers = np.clip(centers, -HALF + 1e-9, HALF - 1e-9)

            # Iterative Radius Increase
            r_try = r_current
            improved = False
            max_iter_r = 25

            for _ in range(max_iter_r):
                if _check_time():
                    break

                # Minimize energy for current r_try
                optimized_centers = _minimize_energy(r_try, centers, max_iter=100)

                # Check if feasible
                actual_radius = _compute_radius(optimized_centers)

                if actual_radius >= r_try - 1e-6:
                    # Feasible! Increase radius
                    r_try += 0.002
                    centers = optimized_centers
                    improved = True
                else:
                    # Not feasible, try a smaller step
                    r_try = r_try - 0.001
                    if r_try < r_current:
                        break

            # Final polish
            final_radius = _compute_radius(centers)

            if final_radius > best_radius:
                best_radius = final_radius
                best_centers = centers.copy()
                r_current = max(r_current, final_radius)

    # Final refinement with the best configuration
    if best_centers is not None and not _check_time():
        r_polish = best_radius
        for _ in range(15):
            if _check_time():
                break
            r_polish += 0.0005
            polished = _minimize_energy(r_polish, best_centers, max_iter=150)
            actual_r = _compute_radius(polished)
            if actual_r >= r_polish - 1e-6:
                best_radius = actual_r
                best_centers = polished
            else:
                break

    return best_centers

"""Combined program: hybrid lattice init + soft-min gradient optimizer + energy-based refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5


def _compute_radius(centers):
    """Compute the exact maximum common radius for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Wall constraints
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    return min(min_pair_dist / 2.0, min_wall_dist)


def _soft_min_obj_grad(flat_pos, beta=100.0):
    """
    Soft-min objective: maximize the soft-minimum of all effective distances.
    soft_min = -logsumexp(-beta * all_eff_dists) / beta
    We minimize -soft_min.
    """
    pos = flat_pos.reshape(N, 3)

    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)

    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]

    # Wall distances
    wall_dists = (HALF - np.abs(pos)).flatten()

    # All effective distances
    all_eff_dists = np.concatenate([pair_dists / 2.0, wall_dists])

    # Soft-min computation
    scaled = -beta * all_eff_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta

    # Gradient
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)

    grad_pos = np.zeros_like(pos)

    # Pair gradient
    n_pairs = len(iu[0])
    pair_weights = weights[:n_pairs]

    i_idx = iu[0]
    j_idx = iu[1]
    d_vals = pair_dists

    valid = d_vals > 1e-14
    if np.any(valid):
        i_v = i_idx[valid]
        j_v = j_idx[valid]
        d_v = d_vals[valid]
        w_v = pair_weights[valid]

        vecs = pos[i_v] - pos[j_v]
        coeffs = w_v / (2.0 * d_v)

        np.add.at(grad_pos, i_v, coeffs[:, np.newaxis] * vecs)
        np.add.at(grad_pos, j_v, -coeffs[:, np.newaxis] * vecs)

    # Wall gradient
    wall_weights = weights[n_pairs:]
    signs = np.sign(pos)
    signs[signs == 0] = 1.0
    grad_wall = -wall_weights.reshape(N, 3) * signs
    grad_pos += grad_wall

    return -soft_min_val, -grad_pos.flatten()


def _energy_and_gradient(x, R):
    """
    Energy: sum of squared violations of distance constraints for target radius R.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)

    energy = 0.0
    grad = np.zeros_like(centers)

    # Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)

    mask = dists < 2.0 * R
    if np.any(mask):
        i_idx, j_idx = np.triu_indices(n, k=1)
        d_ij = dists[i_idx, j_idx]
        mask_pairs = d_ij < 2.0 * R

        if np.any(mask_pairs):
            v_ij = 2.0 * R - d_ij[mask_pairs]
            d_valid = d_ij[mask_pairs]
            i_valid = i_idx[mask_pairs]
            j_valid = j_idx[mask_pairs]

            energy += np.sum(v_ij**2)

            factors = 2.0 * v_ij / d_valid
            vecs = centers[i_valid] - centers[j_valid]
            forces = factors[:, np.newaxis] * vecs

            np.add.at(grad, i_valid, -forces)
            np.add.at(grad, j_valid, forces)

    # Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs

    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]

        energy += np.sum(violations_wall**2)

        max_k = np.argmax(abs_centers[i_valid_wall], axis=1)
        signs = np.sign(centers[i_valid_wall, max_k])
        signs[signs == 0] = 1.0
        grad_vals = 2.0 * violations_wall * signs

        np.add.at(grad, (i_valid_wall, max_k), grad_vals)

    return energy, grad.flatten()


def _minimize_soft_min(init_centers, beta, max_iter=300, deadline=None):
    """Minimize negative soft-min using L-BFGS-B."""
    def objective_and_grad(x):
        val, g = _soft_min_obj_grad(x, beta)
        return val, g

    bounds = [(-0.4999, 0.4999)] * (N * 3)

    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False, 'maxls': 50}
        )
        new_centers = res.x.reshape(N, 3)
    except Exception:
        new_centers = init_centers

    if deadline is not None and time.time() > deadline:
        return init_centers, _compute_radius(init_centers)

    r = _compute_radius(new_centers)
    return new_centers, r


def _minimize_energy(init_centers, target_R, max_iter=500, deadline=None):
    """Minimize energy for a given target radius."""
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.4999, 0.4999)] * (N * 3)

    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
    except Exception:
        new_centers = init_centers

    if deadline is not None and time.time() > deadline:
        return init_centers, _compute_radius(init_centers)

    r = _compute_radius(new_centers)
    return new_centers, r


def _hybrid_lattice_init(n):
    """
    Hybrid initialization: 64 spheres in a 4x4x4 simple cubic lattice,
    36 spheres in body-centered cubic interstitial positions.
    """
    positions = []

    # 4x4x4 simple cubic lattice (64 points)
    m = 4
    spacing = 1.0 / m
    offset = 0.5 * spacing
    for i in range(m):
        for j in range(m):
            for k in range(m):
                positions.append([offset + i * spacing - 0.5,
                                  offset + j * spacing - 0.5,
                                  offset + k * spacing - 0.5])
    positions = np.array(positions)

    if n <= len(positions):
        return positions[:n]

    # Add body-centered cubic interstitial positions
    # In a 4x4x4 grid, the BCC interstitials are at the centers of each cube cell
    # Each cell center is at (i+0.5)*spacing + offset - 0.5 = (i+1)*spacing - 0.5
    # Actually, the BCC centers in a cubic lattice at offset+i*spacing are at
    # offset + (i+0.5)*spacing - 0.5 = (i+1)*spacing - 0.5
    # But we need to be careful. The cell centers between lattice points.
    # For a 4x4x4 grid with points at 0.125, 0.375, 0.625, 0.875 (offset=0.125, spacing=0.25)
    # Cell centers are at 0.25, 0.5, 0.75 in each dimension
    # That's 3x3x3 = 27 points, not enough. Let's also add face centers.

    # Actually, let's use a different approach: place additional points at
    # the centers of the 3x3x3 sub-cubes formed by the 4x4x4 grid, plus
    # face-centered positions.

    # Sub-cube centers: 3x3x3 = 27 points
    for i in range(3):
        for j in range(3):
            for k in range(3):
                cx = offset + (i + 0.5) * spacing - 0.5
                cy = offset + (j + 0.5) * spacing - 0.5
                cz = offset + (k + 0.5) * spacing - 0.5
                positions.append([cx, cy, cz])

    # Face centers of the 4x4x4 grid: each face has 3x3=9 centers, 6 faces = 54
    # But many overlap with sub-cube centers. Let's just add edge centers instead.
    # Edge centers: 12 edges * 3 segments = 36, but let's just add what we need.

    positions = np.array(positions)

    if n <= len(positions):
        return positions[:n]

    # If still not enough, add random points
    rng = np.random.RandomState(42)
    extra = rng.uniform(-0.45, 0.45, size=(n - len(positions), 3))
    positions = np.vstack([positions, extra])

    return positions[:n]


def _fcc_init(n):
    """Generate initial positions using FCC lattice structure."""
    positions = []
    # FCC unit cell: points at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Scale to fit in cube
    # Use a 3x3x3 grid of FCC cells
    cell_size = 1.0 / 3.0
    fcc_offsets = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for ox, oy, oz in fcc_offsets:
                    x = (i + ox) * cell_size - 0.5
                    y = (j + oy) * cell_size - 0.5
                    z = (k + oz) * cell_size - 0.5
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        positions.append([x, y, z])

    positions = np.array(positions)
    if n <= len(positions):
        return positions[:n]

    rng = np.random.RandomState(42)
    extra = rng.uniform(-0.45, 0.45, size=(n - len(positions), 3))
    positions = np.vstack([positions, extra])
    return positions[:n]


def _random_init(n, seed=None):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))


def _optimize_from_start(centers, deadline):
    """
    Run the full optimization pipeline from a given starting configuration.
    Uses soft-min for initial optimization, then energy-based refinement.
    """
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_r

    # Phase 1: Soft-min optimization with increasing beta
    beta = 50.0
    for _ in range(3):
        if time.time() > deadline - 10.0:
            break
        centers, r = _minimize_soft_min(best_centers, beta, max_iter=200, deadline=deadline - 5.0)
        if r > best_radius:
            best_radius = r
            best_centers = centers
        beta *= 1.5
        beta = min(beta, 300.0)

    # Phase 2: Energy-based refinement with increasing target radius
    target_r = best_radius * 1.001
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        centers, r = _minimize_energy(best_centers, target_r, max_iter=300, deadline=deadline - 3.0)
        if r > best_radius:
            best_radius = r
            best_centers = centers
            target_r = best_radius * 1.001
        else:
            # Try a small perturbation and retry
            rng = np.random.RandomState(int(time.time() * 1000) % 100000)
            perturbed = best_centers + rng.randn(N, 3) * 0.003
            perturbed = np.clip(perturbed, -0.4999, 0.4999)
            centers_p, r_p = _minimize_energy(perturbed, target_r, max_iter=200, deadline=deadline - 3.0)
            if r_p > best_radius:
                best_radius = r_p
                best_centers = centers_p
                target_r = best_radius * 1.001
            else:
                break

    # Phase 3: Final soft-min polish
    if time.time() < deadline - 2.0:
        centers, r = _minimize_soft_min(best_centers, 200.0, max_iter=300, deadline=deadline - 1.0)
        if r > best_radius:
            best_radius = r
            best_centers = centers

    return best_centers, best_radius


def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 110.0

    # Try multiple initial configurations
    inits = [
        _hybrid_lattice_init(N),
        _fcc_init(N),
        _random_init(N, 0),
        _random_init(N, 1),
    ]

    best_centers = inits[0]
    best_radius = _compute_radius(best_centers)

    for init in inits[1:]:
        if time.time() > deadline - 30.0:
            break
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init

    # Optimize from the best initial configuration
    best_centers, best_radius = _optimize_from_start(best_centers, deadline)

    # If time remains, try optimizing from a different start
    if time.time() < deadline - 20.0:
        for init in inits:
            if time.time() > deadline - 10.0:
                break
            c, r = _optimize_from_start(init, deadline - 5.0)
            if r > best_radius:
                best_radius = r
                best_centers = c

    # Ensure validity
    best_centers = np.clip(best_centers, -0.499999, 0.499999)

    return best_centers

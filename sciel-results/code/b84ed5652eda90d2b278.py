"""
Sphere packing in unit cube: Riesz s-energy for global landscape +
violation-energy with iterative radius increase + contact-graph refinement.

Strategy:
1. Use Riesz s-energy (multiple s values) to find good global configurations.
2. Switch to violation-energy optimization with an increasing target radius
   to push the minimum distance beyond the Riesz plateau.
3. Use contact-graph refinement to focus on the critical constraints.
4. Multi-start with diverse initializations.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair_dist = np.min(dists[:, 1])
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)


def _riesz_energy_and_grad(x, s=8.0):
    """
    Riesz s-energy: sum_{i<j} 1/d_ij^s.
    Minimizing this pushes points apart uniformly.
    Returns (energy, gradient_flat).
    """
    c = x.reshape(N, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists2, 1.0)
    dists = np.sqrt(dists2)

    iu = np.triu_indices(N, k=1)
    d_pair = dists[iu]
    d_pair = np.maximum(d_pair, 1e-15)

    energy = np.sum(d_pair ** (-s))

    # Gradient: grad[i] = -s * sum_{j != i} (x_i - x_j) / |x_i - x_j|^(s+2)
    inv_d_pow = dists ** (-(s + 2))
    np.fill_diagonal(inv_d_pow, 0.0)
    weighted_diff = diff * inv_d_pow[:, :, np.newaxis]
    grad = -s * np.sum(weighted_diff, axis=1)

    return energy, grad.flatten()


def _violation_energy_and_grad(x, target_r):
    """
    Energy for a fixed target radius: sum of squared violations of
    pairwise separation (2*r) and wall distance (r).
    Returns (energy, gradient_flat).
    """
    c = x.reshape(N, 3)

    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)

    active = dists < 2.0 * target_r
    active[np.diag_indices_from(active)] = False

    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    energy_pair = 0.5 * np.sum(violations ** 2)

    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]

    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)

    wall_dists = HALF - np.abs(c)
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    energy_wall = np.sum(wall_violations ** 2)

    signs = np.sign(c)
    grad_wall = 2.0 * wall_violations * signs

    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()

    return total_energy, total_grad


def _riesz_flat(x_flat, s):
    """Flat wrapper for scipy."""
    return _riesz_energy_and_grad(x_flat, s=s)


def _violation_flat(x_flat, target_r, deadline):
    """Flat wrapper for scipy with timeout guard."""
    if time.time() > deadline:
        raise TimeoutError()
    return _violation_energy_and_grad(x_flat, target_r)


def _local_optimize_violation(init_pts, target_r, deadline, max_iter=3000):
    """
    L-BFGS-B minimization of the violation energy for a fixed target radius.
    """
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)

    try:
        result = minimize(
            _violation_flat,
            init_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            args=(target_r, deadline),
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 8000}
        )
        opt_pts = result.x.reshape(N, 3)
        if np.all(np.isfinite(opt_pts)):
            return opt_pts
        return init_pts
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts


def _local_optimize_riesz(init_pts, s, deadline, max_iter=3000):
    """
    L-BFGS-B minimization of Riesz s-energy.
    """
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)

    try:
        result = minimize(
            _riesz_flat,
            init_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            args=(s,),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 6000}
        )
        opt_pts = result.x.reshape(N, 3)
        if np.all(np.isfinite(opt_pts)):
            return opt_pts
        return init_pts
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts


def _contact_graph_refine(centers, deadline):
    """
    Identify near-contact pairs and refine only the involved spheres
    to maximize the minimum distance among those pairs (and wall).
    Uses a penalty-based approach on the contact subgraph.
    """
    if time.time() > deadline - 3.0:
        return centers

    centers = centers.copy()
    r = _compute_radius(centers)
    if r < 1e-6:
        return centers

    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.05 * r)

    if len(pairs) == 0:
        return centers

    involved = set()
    for i, j in pairs:
        involved.add(i)
        involved.add(j)
    involved = sorted(involved)

    if len(involved) < 2:
        return centers

    idx_array = np.array(involved)
    n_involved = len(involved)

    def sub_objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(n_involved, 3)

        # Penalty: sum of squared violations for contact pairs
        penalty = 0.0
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < 2.0 * r:
                penalty += (2.0 * r - d) ** 2

        # Wall penalty for involved points
        wall_dists = HALF - np.max(np.abs(c[idx_array]), axis=1)
        wall_viol = np.maximum(0.0, r - wall_dists)
        penalty += np.sum(wall_viol ** 2)

        return penalty

    def sub_grad(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(n_involved, 3)

        grad_full = np.zeros((N, 3))

        for i, j in pairs:
            d_vec = c[i] - c[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-12:
                continue
            if d < 2.0 * r:
                unit = d_vec / d
                grad_full[i] += -2.0 * (2.0 * r - d) * unit
                grad_full[j] += 2.0 * (2.0 * r - d) * unit

        wall_dists = HALF - np.abs(c[idx_array])
        wall_viol = np.maximum(0.0, r - wall_dists)
        signs = np.sign(c[idx_array])
        grad_wall = 2.0 * wall_viol * signs
        grad_full[idx_array] += grad_wall

        return grad_full[idx_array].flatten()

    x0 = centers[idx_array].flatten()
    bounds_sub = [(-HALF + 1e-9, HALF - 1e-9)] * (n_involved * 3)

    try:
        result = minimize(
            sub_objective,
            x0,
            jac=sub_grad,
            method='L-BFGS-B',
            bounds=bounds_sub,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_sub = result.x.reshape(n_involved, 3)
        centers[idx_array] = new_sub
    except TimeoutError:
        pass
    except Exception:
        pass

    centers = np.clip(centers, -HALF, HALF)
    return centers


def _generate_grid():
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _generate_bcc():
    """Body-centered cubic: 4x4x4 grid + shifted centers = 128 points, take 100."""
    points = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                x = (ix + 0.5) / 4.0 - 0.5
                y = (iy + 0.5) / 4.0 - 0.5
                z = (iz + 0.5) / 4.0 - 0.5
                points.append([x, y, z])
                # BCC center
                xc = (ix + 1.0) / 4.0 - 0.5
                yc = (iy + 1.0) / 4.0 - 0.5
                zc = (iz + 1.0) / 4.0 - 0.5
                points.append([xc, yc, zc])
    return np.array(points[:N])


def _generate_fcc():
    """Face-centered cubic arrangement."""
    points = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                x = (ix + 0.5) / 4.0 - 0.5
                y = (iy + 0.5) / 4.0 - 0.5
                z = (iz + 0.5) / 4.0 - 0.5
                points.append([x, y, z])
                # FCC face centers
                if ix < 3:
                    points.append([(ix + 1.0) / 4.0 - 0.5, y, z])
                if iy < 3:
                    points.append([x, (iy + 1.0) / 4.0 - 0.5, z])
                if iz < 3:
                    points.append([x, y, (iz + 1.0) / 4.0 - 0.5])
    # Deduplicate and take N
    pts = np.array(points)
    # Simple dedup
    unique = []
    for p in pts:
        if not any(np.allclose(p, u, atol=1e-6) for u in unique):
            unique.append(p)
    unique = np.array(unique[:N])
    if len(unique) < N:
        # Pad with random
        rng = np.random.default_rng(0)
        pad = rng.uniform(-0.4, 0.4, size=(N - len(unique), 3))
        unique = np.vstack([unique, pad])
    return unique


def _iterative_radius_increase(centers, deadline, max_rounds=15):
    """
    Iteratively increase target radius and re-optimize with violation energy.
    This is the key structural mechanism to push past the Riesz plateau.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)

    for round_i in range(max_rounds):
        if time.time() > deadline - 5.0:
            break

        # Try to increase radius by a small step
        delta_r = 0.0005 + 0.0002 * round_i  # Decreasing steps
        target_r = current_r + delta_r

        # Add tiny noise to break symmetry
        rng = np.random.default_rng(42 + round_i)
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(centers + noise, -0.495, 0.495)

        opt_pts = _local_optimize_violation(init_pts, target_r, deadline, max_iter=2000)

        new_r = _compute_radius(opt_pts)

        if new_r > current_r + 1e-8:
            centers = opt_pts
            current_r = new_r
        else:
            # Try a smaller step
            target_r = current_r + delta_r * 0.5
            opt_pts = _local_optimize_violation(init_pts, target_r, deadline, max_iter=1500)
            new_r = _compute_radius(opt_pts)
            if new_r > current_r + 1e-8:
                centers = opt_pts
                current_r = new_r
            else:
                # Stagnation: try contact-graph refinement
                if time.time() > deadline - 10.0:
                    break
                centers = _contact_graph_refine(centers, deadline)
                new_r = _compute_radius(centers)
                if new_r > current_r + 1e-8:
                    current_r = new_r
                else:
                    break

    return centers, current_r


def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin before 150s limit

    best_centers = None
    best_radius = 0.0
    rng = np.random.default_rng(12345)

    # Phase 1: Riesz s-energy optimization from multiple starts
    # This finds good global configurations
    initializations = []

    # Grid
    grid_c = _generate_grid()
    initializations.append(('grid', grid_c))

    # BCC
    bcc_c = _generate_bcc()
    initializations.append(('bcc', bcc_c))

    # FCC
    fcc_c = _generate_fcc()
    initializations.append(('fcc', fcc_c))

    # Perturbed grids
    for i in range(4):
        base = _generate_grid()
        noise_scale = 0.02 + 0.01 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append((f'perturb{i}', perturbed))

    # Random
    for i in range(3):
        random_c = rng.uniform(-0.45, 0.45, size=(N, 3))
        initializations.append((f'rand{i}', random_c))

    # Riesz s values to try
    s_values = [6.0, 8.0, 10.0, 12.0]

    for name, init_pts in initializations:
        if time.time() > deadline - 40.0:
            break

        for s in s_values:
            if time.time() > deadline - 30.0:
                break

            # Add small noise to break symmetry
            noise = rng.normal(0, 0.002, size=(N, 3))
            init_noisy = np.clip(init_pts + noise, -0.49, 0.49)

            opt_pts = _local_optimize_riesz(init_noisy, s, deadline, max_iter=2000)

            r = _compute_radius(opt_pts)
            if r > best_radius:
                best_radius = r
                best_centers = opt_pts.copy()

    # Phase 2: Iterative radius increase with violation energy
    # This pushes past the Riesz plateau
    if best_centers is not None and time.time() < deadline - 25.0:
        refined_centers, refined_r = _iterative_radius_increase(
            best_centers, deadline, max_rounds=12
        )
        if refined_r > best_radius:
            best_radius = refined_r
            best_centers = refined_centers

    # Phase 3: Try iterative radius increase from other good starts
    # Re-run Riesz from the best config with different s to find alternative basins
    if time.time() < deadline - 20.0:
        for s in [10.0, 14.0, 16.0]:
            if time.time() > deadline - 10.0:
                break
            noise = rng.normal(0, 0.003, size=(N, 3))
            init_noisy = np.clip(best_centers + noise, -0.49, 0.49)
            opt_pts = _local_optimize_riesz(init_noisy, s, deadline, max_iter=1500)
            r = _compute_radius(opt_pts)
            if r > best_radius:
                best_radius = r
                best_centers = opt_pts.copy()

            # Try radius increase from this
            if time.time() < deadline - 10.0:
                refined_centers, refined_r = _iterative_radius_increase(
                    best_centers, deadline, max_rounds=8
                )
                if refined_r > best_radius:
                    best_radius = refined_r
                    best_centers = refined_centers

    # Phase 4: Final contact-graph refinement
    if time.time() < deadline - 5.0:
        best_centers = _contact_graph_refine(best_centers, deadline)
        final_r = _compute_radius(best_centers)
        if final_r > best_radius:
            best_radius = final_r

    # Ensure validity
    best_centers = np.clip(best_centers, -HALF + 1e-10, HALF - 1e-10)
    if not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        best_centers = _generate_grid()

    return best_centers


# Keep the original function names as required
def _pairwise_distances_flat(x):
    """x is shape (N, 3). Returns pairwise distances."""
    n = len(x)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(n, k=1)
    return dists[iu], iu


def _energy_and_grad(x, s=8.0, boundary_softness=100.0):
    """Backward-compatible wrapper."""
    return _riesz_energy_and_grad(x, s=s)


def _energy_flat(x_flat, s=8.0, boundary_softness=100.0):
    """Backward-compatible wrapper."""
    return _riesz_energy_and_grad(x_flat, s=s)


def _energy_flat_with_s(x_flat, s):
    """Backward-compatible wrapper."""
    return _riesz_energy_and_grad(x_flat, s=s)


def _local_refinement_step(x, lr=0.001):
    """Backward-compatible: one step of gradient descent on smoothed min-distance."""
    n = len(x)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, np.inf)
    dists = np.sqrt(dists2)

    grad = np.zeros_like(x)
    for i in range(n):
        d_to_i = dists[i, :].copy()
        d_to_i[i] = np.inf
        j = np.argmin(d_to_i)
        if dists[i, j] > 1e-15:
            direction = (x[i] - x[j]) / dists[i, j]
            grad[i] += direction

    for i in range(n):
        for k in range(3):
            if x[i, k] > 0.45:
                grad[i, k] -= 0.1
            elif x[i, k] < -0.45:
                grad[i, k] += 0.1

    x_new = x - lr * grad
    x_new = np.clip(x_new, -0.5, 0.5)
    return x_new

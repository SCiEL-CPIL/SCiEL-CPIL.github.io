"""
Combined program: Multi-start violation-energy optimization with iterative
radius-increase, cKDTree-accelerated radius computation, and contact-graph
refinement. Combines the structural multi-start + radius-increase approach
of the high-scoring inspiration with efficient gradient-based refinement.
"""
import numpy as np
import time
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

def _energy_and_grad(x, target_r):
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
    signs[signs == 0] = 1.0
    grad_wall = 2.0 * wall_violations * signs

    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()

    return total_energy, total_grad

def _local_optimize_violation(init_pts, target_r, deadline, max_iter=2000):
    """
    L-BFGS-B minimization of the violation energy for a fixed target radius.
    Uses the global deadline to bound runtime.
    """
    def objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)

    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        opt_pts = result.x.reshape(N, 3)
        if np.all(np.isfinite(opt_pts)):
            return opt_pts
        return init_pts
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts

def _soft_min_objective_and_grad(flat_pos, beta=100.0):
    """
    Soft-min objective: maximize the soft-minimum of all effective distances.
    soft_min = -logsumexp(-beta * all_eff_dists) / beta
    """
    pos = flat_pos.reshape(N, 3)

    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)

    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]

    d_walls = HALF - np.abs(pos)
    wall_dists = d_walls.flatten()

    all_eff_dists = np.concatenate([pair_dists / 2.0, wall_dists])

    scaled = -beta * all_eff_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta

    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)

    grad_pos = np.zeros_like(pos)

    vecs = pos[iu[0]] - pos[iu[1]]
    d_vals = pair_dists

    valid = d_vals > 1e-14
    coeffs = np.zeros_like(d_vals)
    coeffs[valid] = weights[valid] / (2.0 * d_vals[valid])

    np.add.at(grad_pos, iu[0], coeffs[:, np.newaxis] * vecs)
    np.add.at(grad_pos, iu[1], -coeffs[:, np.newaxis] * vecs)

    w_walls = weights[len(pair_dists):]
    signs = np.sign(pos)
    signs[signs == 0] = 1.0
    grad_pos -= (w_walls.reshape(N, 3) * signs)

    return -soft_min_val, -grad_pos.flatten()

def _soft_min_optimize(pos, beta=100.0, max_iter=500, deadline=None):
    """Optimize using soft-min objective."""
    if deadline is not None and time.time() > deadline - 5.0:
        return pos

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)

    def objective(x):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError()
        return _soft_min_objective_and_grad(x, beta)

    try:
        res = minimize(
            objective,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if np.isfinite(res.fun):
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except TimeoutError:
        pass
    except Exception:
        pass
    return pos

def _generate_grid():
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _generate_bcc():
    """Generate BCC-like lattice points for initialization."""
    points = []
    for ix in range(5):
        for iy in range(5):
            for iz in range(4):
                x = (ix + 0.5) / 5.0 - 0.5
                y = (iy + 0.5) / 5.0 - 0.5
                z = (iz + 0.5) / 4.0 - 0.5
                points.append([x, y, z])
                xc = (ix + 1.0) / 5.0 - 0.5
                yc = (iy + 1.0) / 5.0 - 0.5
                zc = (iz + 1.0) / 4.0 - 0.5
                points.append([xc, yc, zc])
    return np.array(points[:N])

def _contact_graph_refine(centers, radius, deadline):
    """
    Identify near-contact pairs and refine only the involved spheres
    to maximize the minimum distance among those pairs (and wall).
    """
    if time.time() > deadline - 3.0:
        return centers, _compute_radius(centers)

    centers = centers.copy()
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.1 * radius)

    if len(pairs) == 0:
        return centers, _compute_radius(centers)

    involved = set()
    for i, j in pairs:
        involved.add(i)
        involved.add(j)
    involved = sorted(involved)

    if len(involved) < 2:
        return centers, _compute_radius(centers)

    idx_array = np.array(involved)

    def sub_objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(len(involved), 3)
        min_d = np.inf
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        return -(min_d / 2.0 + np.min(wall))

    def sub_grad(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(len(involved), 3)

        min_d = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
                min_pair = (i, j)

        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        min_wall_idx = np.argmin(wall)

        grad_full = np.zeros((N, 3))

        if min_pair is not None:
            i, j = min_pair
            d = c[i] - c[j]
            dist = min_d
            if dist > 1e-12:
                grad_full[i] += d / (2.0 * dist)
                grad_full[j] -= d / (2.0 * dist)

        i_wall = idx_array[min_wall_idx]
        abs_c = np.abs(c[i_wall])
        k_max = np.argmax(abs_c)
        grad_full[i_wall, k_max] += -np.sign(c[i_wall, k_max])

        return grad_full[idx_array].flatten()

    x0 = centers[idx_array].flatten()
    bounds_sub = [(-HALF + 1e-9, HALF - 1e-9)] * (len(involved) * 3)

    try:
        result = minimize(
            sub_objective,
            x0,
            jac=sub_grad,
            method='L-BFGS-B',
            bounds=bounds_sub,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_sub = result.x.reshape(len(involved), 3)
        centers[idx_array] = new_sub
    except TimeoutError:
        pass
    except Exception:
        pass

    centers = np.clip(centers, -HALF, HALF)
    return centers, _compute_radius(centers)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    best_centers = None
    best_radius = 0.0

    rng = np.random.default_rng(42)

    # Phase 1: Multi-start with violation-energy optimization
    initializations = []

    # Grid
    grid_centers = _generate_grid()
    initializations.append(grid_centers)

    # BCC
    bcc_centers = _generate_bcc()
    initializations.append(bcc_centers)

    # Perturbed grids
    for i in range(5):
        if time.time() > deadline - 50.0:
            break
        base = _generate_grid()
        noise_scale = 0.03 + 0.015 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)

    # Random
    for i in range(3):
        if time.time() > deadline - 40.0:
            break
        random_c = rng.uniform(-0.45, 0.45, size=(N, 3))
        initializations.append(random_c)

    # Target radii to try
    target_radii = [0.105, 0.110, 0.112, 0.114, 0.116]

    for init in initializations:
        if time.time() > deadline - 30.0:
            break

        for target_r in target_radii:
            if time.time() > deadline - 25.0:
                break

            # Add small noise to break symmetry
            noise = rng.normal(0, 0.003, size=(N, 3))
            init_pts = np.clip(init + noise, -0.49, 0.49)

            opt_pts = _local_optimize_violation(init_pts, target_r, deadline)

            if np.all(np.isfinite(opt_pts)):
                r = _compute_radius(opt_pts)
                if r > best_radius:
                    best_radius = r
                    best_centers = opt_pts.copy()

    if best_centers is None:
        best_centers = _generate_grid()
        best_radius = _compute_radius(best_centers)

    # Phase 2: Iterative radius increase with energy minimization
    for iteration in range(10):
        if time.time() > deadline - 20.0:
            break

        # Try to increase the radius slightly
        new_target = best_radius * 1.002

        # Add tiny noise to escape local minima
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(best_centers + noise, -HALF + 1e-9, HALF - 1e-9)

        opt_pts = _local_optimize_violation(init_pts, new_target, deadline, max_iter=1000)

        if np.all(np.isfinite(opt_pts)):
            r = _compute_radius(opt_pts)
            if r > best_radius:
                best_radius = r
                best_centers = opt_pts.copy()
            else:
                # Try with a smaller increment
                new_target = best_radius * 1.001
                init_pts = np.clip(best_centers + noise, -HALF + 1e-9, HALF - 1e-9)
                opt_pts = _local_optimize_violation(init_pts, new_target, deadline, max_iter=1000)
                if np.all(np.isfinite(opt_pts)):
                    r = _compute_radius(opt_pts)
                    if r > best_radius:
                        best_radius = r
                        best_centers = opt_pts.copy()

    # Phase 3: Soft-min polishing
    if time.time() < deadline - 10.0:
        polished = _soft_min_optimize(best_centers, beta=100.0, max_iter=500, deadline=deadline)
        r = _compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()

    # Phase 4: Contact graph refinement
    if time.time() < deadline - 5.0:
        best_centers, best_radius = _contact_graph_refine(best_centers, best_radius, deadline)

    # Final safety check
    if not np.all(np.isfinite(best_centers)):
        best_centers = _generate_grid()
        best_radius = _compute_radius(best_centers)

    # Ensure inside cube
    best_centers = np.clip(best_centers, -HALF + 1e-9, HALF - 1e-9)

    return best_centers

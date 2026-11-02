import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats.qmc import Sobol

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))

    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)

    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()

    # 2. Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]

    all_vals = np.concatenate([wall_vals, pair_vals_unique])

    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)

    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta

    return obj

def _soft_min_gradient(centers_flat, beta, iu):
    """
    Gradient of the soft-min objective.
    """
    centers = centers_flat.reshape(-1, 3)

    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()

    # 2. Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]

    all_vals = np.concatenate([wall_vals, pair_vals_unique])

    # Soft-min weights
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp

    # Gradient of soft-min w.r.t. all_vals is -weights
    grad_all = -weights

    # Wall gradient
    grad_wall = grad_all[:N*3].reshape(N, 3)
    # d/dx of (0.5 - |x|) is -sign(x)
    wall_signs = np.sign(centers)
    wall_signs[wall_signs == 0] = 1.0
    grad_centers = grad_wall * wall_signs

    # Pair gradient
    grad_pair_flat = grad_all[N*3:]
    i_idx, j_idx = iu
    for idx in range(len(i_idx)):
        i = i_idx[idx]
        j = j_idx[idx]
        diff_ij = centers[i] - centers[j]
        dist_ij = dists[i, j]
        if dist_ij > 1e-12:
            unit = diff_ij / dist_ij
            grad_centers[i] += grad_pair_flat[idx] * 0.5 * unit
            grad_centers[j] -= grad_pair_flat[idx] * 0.5 * unit

    return grad_centers.flatten()

def _build_initializers():
    """
    Generate diverse high-quality initial configurations.
    """
    inits = []

    # 1. Sobol Sequence (Quasi-random)
    sampler = Sobol(d=3, scramble=True, seed=42)
    points = sampler.random(N)
    centers = points * 1.0 - 0.5
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    inits.append(centers.copy())

    # 2. BCC-inspired Lattice
    # A 5x5x4 grid gives 100 points exactly.
    nx, ny, nz = 5, 5, 4
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
    centers = np.array(points[:N])
    rng = np.random.default_rng(123)
    centers += rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    inits.append(centers.copy())

    # 3. Halton Sequence
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    centers = np.zeros((N, 3))
    for i in range(1, N+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    rng = np.random.default_rng(456)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    inits.append(centers.copy())

    # 4. Random
    rng = np.random.default_rng(789)
    centers = rng.uniform(-0.45, 0.45, (N, 3))
    inits.append(centers)

    # 5. BCC Lattice (Body-Centered Cubic) - from inspiration
    # Use a 5x5x2 grid of BCC cells
    nx, ny, nz = 5, 5, 2
    n_cells = nx * ny * nz  # 50
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
                # Add body center offset
                points.append([cx + 0.5/nx, cy + 0.5/ny, cz + 0.5/nz])
    
    centers = np.array(points[:N])
    rng = np.random.default_rng(999)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    inits.append(centers.copy())

    return inits

def _polish_force_directed(centers, time_budget=10.0):
    """
    Force-directed local search. Fast, handles constraints explicitly.
    Uses KD-Tree for efficient neighbor lookup.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()

    start_time = time.time()

    step_size = 0.003
    decay = 0.995
    min_step = 1e-8
    noise_scale = 0.0003

    no_improve_count = 0
    max_no_improve = 80

    while _time_left() > 1.0 and (time.time() - start_time) < time_budget:
        r_current = _compute_radius(centers)

        if r_current > best_r + 1e-12:
            best_r = r_current
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count > max_no_improve:
            # Perturb to escape local minimum
            rng = np.random.default_rng()
            n_perturb = rng.integers(1, 4)
            for _ in range(n_perturb):
                idx = rng.integers(0, N)
                centers[idx] = rng.uniform(-0.45, 0.45, 3)
            no_improve_count = 0
            continue

        forces = np.zeros_like(centers)

        # 1. Wall Forces
        wall_limit = max(r_current * 1.03, 0.01)
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            mask = dists < wall_limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (wall_limit - dists[mask]) / wall_limit
                forces[mask, k] += dirs * strengths * 3.0

        # 2. Pairwise Forces using KD-Tree
        tree = cKDTree(centers)
        limit = 2.0 * r_current * 1.03
        pairs = tree.query_pairs(r=limit)
        
        for i, j in pairs:
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9 and dist < limit:
                strength = (limit - dist) / limit
                unit_diff = diff / dist
                forces[i] += strength * unit_diff * 3.0
                forces[j] -= strength * unit_diff * 3.0

        # Add noise
        rng = np.random.default_rng()
        noise = rng.normal(0, noise_scale, centers.shape)

        # Normalize and update
        total_forces = forces + noise
        norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        update_vec = total_forces / norms

        centers += step_size * update_vec
        centers = np.clip(centers, -0.4999, 0.4999)

        step_size *= decay
        if step_size < min_step:
            step_size = min_step
            if no_improve_count > 15:
                break

    return best_centers, best_r

def _polish_lbfgs(centers, time_budget=10.0):
    """
    L-BFGS-B optimization using soft-min objective.
    """
    start_time = time.time()
    iu = np.triu_indices(N, k=1)

    # Initial bounds
    bounds = [(-0.4999, 0.4999)] * (N * 3)

    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)

    # We try a few beta values, starting low and increasing.
    beta_start = 20.0
    beta_end = 5000.0
    n_beta_steps = 10

    current_centers = centers.copy()

    for step in range(n_beta_steps):
        if _time_left() < 2.0:
            break

        elapsed = time.time() - start_time
        remaining_for_beta = time_budget - elapsed
        if remaining_for_beta < 1.0:
            break

        beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))

        # For high beta, use fewer iterations but tighter tolerance
        if step < 3:
            max_iter = 200
            ftol = 1e-12
            gtol = 1e-8
        elif step < 6:
            max_iter = 150
            ftol = 1e-13
            gtol = 1e-9
        else:
            max_iter = 100
            ftol = 1e-14
            gtol = 1e-10

        try:
            res = minimize(
                _soft_min_objective,
                current_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 30}
            )
        except Exception:
            pass

        if res and res.x is not None:
            new_centers = res.x.reshape(N, 3)
            # Ensure validity
            if np.all(np.isfinite(new_centers)) and np.all(new_centers >= -0.5) and np.all(new_centers <= 0.5):
                new_r = _compute_radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
                # Continue from this point for higher beta
                current_centers = new_centers.copy()
            else:
                # Fallback to previous best
                current_centers = best_centers.copy()

    # Final high-precision polish with very high beta
    if _time_left() > 3.0 and (time.time() - start_time) < time_budget - 1.0:
        beta_final = 10000.0
        try:
            res = minimize(
                _soft_min_objective,
                best_centers.flatten(),
                args=(beta_final, iu),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
            )
            if res and res.x is not None:
                new_centers = res.x.reshape(N, 3)
                if np.all(np.isfinite(new_centers)) and np.all(new_centers >= -0.5) and np.all(new_centers <= 0.5):
                    new_r = _compute_radius(new_centers)
                    if new_r > best_r:
                        best_r = new_r
                        best_centers = new_centers.copy()
        except Exception:
            pass

    return best_centers, best_r

def _simulated_annealing(centers, time_budget=10.0):
    """
    Simulated annealing to escape local minima.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()

    start_time = time.time()
    rng = np.random.default_rng()

    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.995
    step_scale = 0.01

    no_improve = 0
    max_no_improve = 100

    while _time_left() > 1.0 and (time.time() - start_time) < time_budget and T > T_min:
        r_current = _compute_radius(centers)

        if r_current > best_r + 1e-12:
            best_r = r_current
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > max_no_improve:
            T *= 0.9
            no_improve = 0
            if T < T_min:
                break

        # Generate perturbation
        n_move = rng.integers(1, 4)
        new_centers = centers.copy()
        for _ in range(n_move):
            idx = rng.integers(0, N)
            delta = rng.normal(0, step_scale * T, 3)
            new_centers[idx] += delta
            new_centers[idx] = np.clip(new_centers[idx], -0.4999, 0.4999)

        new_r = _compute_radius(new_centers)
        delta_r = new_r - r_current

        # Metropolis criterion
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            centers = new_centers

        T *= cooling
        step_scale *= 0.999

    return best_centers, best_r

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0

    best_centers = None
    best_radius = 0.0

    inits = _build_initializers()

    # Phase 1: Force Directed Polish (Fast)
    # Budget: ~20s total.
    fd_results = []
    for i, centers in enumerate(inits):
        if _time_left() < 10.0:
            break
        budget = min(4.0, _time_left() - 5.0)
        if budget < 1.0:
            continue

        polished_centers, polished_r = _polish_force_directed(centers, time_budget=budget)
        fd_results.append((polished_r, polished_centers))

        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished_centers.copy()

    # Sort results by radius descending
    fd_results.sort(key=lambda x: x[0], reverse=True)

    # Phase 2: Simulated Annealing on top candidates
    if _time_left() > 15.0:
        sa_candidates = []
        for r, c in fd_results[:3]:
            sa_candidates.append((r, c))
        
        for r, c in sa_candidates:
            if _time_left() < 10.0:
                break
            sa_budget = min(8.0, _time_left() - 5.0)
            if sa_budget < 2.0:
                continue

            sa_centers, sa_r = _simulated_annealing(c, time_budget=sa_budget)
            if sa_r > best_radius:
                best_radius = sa_r
                best_centers = sa_centers.copy()
            fd_results.append((sa_r, sa_centers))

    # Re-sort after SA
    fd_results.sort(key=lambda x: x[0], reverse=True)

    # Phase 3: L-BFGS Polish (Slow, High Precision)
    # Take top 3 unique-ish starts from Phase 1+2
    candidates = []
    for r, c in fd_results:
        # Simple check: if it's not too close to an existing candidate
        is_unique = True
        for _, cc in candidates:
            if np.linalg.norm(c - cc) < 0.05:
                is_unique = False
                break
        if is_unique:
            candidates.append((r, c))
        if len(candidates) >= 3:
            break

    for r, c in candidates:
        if _time_left() < 15.0:
            break

        budget = min(25.0, _time_left() - 5.0)
        if budget < 2.0:
            continue

        polished_centers, polished_r = _polish_lbfgs(c, time_budget=budget)

        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished_centers.copy()

    # Phase 4: Final Force Directed Polish on the best result
    if best_centers is not None and _time_left() > 5.0:
        final_budget = _time_left() - 2.0
        polished_centers, polished_r = _polish_force_directed(best_centers, time_budget=final_budget)
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished_centers.copy()

    # Phase 5: Final high-precision L-BFGS polish
    if best_centers is not None and _time_left() > 8.0:
        final_lbgfs_budget = _time_left() - 3.0
        polished_centers, polished_r = _polish_lbfgs(best_centers, time_budget=final_lbgfs_budget)
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished_centers.copy()

    if best_centers is None:
        best_centers = inits[0]

    return best_centers

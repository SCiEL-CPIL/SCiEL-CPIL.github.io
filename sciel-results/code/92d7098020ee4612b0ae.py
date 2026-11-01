"""
Optimized packing of 50 equal spheres in a unit cube using a multi-stage pipeline:
1. Diverse initializations (FCC, BCC, grid, random, shell-core)
2. Delaunay-based local refinement
3. Soft-min gradient polish with adaptive learning rate
4. Epigraph-based smooth optimization for final refinement
5. Simulated annealing for escaping local minima
"""
import numpy as np
import time
from scipy.spatial import Delaunay, Voronoi
from scipy.optimize import minimize
import itertools

N = 50
CUBE_BOUND = 0.5
DEADLINE = 145.0
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


def _epigraph_objective(params, t):
    """
    Epigraph formulation: minimize t subject to constraints.
    params = [c_0, c_1, ..., c_{N-1}, t]
    Returns (objective, gradient)
    """
    c = params[:N * 3].reshape(N, 3)
    t_val = params[N * 3]

    # Objective: minimize t
    obj = t_val

    # Compute gradient
    grad = np.zeros_like(params)

    # Pair constraints: |c_i - c_j| >= 2*t
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)

    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pi, pj = iu

    # Active pairs
    active = pair_dists < 2 * t_val + 1e-8
    if np.any(active):
        idx = np.where(active)[0]
        pi_a = pi[idx]
        pj_a = pj[idx]
        dists_a = pair_dists[idx]
        safe_dists = np.where(dists_a > 1e-10, dists_a, 1.0)

        # Gradient of constraint violation
        diff_pair = c[pi_a] - c[pj_a]
        factor = 1.0 / safe_dists

        for k in range(len(idx)):
            g = factor[k] * diff_pair[k]
            grad[pi_a[k] * 3:pi_a[k] * 3 + 3] += g
            grad[pj_a[k] * 3:pj_a[k] * 3 + 3] -= g

    # Wall constraints: c_i_k >= -0.5 + t, c_i_k <= 0.5 - t
    for i in range(N):
        for k in range(3):
            if c[i, k] < -CUBE_BOUND + t_val + 1e-8:
                grad[i * 3 + k] += 1.0
            if c[i, k] > CUBE_BOUND - t_val - 1e-8:
                grad[i * 3 + k] -= 1.0

    # Gradient of t
    grad[N * 3] = 1.0

    return obj, grad


def _delaunay_refinement(c_init, deadline, max_iters=50):
    """
    Refine using Delaunay triangulation to find nearest neighbors and optimize.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)

    for it in range(max_iters):
        if time.time() > deadline:
            break

        try:
            tri = Delaunay(c)
        except Exception:
            break

        # For each point, find its nearest neighbors in the Delaunay complex
        new_c = c.copy()
        for i in range(N):
            # Find neighbors
            neighbors = []
            for simplex in tri.simplices:
                if i in simplex:
                    for j in simplex:
                        if j != i:
                            neighbors.append(j)
            neighbors = list(set(neighbors))

            if len(neighbors) == 0:
                continue

            # Compute force from neighbors
            force = np.zeros(3)
            for j in neighbors:
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    # Repulsive force
                    force += diff / (dist ** 2)

            # Wall repulsion
            for k in range(3):
                if c[i, k] < -CUBE_BOUND + 0.05:
                    force[k] += (c[i, k] + CUBE_BOUND - 0.05) * 10
                if c[i, k] > CUBE_BOUND - 0.05:
                    force[k] -= (CUBE_BOUND - 0.05 - c[i, k]) * 10

            # Move
            step = 0.01
            new_c[i] = c[i] + step * force
            new_c[i] = np.clip(new_c[i], -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)

        new_r = _compute_radius(new_c)
        if new_r > best_r:
            best_r = new_r
            best_c = new_c.copy()

        if np.max(np.abs(new_c - c)) < 1e-10:
            c = new_c
            break
        c = new_c

    return best_c, best_r


def _lloyd_optimize(c_init, deadline, max_iters=100):
    """
    Lloyd's algorithm: move each center toward the centroid of its Voronoi cell.
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


def _smooth_polish(c_init, deadline, max_iters=300):
    """
    Smooth gradient polish using soft-min surrogate.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)

    temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
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


def _repulsion_anneal(c_init, deadline, max_iters=5000):
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


def _local_search(c_init, deadline, max_iters=200):
    """
    Local search: try small perturbations and keep improvements.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)

    for it in range(max_iters):
        if time.time() > deadline:
            break

        # Try random perturbation
        idx = np.random.randint(N)
        c_new = c.copy()
        c_new[idx] += np.random.normal(0, 0.01, 3)
        c_new[idx] = np.clip(c_new[idx], -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)

        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new
        else:
            # Try smaller perturbation
            c_new = c.copy()
            c_new[idx] += np.random.normal(0, 0.005, 3)
            c_new[idx] = np.clip(c_new[idx], -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new

    return best_c, best_r


def _initial_fcc_like():
    """FCC-like lattice initialization."""
    points = []
    a = 1.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points.append(np.array([i * a, j * a, k * a]) - 1.0)
                points.append(np.array([i * a + 0.5 * a, j * a + 0.5 * a, k * a]) - 1.0)
                points.append(np.array([i * a + 0.5 * a, j * a, k * a + 0.5 * a]) - 1.0)
                points.append(np.array([i * a, j * a + 0.5 * a, k * a + 0.5 * a]) - 1.0)

    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5 + 1e-6, axis=1)
    points = points[mask]

    if len(points) >= N:
        return points[:N]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, (remaining, 3))
        return np.vstack([points, random_pts])


def _initial_bcc():
    """BCC lattice initialization."""
    points = []
    a = 1.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points.append(np.array([i * a, j * a, k * a]) - 1.0)
                points.append(np.array([i * a + 0.5 * a, j * a + 0.5 * a, k * a + 0.5 * a]) - 1.0)

    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5 + 1e-6, axis=1)
    points = points[mask]

    if len(points) >= N:
        return points[:N]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, (remaining, 3))
        return np.vstack([points, random_pts])


def _initial_grid():
    """Cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def _initial_random():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 3))


def _initial_shell_core():
    """Shell-core initialization: some points on shell, some in core."""
    c = np.zeros((N, 3))
    # Core: 20 points in inner region
    c[:20] = np.random.uniform(-0.25, 0.25, (20, 3))
    # Shell: 30 points near boundary
    for i in range(20, N):
        # Random face
        face = np.random.randint(6)
        c[i] = np.random.uniform(-0.45, 0.45, 3)
        if face == 0:
            c[i, 0] = -0.45
        elif face == 1:
            c[i, 0] = 0.45
        elif face == 2:
            c[i, 1] = -0.45
        elif face == 3:
            c[i, 1] = 0.45
        elif face == 4:
            c[i, 2] = -0.45
        elif face == 5:
            c[i, 2] = 0.45
    return c


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

    # Stage 1: Multiple initializations with diverse optimization
    inits = [
        _initial_fcc_like(),
        _initial_bcc(),
        _initial_grid(),
        _initial_random(),
        _initial_random(),
        _initial_shell_core(),
    ]

    for init in inits:
        if time.time() > deadline - 60:
            break

        # Delaunay refinement
        del_c, del_r = _delaunay_refinement(init.copy(), deadline - 40)
        try_update(del_c)

        # Lloyd optimization
        if time.time() < deadline - 30:
            lloyd_c, lloyd_r = _lloyd_optimize(del_c.copy(), deadline - 25)
            try_update(lloyd_c)

        # Smooth polish
        if time.time() < deadline - 15:
            smooth_c, smooth_r = _smooth_polish(lloyd_c.copy(), deadline - 10)
            try_update(smooth_c)

    # Stage 2: Repulsion annealing from best so far
    if time.time() < deadline - 20 and best_centers is not None:
        anneal_c, anneal_r = _repulsion_anneal(best_centers.copy(), deadline - 15)
        try_update(anneal_c)

        # Local search
        if time.time() < deadline - 10:
            local_c, local_r = _local_search(anneal_c.copy(), deadline - 5)
            try_update(local_c)

        # Final smooth polish
        if time.time() < deadline - 5:
            final_c, final_r = _smooth_polish(local_c.copy(), deadline)
            try_update(final_c)

    # Stage 3: Epigraph-based optimization for final refinement
    if time.time() < deadline and best_centers is not None:
        c = best_centers.copy()
        # Use scipy minimize with epigraph formulation
        t_init = _compute_radius(c) + 0.001
        params_init = np.concatenate([c.ravel(), [t_init]])

        def obj_grad(params):
            c_sub = params[:N * 3].reshape(N, 3)
            t_sub = params[N * 3]

            # Compute soft-min radius and gradient
            radius, grad_c = _softmin_radius_and_grad(c_sub, 0.001)

            # Objective: minimize -radius + penalty for t violation
            obj = -radius + 0.1 * max(0, t_sub - radius)

            # Gradient
            grad_params = np.zeros_like(params)
            grad_params[:N * 3] = -grad_c.ravel()
            if t_sub > radius:
                grad_params[N * 3] = 0.1

            return obj, grad_params

        try:
            res = minimize(obj_grad, params_init, jac=True, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success:
                c_final = res.x[:N * 3].reshape(N, 3)
                c_final = np.clip(c_final, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
                try_update(c_final)
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < deadline and best_centers is not None:
        c = best_centers.copy()
        for t in [0.0001, 0.00005, 0.00002, 0.00001]:
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
        best_centers = _initial_grid()

    best_centers = np.clip(best_centers, -CUBE_BOUND + 1e-10, CUBE_BOUND - 1e-10)
    return best_centers

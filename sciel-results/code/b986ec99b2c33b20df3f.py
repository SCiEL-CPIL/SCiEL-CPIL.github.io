"""Combined solver: multi-stage optimization for 50 equal spheres in unit cube.
Stages: seeds → repulsion annealing → Lloyd/Voronoi balancing → softmin gradient polish → local search → Metropolis MC.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5, 0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, cKDTree

START_TIME = time.time()
TIME_LIMIT = 130.0  # Safety margin before 150s kill

# Precompute upper-triangle pair indices (used everywhere)
_iu, _ju = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_iu)

def _compute_radius(pts):
    """Compute the maximum valid radius for given centers (N,3)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3):
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    diff = pts[_iu] - pts[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _softmin_radius_and_grad(x, beta=100.0):
    """
    Smooth surrogate: softmin of all constraints (pair distances/2, wall distances).
    Returns (negative softmin value, flattened gradient) for L-BFGS-B minimization.
    Uses precomputed pair indices for O(N^2) without full matrix allocation.
    """
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6, np.zeros_like(x)

    # Pair distances / 2
    diff = pts[_iu] - pts[_ju]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists_sq = np.maximum(dists_sq, 1e-14)
    dists = np.sqrt(dists_sq)
    pair_vals = dists / 2.0

    # Wall distances
    wall_vals = (0.5 - np.abs(pts)).ravel()

    all_vals = np.concatenate([pair_vals, wall_vals])
    d_min = np.min(all_vals)
    shifted = -beta * (all_vals - d_min)
    shifted = np.clip(shifted, -700, 0.0)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    log_sum = np.log(sum_exp)
    soft_min_val = d_min - log_sum / beta

    weights = exp_vals / sum_exp

    grad_pts = np.zeros_like(pts)

    # Pair contributions (vectorized)
    valid_dists = np.maximum(dists, 1e-8)
    dir_vecs = diff / valid_dists[:, np.newaxis]
    pair_w = weights[:_NUM_PAIRS] * 0.5

    # grad on pts[i] += pair_w * dir_vec ; grad on pts[j] -= pair_w * dir_vec
    np.add.at(grad_pts, _iu, pair_w[:, np.newaxis] * dir_vecs)
    np.add.at(grad_pts, _ju, -pair_w[:, np.newaxis] * dir_vecs)

    # Wall contributions
    wall_w = weights[_NUM_PAIRS:].reshape(N, 3)
    signs = np.sign(pts)
    grad_pts += wall_w * (-signs)

    return -soft_min_val, grad_pts.ravel()

def _lloyd_optimize(pts, n_iters=80, deadline=None):
    """
    Lloyd's algorithm: iteratively move each center to the centroid of its
    Voronoi cell (clipped to the cube). Produces balanced spacing.
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)

    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        try:
            vor = Voronoi(pts)
        except Exception:
            break

        new_pts = pts.copy()
        for i in range(N):
            region = vor.regions[vor.point_region[i]]
            if -1 in region or len(region) < 3:
                # Fallback: use nearest neighbors via KDTree
                tree = cKDTree(pts)
                _, idx = tree.query(pts[i], k=10)
                centroid = np.mean(pts[idx], axis=0)
            else:
                verts = vor.vertices[region]
                # Clip vertices to cube
                verts = np.clip(verts, -0.5, 0.5)
                centroid = np.mean(verts, axis=0)
            new_pts[i] = np.clip(centroid, -0.499, 0.499)

        # Damped update to avoid divergence
        pts = 0.5 * pts + 0.5 * new_pts
        pts = np.clip(pts, -0.499, 0.499)

        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts, best_r

def _repulsion_anneal(pts, n_steps=80, lr_start=0.008, lr_end=0.001, deadline=None):
    """
    Repulsion annealing: push spheres apart using inverse-square forces.
    Vectorized pairwise computation using precomputed indices.
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        lr = lr_start * (lr_end / lr_start) ** (step / max(1, n_steps - 1))

        forces = np.zeros_like(pts)

        # Pairwise repulsion (vectorized with precomputed indices)
        diff = pts[_iu] - pts[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        dists = np.maximum(dists, 1e-8)
        inv_dist3 = 1.0 / (dists ** 3 + 1e-10)
        pair_forces = inv_dist3[:, np.newaxis] * diff

        np.add.at(forces, _iu, pair_forces)
        np.add.at(forces, _ju, -pair_forces)

        # Wall repulsion
        for k in range(3):
            pos = pts[:, k]
            wall_pos = np.maximum(0.5 - pos, 1e-4)
            wall_neg = np.maximum(0.5 + pos, 1e-4)
            f_pos = np.where(wall_pos < 0.12, -1.0 / (wall_pos ** 2 + 0.001), 0.0)
            f_neg = np.where(wall_neg < 0.12, 1.0 / (wall_neg ** 2 + 0.001), 0.0)
            forces[:, k] += f_pos + f_neg

        pts += lr * forces
        pts = np.clip(pts, -0.498, 0.498)

        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts, best_r

def _local_search(pts, n_steps=2000, deadline=None):
    """
    Local search: find the most violated constraint (pair or wall) and nudge to fix it.
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    rng = np.random.default_rng(123)

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        diff = pts[_iu] - pts[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_d = np.min(dists) / 2.0
        min_pair_idx = np.argmin(dists)
        i, j = _iu[min_pair_idx], _ju[min_pair_idx]

        wall_dists = 0.5 - np.abs(pts)
        min_wall = np.min(wall_dists)
        wi, wk = np.unravel_index(np.argmin(wall_dists), (N, 3))

        if min_pair_d < min_wall:
            direction = pts[i] - pts[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
            improved = False
            for scale in (0.004, 0.002, 0.001, 0.0005, 0.0002):
                trial = pts.copy()
                trial[i] += scale * direction
                trial[j] -= scale * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    pts = trial
                    best_r = r_trial
                    best_pts = trial.copy()
                    improved = True
                    break
            if improved:
                continue
        else:
            sgn = np.sign(pts[wi, wk])
            if sgn == 0:
                sgn = 1.0
            direction = np.zeros(3)
            direction[wk] = sgn
            improved = False
            for scale in (0.004, 0.002, 0.001, 0.0005, 0.0002):
                trial = pts.copy()
                trial[wi] += scale * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    pts = trial
                    best_r = r_trial
                    best_pts = trial.copy()
                    improved = True
                    break
            if improved:
                continue

        # Periodic random perturbation to escape local optima
        if step % 150 == 0 and step > 0:
            for scale in (0.008, 0.004, 0.002):
                trial = pts.copy()
                trial += rng.normal(scale=scale, size=(N, 3))
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    pts = trial
                    best_r = r_trial
                    best_pts = trial.copy()
                    break

    return best_pts, best_r

def _metropolis_mc(pts, n_steps=1500, temp_start=0.04, temp_end=0.003, deadline=None):
    """
    Metropolis Monte Carlo: accept moves that improve radius or with Boltzmann probability.
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    rng = np.random.default_rng(42)

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = step / max(n_steps - 1, 1)
        T = temp_start * (temp_end / temp_start) ** frac

        trial = pts.copy()
        trial += rng.normal(scale=T, size=(N, 3))
        trial = np.clip(trial, -0.499, 0.499)

        r_trial = _compute_radius(trial)
        r_cur = _compute_radius(pts)

        if r_trial >= r_cur:
            pts = trial
            if r_trial > best_r:
                best_r = r_trial
                best_pts = trial.copy()
        else:
            delta_r = r_cur - r_trial
            if delta_r < T * 8:
                accept_prob = np.exp(-delta_r / max(T, 1e-10))
                if rng.random() < accept_prob:
                    pts = trial

    return best_pts, best_r

def _seed_fcc():
    """FCC-inspired lattice, farthest-point sampled to N points."""
    m = 4
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    px = x + dx * spacing
                    py = y + dy * spacing
                    pz = z + dz * spacing
                    if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5 and -0.5 <= pz <= 0.5:
                        points.append([px, py, pz])
    points = np.array(points)
    if len(points) < N:
        rng = np.random.default_rng(0)
        extra = rng.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        # Farthest-point sampling
        sel = [int(np.argmin(np.sum(points**2, axis=1)))]
        rem = list(range(len(points)))
        rem.remove(sel[0])
        while len(sel) < N and rem:
            dm = np.array([np.min(np.linalg.norm(points[idx] - points[sel], axis=1)) for idx in rem])
            bi = int(np.argmax(dm))
            sel.append(rem[bi])
            rem.pop(bi)
        points = points[sel]
    return points

def _seed_grid():
    """4x4x4 grid, randomly sampled to N."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.default_rng(5)
    idx = rng.choice(len(grid), N, replace=False)
    return grid[idx]

def _seed_bcc():
    """BCC lattice, farthest-point sampled."""
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
                points.append([x + 0.5*spacing, y + 0.5*spacing, z + 0.5*spacing])
    points = np.array(points)
    if len(points) < N:
        rng = np.random.default_rng(1)
        extra = rng.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        sel = [int(np.argmin(np.sum(points**2, axis=1)))]
        rem = list(range(len(points)))
        rem.remove(sel[0])
        while len(sel) < N and rem:
            dm = np.array([np.min(np.linalg.norm(points[idx] - points[sel], axis=1)) for idx in rem])
            bi = int(np.argmax(dm))
            sel.append(rem[bi])
            rem.pop(bi)
        points = points[sel]
    return points

def _seed_shell_core():
    """Spherical shell + central grid cluster."""
    rng = np.random.default_rng(7)
    shell_n = 35
    core_n = N - shell_n
    idx = np.arange(shell_n)
    phi = np.arccos(1 - 2 * (idx + 0.5) / shell_n)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    R = 0.33
    shell = np.column_stack([
        R * np.sin(phi) * np.cos(theta),
        R * np.sin(phi) * np.sin(theta),
        R * np.cos(phi),
    ])
    shell = np.clip(shell, -0.45, 0.45)
    ticks = (np.arange(3) + 0.5) / 3 - 0.5
    core = np.array([(x * 0.18, y * 0.18, z * 0.18)
                     for z in ticks for y in ticks for x in ticks])[:core_n]
    return np.vstack([shell, core])

def _seed_random(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _polish_softmin(c, deadline):
    """Run softmin gradient optimization with increasing beta."""
    c_cur = c.copy()
    r_cur = _compute_radius(c_cur)
    x0 = c_cur.ravel()
    bounds = [(-0.499, 0.499)] * (N * 3)

    for beta in (30.0, 80.0, 200.0, 500.0, 1000.0):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _softmin_radius_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                args=(beta,),
                options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 40},
            )
            c_new = np.clip(res.x.reshape(N, 3), -0.499, 0.499)
            r_new = _compute_radius(c_new)
            if r_new > r_cur:
                r_cur = r_new
                c_cur = c_new
            x0 = c_cur.ravel()
        except Exception:
            pass

    return c_cur, r_cur

def construct():
    """Maximize the common radius of 50 equal spheres in [-0.5,0.5]^3."""
    start = time.time()
    deadline = start + (150.0 - 20.0)  # 20s safety margin

    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_bcc(),
        _seed_shell_core(),
        _seed_random(1),
        _seed_random(2),
        _seed_random(3),
    ]

    best_c = None
    best_r = -1.0

    for si, s in enumerate(seeds):
        if time.time() > deadline:
            break
        s = np.clip(s, -0.49, 0.49)

        # Stage A: Repulsion annealing
        c_an, r_an = _repulsion_anneal(s, n_steps=80, deadline=deadline)

        # Stage B: Lloyd's algorithm for balanced spacing
        c_lloyd, r_lloyd = _lloyd_optimize(c_an, n_iters=60, deadline=deadline)
        if r_lloyd > r_an:
            c_cur = c_lloyd
            r_cur = r_lloyd
        else:
            c_cur = c_an
            r_cur = r_an

        # Stage C: Softmin gradient polish
        if time.time() < deadline:
            c_cur, r_cur = _polish_softmin(c_cur, deadline)

        # Stage D: Local search on binding constraints
        if time.time() < deadline:
            c_ls, r_ls = _local_search(c_cur, n_steps=1500, deadline=deadline)
            if r_ls > r_cur:
                r_cur = r_ls
                c_cur = c_ls

        # Stage E: Metropolis MC
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(c_cur, n_steps=1200, deadline=deadline)
            if r_mc > r_cur:
                r_cur = r_mc
                c_cur = c_mc

        if r_cur > best_r:
            best_r = r_cur
            best_c = c_cur.copy()

    # Final global polish on the best result
    if best_c is not None and time.time() < deadline:
        # One more round of softmin at high beta
        c_final, r_final = _polish_softmin(best_c, deadline)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final

        # Local search
        if time.time() < deadline:
            c_ls, r_ls = _local_search(best_c, n_steps=1000, deadline=deadline)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls

        # Final MC
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(best_c, n_steps=800, deadline=deadline)
            if r_mc > best_r:
                best_r = r_mc
                best_c = c_mc

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

# EVOLVE-BLOCK-END

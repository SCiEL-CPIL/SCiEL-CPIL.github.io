"""Combined solver: Voronoi-Lloyd + analytic soft-min gradient polish + repulsion + local search.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5, 0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, cKDTree

DEADLINE_MARGIN = 15.0  # safety margin before 150s kill

# Precompute upper-triangle pair indices
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)

# Cube corners for Voronoi bounding
_CORNERS = np.array([[sx, sy, sz] for sz in (-0.5, 0.5) for sy in (-0.5, 0.5) for sx in (-0.5, 0.5)])


def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    return min(min_pair / 2.0, min_wall)


def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


def _lloyd_step(c, deadline, steps=200, damping=0.5):
    """Voronoi-based Lloyd's algorithm using scipy Voronoi + cKDTree.
    Adds cube corners as extra sites so Voronoi cells are bounded."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)

    for it in range(steps):
        if time.time() > deadline:
            break

        try:
            # Build Voronoi on points + corners
            pts = np.vstack([c, _CORNERS])
            vor = Voronoi(pts)
            n_sites = len(pts)
            n_real = N

            # Collect all Voronoi vertices (finite ones)
            verts = vor.vertices
            finite_mask = np.all(np.isfinite(verts), axis=1)
            verts = verts[finite_mask]
            if len(verts) == 0:
                break

            # For each Voronoi vertex, find which region(s) it belongs to
            # vor.regions: list of vertex indices per region
            # We need to assign each vertex to the site whose region contains it.
            # Efficient approach: for each vertex, check which regions contain it.
            # But that's O(num_verts * num_regions). Instead, use the fact that
            # each vertex is shared by 3-4 regions. We'll build a vertex->regions map.

            # Build mapping: vertex_index_in_full_array -> list of site indices
            # vor.vertices is indexed 0..len-1, vor.regions references these indices.
            # But some regions have -1 (unbounded). We filtered finite verts.
            # Actually vor.vertices contains ALL vertices including unbounded (-1 entries).
            # Let's redo: use vor.vertices as-is, and only consider finite ones.

            # Rebuild: get all finite vertices with their original indices
            all_verts = vor.vertices  # (M, 3), some rows have -1
            finite_orig_idx = np.where(np.all(all_verts != -1, axis=1))[0]
            finite_verts = all_verts[finite_orig_idx]
            if len(finite_verts) == 0:
                break

            # For each finite vertex, determine which site it belongs to.
            # A vertex belongs to a region if the vertex index is in that region's vertex list.
            # Build a set-based lookup: for each region, store its vertex indices as a set.
            # Then for each finite vertex, check which regions contain it.
            # This is O(num_verts * num_regions) but num_regions = N+8 = 58, num_verts ~ few hundred.
            
            region_vertex_sets = []
            for reg in vor.regions:
                if len(reg) == 0:
                    region_vertex_sets.append(set())
                else:
                    region_vertex_sets.append(set(reg))

            # For each finite vertex (original index), find which sites claim it
            # A vertex is on the boundary of multiple regions; assign to the nearest real site.
            # Actually for Voronoi cells, a vertex belongs to all regions that list it.
            # For Lloyd, we want the centroid of each site's cell.
            # A vertex contributes to all sites whose region contains it.
            
            cell_sums = np.zeros((N, 3))
            cell_counts = np.zeros(N)
            
            # For each finite vertex, find which real-site regions contain it
            # Build reverse: vertex_orig_idx -> list of site indices
            vertex_to_sites = {}
            for site_idx in range(n_real):  # only real sites
                if site_idx >= len(region_vertex_sets):
                    continue
                vs = region_vertex_sets[site_idx]
                for vi in vs:
                    if vi not in vertex_to_sites:
                        vertex_to_sites[vi] = []
                    vertex_to_sites[vi].append(site_idx)
            
            # Now for each finite vertex, add its coordinates to all sites that claim it
            for orig_idx in finite_orig_idx:
                if orig_idx in vertex_to_sites:
                    v = all_verts[orig_idx]
                    for si in vertex_to_sites[orig_idx]:
                        cell_sums[si] += v
                        cell_counts[si] += 1
            
            valid = cell_counts > 0
            if np.any(valid):
                targets = cell_sums[valid] / cell_counts[valid, None]
                c[valid] = c[valid] + damping * (targets - c[valid])

            c = np.clip(c, -0.49, 0.49)

        except Exception:
            # Fallback: simple repulsion step
            d = _pair_dist_matrix(c)
            inv = 1.0 / np.maximum(d, 1e-6) ** 2
            diff = c[:, None, :] - c[None, :, :]
            F = np.sum(diff * inv[:, :, None], axis=1)
            c = np.clip(c + 0.01 * F, -0.49, 0.49)

    return c


def _lloyd_optimize(c, deadline):
    """Iterate Lloyd's algorithm with decreasing damping."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)

    for damp in (0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05):
        if time.time() > deadline:
            break
        steps = max(30, int(200 * damp))
        c = _lloyd_step(c, deadline, steps=steps, damping=damp)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c.copy()

    return best, best_r


def _softmin_radius_and_grad(c, t):
    """Compute softmin radius and its gradient analytically.
    Returns (softmin_value, gradient) where gradient is (N,3)."""
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0

    wall_vals = 0.5 - np.abs(c)

    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    m_val = np.min(all_vals)

    # Weights (shifted by min for stability)
    pair_weights = np.exp(-(pair_vals - m_val) / t)
    wall_weights = np.exp(-(wall_vals - m_val) / t)

    total_sum = np.sum(pair_weights) + np.sum(wall_weights)

    if total_sum < 1e-300:
        return m_val, np.zeros_like(c)

    softmin_val = m_val - t * np.log(total_sum)

    # Gradient
    grad_c = np.zeros_like(c)

    # Pair contributions
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / np.maximum(dists, 1e-12)[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0

    # For pair (i,j): d/dc_i (d_ij/2) = 0.5 * (c_i - c_j)/d_ij
    # softmin = -t * log(sum w_k), w_k = exp(-val_k/t)
    # d(softmin)/dc_i = -t * (1/sum) * sum_k dw_k/dc_i
    # dw_k/dc_i = w_k * (-1/t) * d(val_k)/dc_i
    # So d(softmin)/dc_i = -t * (1/sum) * sum_k w_k * (-1/t) * d(val_k)/dc_i
    #               = (1/sum) * sum_k w_k * d(val_k)/dc_i
    # For pair (i,j): d(val_k)/dc_i = 0.5 * (c_i-c_j)/d_ij = 0.5 * unit_vec
    #                d(val_k)/dc_j = -0.5 * unit_vec

    pair_grad_i = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
    # Scatter to grad_c
    for idx in range(_NUM_PAIRS):
        i, j = _PAIR_IDX[idx]
        grad_c[i] += pair_grad_i[idx]
        grad_c[j] -= pair_grad_i[idx]

    # Wall contributions: d(val)/dc_ik = -sign(c_ik)
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    grad_c += wall_weights * signs

    grad_c /= total_sum

    return softmin_val, grad_c


def _smooth_obj_and_grad(x, t):
    """Objective to MINIMIZE: negative softmin radius plus boundary penalty.
    Returns (value, gradient)."""
    c = x.reshape(N, 3)
    r, grad_r = _softmin_radius_and_grad(c, t)
    # Boundary penalty
    over = np.maximum(0.0, np.abs(c) - 0.5 + 1e-6)
    pen = np.sum(over ** 2)
    pen_grad = 2.0 * over * np.sign(c)
    pen_grad[pen_grad == 0] = 0.0

    val = -r + 1e3 * pen
    grad = -grad_r + 1e3 * pen_grad
    return val, grad


def _smooth_polish(c, deadline):
    """Polish with analytic soft-min gradient via L-BFGS-B, annealing temperature."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)

    for t in (0.05, 0.02, 0.008, 0.003, 0.001, 0.0005, 0.0002):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12,
                         'maxls': 50},
            )
            c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c.copy()
            x0 = c.flatten()
        except Exception:
            pass
    return best, best_r


def _repulsion_anneal(c, deadline, steps=3000):
    """Vectorized repulsion + wall-attraction gradient descent."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)

    for it in range(steps):
        if time.time() > deadline:
            break
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        inv = 1.0 / np.maximum(dists, 1e-6) ** 2
        unit_vecs = diff / np.maximum(dists, 1e-12)[:, np.newaxis]
        unit_vecs[np.isnan(unit_vecs)] = 0.0
        pair_forces = (inv * 0.5)[:, np.newaxis] * unit_vecs

        F = np.zeros_like(c)
        for idx in range(_NUM_PAIRS):
            i, j = _PAIR_IDX[idx]
            F[i] += pair_forces[idx]
            F[j] -= pair_forces[idx]

        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))

        frac = it / max(steps - 1, 1)
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        c = c + lr * (rep_scale * F + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)

    return c


def _metropolis_mc(c, deadline, iters=3000):
    """Metropolis Monte Carlo: random displacements with acceptance based on radius."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(123)

    T0 = 0.003
    T_min = 0.0001

    for it in range(iters):
        if time.time() > deadline:
            break

        frac = it / max(iters - 1, 1)
        T = T0 * (T_min / T0) ** frac

        trial = c.copy()
        trial += rng.normal(scale=T, size=(N, 3))
        trial = np.clip(trial, -0.499, 0.499)

        r_trial = _compute_radius(trial)
        r_cur = _compute_radius(c)

        if r_trial >= r_cur:
            c = trial
            if r_trial > best_r:
                best_r = r_trial
                best_c = c.copy()
        else:
            delta_r = r_cur - r_trial
            if delta_r < T * 10:
                accept_prob = np.exp(-delta_r / max(T, 1e-10))
                if rng.random() < accept_prob:
                    c = trial

    return best_c, best_r


def _local_search(c, deadline, iters=4000):
    """Local search using cKDTree for fast nearest-pair identification."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(42)

    for it in range(iters):
        if time.time() > deadline:
            break

        # Find closest pair via cKDTree
        tree = cKDTree(c)
        pairs = tree.query_pairs(r=0.5)  # all pairs within 0.5
        if len(pairs) == 0:
            # No close pairs; check walls
            wall = 0.5 - np.abs(c)
            min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
            i, k = min_wall_idx
            sgn = np.sign(c[i, k])
            if sgn == 0:
                sgn = 1.0
            direction = np.zeros(3)
            direction[k] = sgn

            improved = False
            for step in (0.005, 0.002, 0.001, 0.0005, 0.0002):
                trial = c.copy()
                trial[i] += step * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    improved = True
                    break
            if improved:
                continue
        else:
            # Find the closest pair
            min_d = np.inf
            min_pair = None
            for (i, j) in pairs:
                d = np.linalg.norm(c[i] - c[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)

            if min_pair is None:
                continue

            i, j = min_pair
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)

            improved = False
            for step in (0.005, 0.002, 0.001, 0.0005, 0.0002):
                trial = c.copy()
                trial[i] += step * direction
                trial[j] -= step * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    improved = True
                    break
            if improved:
                continue

        # Random perturbation to escape local optima
        if it % 80 == 0 and it > 0:
            for step in (0.01, 0.005, 0.002, 0.001):
                trial = c.copy()
                trial += rng.normal(scale=step, size=(N, 3))
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    break

    return best_c, best_r


def _seed_fcc():
    """FCC lattice supercell, then farthest-point sample to N points."""
    pts = []
    a = 1.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    pts = np.array(pts)
    pts = pts - pts.mean(axis=0)
    me = np.max(np.abs(pts))
    if me > 0:
        pts = pts / me * 0.45

    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = list(range(len(pts)))
    rem.remove(sel[0])
    while len(sel) < N and rem:
        dm = np.full(len(rem), np.inf)
        for ri, idx in enumerate(rem):
            dmin = np.min(np.linalg.norm(pts[idx] - pts[sel], axis=1))
            dm[ri] = dmin
        bi = int(np.argmax(dm))
        sel.append(rem[bi])
        rem.pop(bi)
    return pts[sel]


def _seed_grid():
    m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N]


def _seed_random(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def _seed_shell_core():
    """Points on a spherical shell plus a central cluster."""
    rng = np.random.default_rng(7)
    shell_n = 35
    core_n = N - shell_n
    idx = np.arange(shell_n)
    phi = np.arccos(1 - 2 * (idx + 0.5) / shell_n)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    R = 0.32
    shell = np.column_stack([
        R * np.sin(phi) * np.cos(theta),
        R * np.sin(phi) * np.sin(theta),
        R * np.cos(phi),
    ])
    shell = np.clip(shell, -0.45, 0.45)
    ticks = (np.arange(3) + 0.5) / 3 - 0.5
    core = np.array([(x * 0.2, y * 0.2, z * 0.2)
                     for z in ticks for y in ticks for x in ticks])[:core_n]
    return np.vstack([shell, core])


def _seed_bcc():
    """BCC lattice points, scaled to fit in cube."""
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=float)
    pts = (pts - 1.0)
    pts = pts / np.max(np.abs(pts)) * 0.45
    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = list(range(len(pts)))
    rem.remove(sel[0])
    while len(sel) < N and rem:
        dm = np.full(len(rem), np.inf)
        for ri, idx in enumerate(rem):
            dmin = np.min(np.linalg.norm(pts[idx] - pts[sel], axis=1))
            dm[ri] = dmin
        bi = int(np.argmax(dm))
        sel.append(rem[bi])
        rem.pop(bi)
    return pts[sel]


def construct():
    """Maximize the common radius of 50 equal spheres in [-0.5,0.5]^3."""
    start = time.time()
    deadline = start + (150.0 - DEADLINE_MARGIN)

    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_shell_core(),
        _seed_bcc(),
        _seed_random(1),
        _seed_random(2),
    ]

    best_c = None
    best_r = -1.0

    for si, s in enumerate(seeds):
        if time.time() > deadline:
            break
        s = np.clip(s, -0.49, 0.49)

        # Stage A: repulsion annealing
        c_an = _repulsion_anneal(s, deadline, steps=1500)
        r_an = _compute_radius(c_an)

        # Stage B: Lloyd's algorithm (Voronoi equalization)
        c_lloyd, r_lloyd = _lloyd_optimize(c_an, deadline)

        # Stage C: smooth softmin polish on the best of A and B
        c_cur = c_lloyd if r_lloyd >= r_an else c_an
        r_cur = max(r_lloyd, r_an)

        if time.time() < deadline:
            c_pol, r_pol = _smooth_polish(c_cur, deadline)
            if r_pol > r_cur:
                r_cur = r_pol
                c_cur = c_pol

        # Stage D: local search
        if time.time() < deadline:
            c_ls, r_ls = _local_search(c_cur, deadline, iters=2000)
            if r_ls > r_cur:
                r_cur = r_ls
                c_cur = c_ls

        # Stage E: Metropolis MC to escape local minima
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(c_cur, deadline, iters=1500)
            if r_mc > r_cur:
                r_cur = r_mc
                c_cur = c_mc

        if r_cur > best_r:
            best_r = r_cur
            best_c = c_cur.copy()

    # Final extra polish on the global best if time remains
    if best_c is not None and time.time() < deadline:
        c_pol, r_pol = _smooth_polish(best_c, deadline)
        if r_pol > best_r:
            best_r = r_pol
            best_c = c_pol

        if time.time() < deadline:
            c_ls, r_ls = _local_search(best_c, deadline, iters=1500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls

        # Metropolis MC to escape any remaining local minima
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(best_c, deadline, iters=1500)
            if r_mc > best_r:
                best_r = r_mc
                best_c = c_mc

        # Final high-precision polish with very tight tolerance
        if time.time() < deadline - 5:
            c_pol2, r_pol2 = _smooth_polish(best_c, deadline - 2)
            if r_pol2 > best_r:
                best_r = r_pol2
                best_c = c_pol2

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

# EVOLVE-BLOCK-END

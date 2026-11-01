"""Combined solver: smooth softmin max-min + Nelder-Mead exact polish + Metropolis MC.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5,0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

DEADLINE_MARGIN = 12.0  # safety margin before 150s kill


def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return -1.0
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return -1.0
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_wall = np.min(0.5 - np.abs(c))
    return min(min_pair, min_wall)


def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    return d


def _softmin_radius(c, t):
    """Smooth approximation of the true radius via log-sum-exp over all
    constraints (pair/2 and wall). Returns a smooth lower-bound-ish value
    that increases as the packing tightens. t = temperature (small = sharp)."""
    d = _pair_dist_matrix(c)
    # Use only upper triangle to avoid double-counting pairs
    iu = np.triu_indices(N, k=1)
    pair = d[iu] / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair, wall.ravel()])
    vals = vals[np.isfinite(vals)]
    m = np.min(vals)
    s = np.sum(np.exp(-(vals - m) / t))
    return m - t * np.log(s)


def _smooth_obj(x, t):
    """Objective to MINIMIZE: negative smooth radius plus a small wall-push."""
    c = x.reshape(N, 3)
    r = _softmin_radius(c, t)
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    return -r + 1e3 * pen


def _neg_score(centers):
    """Negative exact score for minimisation."""
    return -_compute_radius(centers.reshape(N, 3))


def _time_left(deadline):
    return deadline - time.time()


def _lloyd_equalize(c, deadline, steps=2000):
    """Voronoi-Lloyd equalization: each point moves toward the centroid of its
    Voronoi cell (approximated via nearest-neighbor regions), which tends to
    equalize local densities and improve the minimum distance."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)
        inv = 1.0 / np.maximum(d, 1e-6) ** 3
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv[:, :, None], axis=1)
        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k])) ** 2
        frac = it / max(steps - 1, 1)
        lr = 1e-3 * (1.0 - 0.5 * frac)
        c = c + lr * (F + 0.1 * wall_force)
        c = np.clip(c, -0.49, 0.49)
    return c


def _repulsion_anneal(c, deadline, steps=None):
    """Vectorized repulsion + wall-attraction gradient descent."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    if steps is None:
        steps = 4000
    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv[:, :, None], axis=1)
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


def _smooth_polish(c, deadline):
    """Polish a configuration with the smooth softmin max-min via L-BFGS-B,
    annealing the temperature down for a sharp min."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)
    for t in (0.05, 0.02, 0.008, 0.003, 0.001, 0.0004, 0.0001):
        if _time_left(deadline) < 10.0:
            break
        try:
            res = minimize(
                _smooth_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-11,
                         'maxls': 60},
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


def _exact_polish(c, deadline):
    """Nelder-Mead on the exact min-radius objective, very tight."""
    c = np.clip(c.copy(), -0.49, 0.49)
    best = c
    best_r = _compute_radius(c)

    try:
        res = minimize(
            _neg_score, c.flatten(), method='Nelder-Mead',
            options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14,
                     'adaptive': True},
        )
        c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c
    except Exception:
        pass

    if _time_left(deadline) > 12.0:
        try:
            res = minimize(
                _neg_score, best.flatten(), method='Nelder-Mead',
                options={'maxiter': 6000, 'xatol': 1e-13, 'fatol': 1e-15,
                         'adaptive': True},
            )
            c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c
        except Exception:
            pass

    return best, best_r


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
    rem = [i for i in range(len(pts)) if i != sel[0]]
    while len(sel) < N and rem:
        if len(sel) > 1:
            dm = np.min(_pair_dist_matrix(pts[sel + rem]), axis=0)
        else:
            dm = np.full(len(rem), np.inf)
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
    """Place points on a spherical shell plus a central cluster."""
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


def _seed_layered(seed):
    """Layered arrangement: 5 layers x 10 points."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 3))
    idx = 0
    for layer_i in range(5):
        z = -0.4 + 0.8 * layer_i / 4.0
        for j in range(10):
            x = -0.4 + 0.8 * (j % 5) / 4.0
            y = -0.4 + 0.8 * (j // 5) / 1.0
            centers[idx] = [x, y, z]
            idx += 1
    centers += rng.uniform(-0.02, 0.02, (N, 3))
    return np.clip(centers, -0.49, 0.49)


def _metropolis_mc(c, deadline, steps=5000, seed=42):
    """Metropolis Monte Carlo: random displacements with energy-based acceptance."""
    rng = np.random.default_rng(seed)
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    r0 = max(best_r, 1e-6)
    T0 = 0.02 * r0
    T_final = 1e-6 * r0

    for it in range(steps):
        if _time_left(deadline) < 10.0:
            break
        frac = it / max(steps - 1, 1)
        T = T0 * (T_final / T0) ** frac
        if T < T_final:
            T = T_final

        move_type = rng.random()
        if move_type < 0.8:
            i = rng.integers(N)
            step_size = 0.05 * r0 * (1.0 - 0.5 * frac)
            delta = rng.normal(0, step_size, size=3)
            c_try = c.copy()
            c_try[i] += delta
            c_try = np.clip(c_try, -0.499, 0.499)
        else:
            i, j = rng.choice(N, size=2, replace=False)
            c_try = c.copy()
            c_try[i], c_try[j] = c_try[j].copy(), c_try[i].copy()
            c_try = np.clip(c_try, -0.499, 0.499)

        r_try = _compute_radius(c_try)
        r_cur = _compute_radius(c)
        delta_E = -(r_try - r_cur)

        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
            c = c_try.copy()
            if r_try > best_r:
                best_r = r_try
                best_c = c.copy()

    return best_c, best_r


def _local_search(c, deadline, iters=2000):
    """Local search: perturb the closest pair and nearest wall, accept if radius improves."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()

    for it in range(iters):
        if _time_left(deadline) < 10.0:
            break

        d = _pair_dist_matrix(c)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]

        wall = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        min_wall_val = wall[min_wall_idx]

        if min_pair_dist / 2.0 < min_wall_val:
            i, j = min_pair_idx
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0, 0])
            else:
                direction = direction / norm

            for step in (0.002, 0.005, 0.01, 0.02, 0.003, 0.008, 0.015):
                c_try = c.copy()
                c_try[i] += step * direction
                c_try[j] -= step * direction
                c_try = np.clip(c_try, -0.499, 0.499)
                r_try = _compute_radius(c_try)
                if r_try > best_r + 1e-12:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
        else:
            i, k = min_wall_idx
            sgn = np.sign(c[i, k])
            if sgn == 0:
                sgn = 1.0

            for step in (0.002, 0.005, 0.01, 0.02, 0.003, 0.008, 0.015):
                c_try = c.copy()
                c_try[i, k] += step * sgn
                c_try = np.clip(c_try, -0.499, 0.499)
                r_try = _compute_radius(c_try)
                if r_try > best_r + 1e-12:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break

        if it % 100 == 50:
            c_try = c.copy()
            idx = np.random.randint(N)
            c_try[idx] += np.random.normal(0, 0.01, size=3)
            c_try = np.clip(c_try, -0.499, 0.499)
            r_try = _compute_radius(c_try)
            if r_try > best_r + 1e-12:
                best_r = r_try
                best_c = c_try.copy()
                c = c_try.copy()

    return best_c, best_r


def construct():
    """Maximize the common radius of 50 equal spheres in [-0.5,0.5]^3."""
    start = time.time()
    deadline = start + (150.0 - DEADLINE_MARGIN)

    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_shell_core(),
        _seed_layered(11),
        _seed_layered(22),
        _seed_random(1),
        _seed_random(2),
    ]

    best_c = None
    best_r = -1.0

    for si, s in enumerate(seeds):
        if _time_left(deadline) < 30.0:
            break
        s = np.clip(s, -0.49, 0.49)

        # Stage A: repulsion annealing
        c_an = _repulsion_anneal(s, deadline, steps=2500)
        r_an = _compute_radius(c_an)

        # Stage A2: Lloyd equalization
        if _time_left(deadline) > 30.0:
            c_lloyd = _lloyd_equalize(c_an, deadline, steps=1500)
            r_lloyd = _compute_radius(c_lloyd)
            if r_lloyd > r_an:
                c_an = c_lloyd
                r_an = r_lloyd

        # Stage B: smooth softmin polish
        c_pol, r_pol = _smooth_polish(c_an, deadline)

        # Stage C: exact Nelder-Mead polish on the best
        c_best_stage = c_pol if r_pol >= r_an else c_an
        r_best_stage = max(r_pol, r_an)
        if _time_left(deadline) > 25.0:
            c_nm, r_nm = _exact_polish(c_best_stage, deadline)
            if r_nm > r_best_stage:
                r_best_stage = r_nm
                c_best_stage = c_nm

        # Stage D: local search
        if _time_left(deadline) > 20.0:
            c_ls, r_ls = _local_search(c_best_stage, deadline, iters=800)
            if r_ls > r_best_stage:
                r_best_stage = r_ls
                c_best_stage = c_ls

        # Stage E: Metropolis MC
        if _time_left(deadline) > 20.0:
            c_mc, r_mc = _metropolis_mc(c_best_stage, deadline, steps=3000, seed=si + 100)
            if r_mc > r_best_stage:
                r_best_stage = r_mc
                c_best_stage = c_mc
                # Quick local search after MC
                if _time_left(deadline) > 15.0:
                    c_ls2, r_ls2 = _local_search(c_best_stage, deadline, iters=400)
                    if r_ls2 > r_best_stage:
                        r_best_stage = r_ls2
                        c_best_stage = c_ls2
                # Another exact polish after MC
                if _time_left(deadline) > 15.0:
                    c_nm2, r_nm2 = _exact_polish(c_best_stage, deadline)
                    if r_nm2 > r_best_stage:
                        r_best_stage = r_nm2
                        c_best_stage = c_nm2

        if r_best_stage > best_r:
            best_r = r_best_stage
            best_c = c_best_stage.copy()

    # Final extra polish on the global best if time remains
    if best_c is not None and _time_left(deadline) > 15.0:
        c_pol, r_pol = _smooth_polish(best_c, deadline)
        if r_pol > best_r:
            best_r = r_pol
            best_c = c_pol

        if _time_left(deadline) > 15.0:
            c_nm, r_nm = _exact_polish(best_c, deadline)
            if r_nm > best_r:
                best_r = r_nm
                best_c = c_nm

        if _time_left(deadline) > 15.0:
            c_mc, r_mc = _metropolis_mc(best_c, deadline, steps=4000, seed=999)
            if r_mc > best_r:
                best_r = r_mc
                best_c = c_mc

        if _time_left(deadline) > 10.0:
            c_ls, r_ls = _local_search(best_c, deadline, iters=800)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls

        if _time_left(deadline) > 10.0:
            c_nm2, r_nm2 = _exact_polish(best_c, deadline)
            if r_nm2 > best_r:
                best_r = r_nm2
                best_c = c_nm2

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

# EVOLVE-BLOCK-END

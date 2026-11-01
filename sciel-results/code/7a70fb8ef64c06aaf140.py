"""Rewritten solver: smooth softmin max-min + vectorized repulsion annealing.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5,0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_MARGIN = 12.0  # safety margin before 150s kill


def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
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


def _softmin_radius(c, t):
    """Smooth approximation of the true radius via log-sum-exp over all
    constraints (pair/2 and wall). Returns a smooth lower-bound-ish value
    that increases as the packing tightens. t = temperature (small = sharp)."""
    d = _pair_dist_matrix(c)
    pair = d / 2.0                       # (N,N)
    wall = 0.5 - np.abs(c)               # (N,3)
    # gather all constraint values
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    # log-sum-exp for min: min ≈ -logsumexp(-vals/t)*t
    m = np.min(vals)
    s = np.sum(np.exp(-(vals - m) / t))
    return m - t * np.log(s)


def _smooth_obj(x, t):
    """Objective to MINIMIZE: negative smooth radius plus a small wall-push."""
    c = x.reshape(N, 3)
    # keep strictly inside with a tiny margin to avoid boundary issues
    r = _softmin_radius(c, t)
    # small quadratic penalty if any coordinate escapes the cube
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    return -r + 1e3 * pen


def _repulsion_anneal(c, deadline, steps=None):
    """Vectorized repulsion + wall-attraction gradient descent.
    Different search family from the smooth max-min: a physical relaxation
    that spreads points and pushes them toward walls, then tightens."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    if steps is None:
        steps = 4000
    # anneal temperature of the repulsion kernel and the wall strength
    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)                       # (N,N)
        # repulsive force: for each pair, push apart with strength ~ 1/(d^2)
        # but only meaningful when d is small relative to a scale
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        # force on i from j = (c_i - c_j) * inv_ij  (repulsion)
        diff = c[:, None, :] - c[None, :, :]           # (N,N,3)
        F = np.sum(diff * inv[:, :, None], axis=1)     # (N,3)
        # wall attraction: push toward nearest wall face
        # gradient of wall distance: for coordinate k, if c_ik>0 attract to +0.5
        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            # stronger attraction as the point gets closer to the centre
            # (we want to maximize min wall, so push toward walls)
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))
        # anneal: early = strong repulsion to spread, late = strong wall push
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
    for t in (0.05, 0.02, 0.008, 0.003, 0.001):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-10,
                         'maxls': 40},
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
    # farthest point sampling
    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = [i for i in range(len(pts)) if i != sel[0]]
    while len(sel) < N and rem:
        dmin = np.min(_pair_dist_matrix(pts[sel]), axis=1) if len(sel) > 1 \
            else np.full(len(pts), np.inf)
        # for each remaining, min dist to selected
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
    """Place points on a spherical shell (near the cube's inscribed sphere)
    plus a central cluster; gives diverse wall-near / interior mix."""
    rng = np.random.default_rng(7)
    # 35 on a shell of radius ~0.33, 15 in a small inner grid
    shell_n = 35
    core_n = N - shell_n
    # fibonacci sphere for even shell distribution
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
    # inner grid
    ticks = (np.arange(3) + 0.5) / 3 - 0.5
    core = np.array([(x * 0.2, y * 0.2, z * 0.2)
                     for z in ticks for y in ticks for x in ticks])[:core_n]
    return np.vstack([shell, core])


def _metropolis_mc(c, deadline, steps=5000, seed=42):
    """Metropolis Monte Carlo: random displacements and swaps with energy-based
    acceptance to escape local minima. Energy = -radius (minimize)."""
    rng = np.random.default_rng(seed)
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    # Initial temperature: allow moves that degrade radius by up to ~5%
    r0 = max(best_r, 1e-6)
    T0 = 0.02 * r0
    T_final = 1e-6 * r0
    
    for it in range(steps):
        if time.time() > deadline:
            break
        
        # Anneal temperature geometrically
        frac = it / max(steps - 1, 1)
        T = T0 * (T_final / T0) ** frac
        if T < T_final:
            T = T_final
        
        # Choose move type: 80% displacement, 20% swap
        move_type = rng.random()
        
        if move_type < 0.8:
            # Random displacement of one point
            i = rng.integers(N)
            # Step size scales with current radius
            step_size = 0.05 * r0 * (1.0 - 0.5 * frac)
            delta = rng.normal(0, step_size, size=3)
            c_try = c.copy()
            c_try[i] += delta
            c_try = np.clip(c_try, -0.499, 0.499)
        else:
            # Swap two points (helps with symmetry breaking)
            i, j = rng.choice(N, size=2, replace=False)
            c_try = c.copy()
            c_try[i], c_try[j] = c_try[j].copy(), c_try[i].copy()
            c_try = np.clip(c_try, -0.499, 0.499)
        
        r_try = _compute_radius(c_try)
        delta_E = -(r_try - best_r)  # Energy = -radius, so delta_E = -(r_try - r_best)
        
        # Metropolis acceptance
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
        if time.time() > deadline:
            break
        
        # Find the binding constraints
        d = _pair_dist_matrix(c)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]
        
        wall = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        min_wall_val = wall[min_wall_idx]
        
        # Determine which constraint is binding
        if min_pair_dist / 2.0 < min_wall_val:
            # Binding pair
            i, j = min_pair_idx
            # Perturb these two points apart
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0, 0])
            else:
                direction = direction / norm
            
            # Try several step sizes
            for step in (0.005, 0.01, 0.02, 0.003, 0.008):
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
            # Binding wall
            i, k = min_wall_idx
            # Push point i away from the wall
            sgn = np.sign(c[i, k])
            if sgn == 0:
                sgn = 1.0
            
            for step in (0.005, 0.01, 0.02, 0.003, 0.008):
                c_try = c.copy()
                c_try[i, k] += step * sgn
                c_try = np.clip(c_try, -0.499, 0.499)
                r_try = _compute_radius(c_try)
                if r_try > best_r + 1e-12:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
        
        # Occasionally add random perturbation to escape local minima
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

        # Stage A: repulsion annealing (cheap, parallelizable)
        c_an = _repulsion_anneal(s, deadline, steps=2500)
        r_an = _compute_radius(c_an)

        # Stage B: smooth softmin polish
        c_pol, r_pol = _smooth_polish(c_an, deadline)

        # Stage C: local search on the best of A and B
        c_ls = c_pol if r_pol >= r_an else c_an
        r_ls = max(r_pol, r_an)
        if time.time() < deadline:
            c_ls, r_ls = _local_search(c_ls, deadline, iters=800)

        # Stage D: Metropolis Monte Carlo to escape local minima
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(c_ls, deadline, steps=3000, seed=si + 100)
            if r_mc > r_ls:
                r_ls = r_mc
                c_ls = c_mc
                # Quick local search after MC
                if time.time() < deadline:
                    c_ls, r_ls = _local_search(c_ls, deadline, iters=400)

        r = max(r_an, r_pol, r_ls)
        c = c_ls if r_ls >= max(r_an, r_pol) else (c_pol if r_pol >= r_an else c_an)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # Final extra polish on the global best if time remains
    if best_c is not None and time.time() < deadline:
        c_pol, r_pol = _smooth_polish(best_c, deadline)
        if r_pol > best_r:
            best_r = r_pol
            best_c = c_pol
        
        # Final Metropolis MC to escape any remaining local minima
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(best_c, deadline, steps=4000, seed=999)
            if r_mc > best_r:
                best_r = r_mc
                best_c = c_mc
        
        # Final local search
        if time.time() < deadline:
            c_ls, r_ls = _local_search(best_c, deadline, iters=500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

# EVOLVE-BLOCK-END

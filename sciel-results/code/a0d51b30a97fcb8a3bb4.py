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


def _seed_bcc():
    """BCC lattice supercell, then farthest-point sample to N points."""
    pts = []
    a = 1.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    pts = np.array(pts)
    pts = pts - pts.mean(axis=0)
    me = np.max(np.abs(pts))
    if me > 0:
        pts = pts / me * 0.45
    # farthest point sampling
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


def _epigraph_obj(x, t, eps=1e-8):
    """Epigraph formulation: minimize -t subject to t <= all constraints.
    x = [c.flatten(), t] where t is the radius variable.
    We minimize -t (i.e., maximize t) with soft constraints."""
    c = x[:N*3].reshape(N, 3)
    t_val = x[N*3]
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    # soft constraint: penalty if t_val > any val
    viol = np.maximum(0, t_val - vals)
    pen = np.sum(viol**2)
    return -t_val + 1e4 * pen


def _epigraph_grad(x, t, eps=1e-8):
    """Gradient for the epigraph formulation."""
    c = x[:N*3].reshape(N, 3)
    t_val = x[N*3]
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    viol = np.maximum(0, t_val - vals)
    # gradient of penalty w.r.t. c
    grad_c = np.zeros(N*3)
    # pair contributions
    for i in range(N):
        for j in range(i+1, N):
            diff = c[i] - c[j]
            dist = np.sqrt(np.sum(diff**2))
            if dist < 1e-10:
                continue
            p_val = dist / 2.0
            if t_val > p_val:
                viol_i = t_val - p_val
                # d(pen)/d(c_i) += 2*viol_i * d(p_val)/d(c_i)
                # p_val = dist/2, d(dist)/d(c_i) = diff/dist
                g = 2 * viol_i * 0.5 * diff / dist
                grad_c[i*3:(i+1)*3] += g
                grad_c[j*3:(j+1)*3] -= g
    # wall contributions
    for i in range(N):
        for k in range(3):
            w_val = 0.5 - abs(c[i,k])
            if t_val > w_val:
                viol_i = t_val - w_val
                if c[i,k] > 0:
                    g = 2 * viol_i * (-1.0)  # d(w_val)/d(c_ik) = -1
                else:
                    g = 2 * viol_i * (1.0)   # d(w_val)/d(c_ik) = +1
                grad_c[i*3+k] += g
    # gradient w.r.t. t
    grad_t = -1.0 + 2 * 1e4 * np.sum(viol)
    grad = np.zeros(N*3+1)
    grad[:N*3] = grad_c
    grad[N*3] = grad_t
    return grad


def _epigraph_polish(c, deadline):
    """Polish using epigraph formulation with auxiliary variable t."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c
    best_r = _compute_radius(c)
    t0 = best_r
    x0 = np.concatenate([c.flatten(), [t0]])
    bounds = [(-0.49, 0.49)] * (N*3) + [(0.0, 0.5)]
    for t in (0.03, 0.01, 0.005, 0.002):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _epigraph_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10,
                         'maxls': 30},
            )
            c_new = np.clip(res.x[:N*3].reshape(N, 3), -0.49, 0.49)
            r = _compute_radius(c_new)
            if r > best_r:
                best_r = r
                best = c_new.copy()
            x0 = np.concatenate([c_new.flatten(), [best_r]])
        except Exception:
            pass
    return best, best_r


def _local_search(c, deadline, n_perturb=200):
    """Perturbation-based local search: randomly nudge points and polish."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c
    best_r = _compute_radius(c)
    rng = np.random.default_rng(42)
    for it in range(n_perturb):
        if time.time() > deadline:
            break
        c_try = c.copy()
        # random perturbation
        idx = rng.integers(0, N, size=rng.integers(1, 5))
        scale = 0.01 * (1.0 - it/n_perturb) + 0.001
        c_try[idx] += rng.normal(0, scale, size=(len(idx), 3))
        c_try = np.clip(c_try, -0.49, 0.49)
        # quick polish
        c_pol, r_pol = _smooth_polish(c_try, deadline)
        if r_pol > best_r:
            best_r = r_pol
            best = c_pol.copy()
            c = c_pol.copy()
    return best, best_r


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
        _seed_random(3),
    ]

    best_c = None
    best_r = -1.0

    for si, s in enumerate(seeds):
        if time.time() > deadline:
            break
        s = np.clip(s, -0.49, 0.49)

        # Stage A: repulsion annealing (cheap, parallelizable)
        c_an = _repulsion_anneal(s, deadline, steps=2000)
        r_an = _compute_radius(c_an)

        # Stage B: smooth softmin polish
        c_pol, r_pol = _smooth_polish(c_an, deadline)

        # Stage C: epigraph polish
        c_ep, r_ep = _epigraph_polish(c_pol, deadline)

        r = max(r_an, r_pol, r_ep)
        c = c_ep if r_ep == r else (c_pol if r_pol == r else c_an)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # Final extra polish on the global best if time remains
    if best_c is not None and time.time() < deadline:
        c_pol, r_pol = _smooth_polish(best_c, deadline)
        if r_pol > best_r:
            best_r = r_pol
            best_c = c_pol
        c_ep, r_ep = _epigraph_polish(best_c, deadline)
        if r_ep > best_r:
            best_r = r_ep
            best_c = c_ep

    # Local search on the best if time remains
    if best_c is not None and time.time() < deadline:
        c_ls, r_ls = _local_search(best_c, deadline, n_perturb=100)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

# EVOLVE-BLOCK-END

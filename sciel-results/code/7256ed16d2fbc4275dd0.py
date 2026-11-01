"""Rewritten solver: multi-start with Voronoi-Lloyd + smooth epigraph polish + local search.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5,0.5]^3."""
N = 50

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

DEADLINE_MARGIN = 12.0


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


def _softmin_radius(c, beta):
    """Smooth approximation of min constraint via log-sum-exp. beta = inverse temp."""
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    m = np.min(vals)
    s = np.sum(np.exp(-(vals - m) * beta))
    return m - np.log(s) / beta


def _epigraph_obj(x, beta):
    """Epigraph: x = [c_flat (150), t (1)]. Minimize -t + penalty.
    t should be <= all constraints. We maximize t subject to soft constraints."""
    c = x[:N * 3].reshape(N, 3)
    t = x[N * 3]
    # Soft constraint: t <= each constraint value
    # Minimize: -t + sum of soft violations
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    # Soft penalty for t exceeding any constraint
    viol = np.maximum(0.0, t - vals)
    # Smooth: use soft-max of violations
    v_max = np.max(viol)
    if v_max > 1e-15:
        pen = v_max + np.log(np.sum(np.exp((viol - v_max) * beta))) / beta
    else:
        pen = 0.0
    # Boundary penalty
    bpen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-4) ** 2)
    return -t + 1e3 * pen + 1e5 * bpen


def _epigraph_polish(c, deadline):
    """Polish using epigraph formulation with L-BFGS-B, annealing beta."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best = c.copy()
    best_r = _compute_radius(c)
    
    for beta in (100.0, 500.0, 2000.0, 10000.0, 50000.0, 200000.0):
        if time.time() > deadline:
            break
        x0 = np.concatenate([c.flatten(), [_compute_radius(c) - 1e-6]])
        bounds = [(-0.499, 0.499)] * (N * 3) + [(0.0, 0.5)]
        try:
            res = minimize(
                _epigraph_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(beta,),
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 80},
            )
            c_new = np.clip(res.x[:N * 3].reshape(N, 3), -0.499, 0.499)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best = c_new.copy()
            c = c_new
        except Exception:
            pass
    return best, best_r


def _lloyd_step(c, deadline, steps=150, damping=0.5):
    """Voronoi-based Lloyd's algorithm."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    
    for it in range(steps):
        if time.time() > deadline:
            break
        try:
            tri = Delaunay(c)
        except Exception:
            d = _pair_dist_matrix(c)
            inv = 1.0 / np.maximum(d, 1e-6) ** 2
            diff = c[:, None, :] - c[None, :, :]
            F = np.sum(diff * inv[:, :, None], axis=1)
            c = np.clip(c + 0.01 * F, -0.49, 0.49)
            continue
        
        simplices = tri.simplices
        cell_centroids = np.zeros((N, 3))
        cell_counts = np.zeros(N)
        
        for idx in range(len(simplices)):
            si = simplices[idx]
            v0, v1, v2, v3 = c[si[0]], c[si[1]], c[si[2]], c[si[3]]
            A = np.array([v1 - v0, v2 - v0, v3 - v0])
            b = np.array([
                np.dot(v1, v1) - np.dot(v0, v0),
                np.dot(v2, v2) - np.dot(v0, v0),
                np.dot(v3, v3) - np.dot(v0, v0)
            ])
            try:
                cc = np.linalg.solve(2.0 * A, b)
            except np.linalg.LinAlgError:
                cc = np.mean([v0, v1, v2, v3])
            for vi in si:
                cell_centroids[vi] += cc
                cell_counts[vi] += 1
        
        valid = cell_counts > 0
        if np.any(valid):
            targets = cell_centroids[valid] / cell_counts[valid, None]
            c[valid] = c[valid] + damping * (targets - c[valid])
        
        c = np.clip(c, -0.49, 0.49)
    
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


def _repulsion_anneal(c, deadline, steps=2000):
    """Vectorized repulsion + wall-attraction gradient descent."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    
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


def _local_search(c, deadline, iters=3000):
    """Local search: perturb binding constraints with adaptive steps."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(42)
    
    step_size = 0.005
    no_improve_count = 0
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        d = _pair_dist_matrix(c)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]
        
        wall = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        min_wall_val = wall[min_wall_idx]
        
        improved = False
        
        if min_pair_dist / 2.0 < min_wall_val:
            i, j = min_pair_idx
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
            
            for step in (step_size, step_size * 0.5, step_size * 2.0, step_size * 0.25):
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
        else:
            i, k = min_wall_idx
            sgn = np.sign(c[i, k])
            if sgn == 0:
                sgn = 1.0
            direction = np.zeros(3)
            direction[k] = sgn
            
            for step in (step_size, step_size * 0.5, step_size * 2.0, step_size * 0.25):
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
            no_improve_count = 0
            step_size = min(step_size * 1.1, 0.01)
        else:
            no_improve_count += 1
            if no_improve_count % 50 == 0:
                step_size = max(step_size * 0.8, 0.0001)
        
        if no_improve_count >= 100 and it % 50 == 0:
            for step in (0.01, 0.005, 0.002, 0.001):
                trial = c.copy()
                trial += rng.normal(scale=step, size=(N, 3))
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    no_improve_count = 0
                    break
    
    return best_c, best_r


def _metropolis_mc(c, deadline, iters=3000):
    """Metropolis Monte Carlo."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(123)
    
    T0 = 0.002
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
        else:
            delta = r_cur - r_trial
            if delta < 1e-12:
                c = trial
            else:
                prob = np.exp(-delta / max(T, 1e-8))
                if rng.random() < prob:
                    c = trial
        
        if _compute_radius(c) > best_r:
            best_r = _compute_radius(c)
            best_c = c.copy()
    
    return best_c, best_r


def _seed_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N]


def _seed_fcc():
    pts = []
    a = 1.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i * a, j * a, k * a])
                pts.append([(i + 0.5) * a, (j + 0.5) * a, k * a])
                pts.append([(i + 0.5) * a, j * a, (k + 0.5) * a])
                pts.append([i * a, (j + 0.5) * a, (k + 0.5) * a])
    pts = np.array(pts)
    pts = pts - pts.mean(axis=0)
    me = np.max(np.abs(pts))
    if me > 0:
        pts = pts / me * 0.45
    
    sel = [int(np.argmin(np.sum(pts ** 2, axis=1)))]
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


def _seed_bcc():
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=float)
    pts = (pts - 1.0)
    pts = pts / np.max(np.abs(pts)) * 0.45
    sel = [int(np.argmin(np.sum(pts ** 2, axis=1)))]
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


def _seed_random(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def _seed_shell_core():
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


def _optimize_one(s, deadline):
    """Run the full optimization pipeline on a single seed."""
    s = np.clip(s, -0.49, 0.49)
    
    # Stage A: repulsion annealing
    c_an = _repulsion_anneal(s, deadline, steps=1500)
    r_an = _compute_radius(c_an)
    
    # Stage B: Lloyd's algorithm
    c_lloyd, r_lloyd = _lloyd_optimize(c_an, deadline)
    
    # Pick best so far
    c_cur = c_lloyd if r_lloyd >= r_an else c_an
    r_cur = max(r_lloyd, r_an)
    
    # Stage C: epigraph polish
    if time.time() < deadline:
        c_pol, r_pol = _epigraph_polish(c_cur, deadline)
        if r_pol > r_cur:
            r_cur = r_pol
            c_cur = c_pol
    
    # Stage D: Metropolis MC
    if time.time() < deadline:
        c_mc, r_mc = _metropolis_mc(c_cur, deadline, iters=2000)
        if r_mc > r_cur:
            r_cur = r_mc
            c_cur = c_mc
    
    # Stage E: local search
    if time.time() < deadline:
        c_ls, r_ls = _local_search(c_cur, deadline, iters=2000)
        if r_ls > r_cur:
            r_cur = r_ls
            c_cur = c_ls
    
    return c_cur, r_cur


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
        c_cur, r_cur = _optimize_one(s, deadline)
        if r_cur > best_r:
            best_r = r_cur
            best_c = c_cur.copy()
    
    # Final polish on global best
    if best_c is not None and time.time() < deadline:
        c_pol, r_pol = _epigraph_polish(best_c, deadline)
        if r_pol > best_r:
            best_r = r_pol
            best_c = c_pol.copy()
        
        if time.time() < deadline:
            c_ls, r_ls = _local_search(best_c, deadline, iters=1500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
    
    if best_c is None:
        best_c = _seed_grid()
    
    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

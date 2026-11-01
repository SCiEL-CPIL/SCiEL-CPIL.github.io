"""Combined solver: Voronoi-Lloyd equalization + softmin gradient polish + repulsion + local search.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5,0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

DEADLINE_MARGIN = 15.0  # safety margin before 150s kill

# Precompute upper triangle indices for efficient pairwise operations
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)


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


def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient using log-sum-exp for numerical stability.
    Returns (softmin_value, gradient) where gradient is (N, 3).
    """
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Log-sum-exp for numerical stability
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    m_val = np.min(all_vals)
    
    # Weights (unnormalized)
    pair_weights = np.exp(-pair_vals / t)
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return m_val, np.zeros_like(c)
    
    # Softmin value
    softmin_val = m_val - t * np.log(total_sum)
    
    # Gradient calculation
    grad_c = np.zeros_like(c)
    
    # Pair contributions: for pair (i,j), d(pair_val)/dc[i] = -0.5 * (c[i]-c[j])/|c[i]-c[j]|
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # grad_c[i] -= 0.5 * w_ij * unit_vec_ij
    # grad_c[j] += 0.5 * w_ij * unit_vec_ij
    pair_grad = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
    for idx in range(_NUM_PAIRS):
        i, j = _PAIR_IDX[idx]
        grad_c[i] -= pair_grad[idx]
        grad_c[j] += pair_grad[idx]
    
    # Wall contributions: d(wall_val)/dc[i,k] = -sign(c[i,k])
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    grad_c -= wall_weights * signs
    
    # Normalize by total_sum (since softmin = m - t*log(sum), grad = -t * sum(w*grad_val)/sum(w))
    # Actually: d/dc[m - t*log(S)] where S = sum(w_k)
    # = -t * sum_k (w_k/S) * d(w_k)/dc ... but w_k = exp(-v_k/t), d(w_k)/dc = -(w_k/t)*dv_k/dc
    # So d softmin/dc = -t * sum_k (w_k/S) * (-(w_k/t)*dv_k/dc) ... this is getting complex.
    # Simpler: softmin = -t*log(sum(exp(-v_k/t)))
    # d softmin/dc = -t * [sum_k exp(-v_k/t) * (-1/t) * dv_k/dc] / sum_k exp(-v_k/t)
    # = sum_k w_k * dv_k/dc / sum_k w_k
    # So grad_c = sum_k w_k * dv_k/dc / S
    # For pairs: dv_pair/dc[i] = -0.5*unit_vec, dv_pair/dc[j] = +0.5*unit_vec
    # For walls: dv_wall/dc[i,k] = -sign(c[i,k])
    
    # The above pair_grad and wall contributions already compute sum_k w_k * dv_k/dc
    # We need to divide by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c


def _lloyd_step(c, deadline, steps=200, damping=0.5):
    """Voronoi-based Lloyd's algorithm: move each center toward the centroid
    of its Voronoi cell, clipped to stay inside the cube."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    
    for it in range(steps):
        if time.time() > deadline:
            break
        
        try:
            tri = Delaunay(c)
        except Exception:
            # Fallback: simple repulsion step
            d = _pair_dist_matrix(c)
            inv = 1.0 / np.maximum(d, 1e-6) ** 2
            diff = c[:, None, :] - c[None, :, :]
            F = np.sum(diff * inv[:, :, None], axis=1)
            c = np.clip(c + 0.01 * F, -0.49, 0.49)
            continue
        
        simplices = tri.simplices
        
        cell_centroids = np.zeros((N, 3))
        cell_counts = np.zeros(N)
        
        def _circumcenter(v0, v1, v2, v3):
            A = np.array([
                v1 - v0,
                v2 - v0,
                v3 - v0
            ])
            b = np.array([
                np.dot(v1, v1) - np.dot(v0, v0),
                np.dot(v2, v2) - np.dot(v0, v0),
                np.dot(v3, v3) - np.dot(v0, v0)
            ])
            try:
                x = np.linalg.solve(2.0 * A, b)
                return x
            except np.linalg.LinAlgError:
                return np.mean([v0, v1, v2, v3])
        
        for idx in range(len(simplices)):
            si = simplices[idx]
            v0, v1, v2, v3 = c[si[0]], c[si[1]], c[si[2]], c[si[3]]
            cc = _circumcenter(v0, v1, v2, v3)
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
        steps = max(50, int(300 * damp))
        c = _lloyd_step(c, deadline, steps=steps, damping=damp)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c.copy()
    
    return best, best_r


def _smooth_obj_and_grad(x, t):
    """Objective to MINIMIZE: negative smooth radius. Returns (value, gradient)."""
    c = x.reshape(N, 3)
    r, grad_r = _softmin_radius_and_grad(c, t)
    # Boundary penalty
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    pen_grad = np.zeros_like(c)
    over = np.maximum(0.0, np.abs(c) - 0.5 + 1e-6)
    if np.any(over > 0):
        signs = np.sign(c)
        signs[signs == 0] = 1.0
        pen_grad = 2.0 * over * signs
    val = -r + 1e3 * pen
    grad = -grad_r + 1e3 * pen_grad
    return val, grad.flatten()


def _smooth_polish(c, deadline):
    """Polish with smooth softmin max-min via L-BFGS-B, annealing temperature."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)
    
    for t in (0.05, 0.02, 0.008, 0.003, 0.001, 0.0005, 0.0001):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                lambda x: _smooth_obj_and_grad(x, t)[0],
                x0,
                jac=lambda x: _smooth_obj_and_grad(x, t)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-11,
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
        unit_vecs = diff / np.maximum(dists, 1e-12)[:, None]
        
        F = np.zeros_like(c)
        for idx in range(_NUM_PAIRS):
            i, j = _PAIR_IDX[idx]
            F[i] += inv[idx] * unit_vecs[idx]
            F[j] -= inv[idx] * unit_vecs[idx]
        
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


def _metropolis_mc(c, deadline, iters=5000):
    """Metropolis Monte Carlo: random displacements with acceptance based on radius."""
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


def _local_search(c, deadline, iters=4000):
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
        
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_idx = np.argmin(dists)
        min_pair_dist = dists[min_pair_idx]
        
        wall = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        min_wall_val = wall[min_wall_idx]
        
        improved = False
        
        if min_pair_dist / 2.0 < min_wall_val:
            i, j = _PAIR_IDX[min_pair_idx]
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
        c_an = _repulsion_anneal(s, deadline, steps=2000)
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

        # Stage D: Metropolis MC to escape local minima
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(c_cur, deadline, iters=3000)
            if r_mc > r_cur:
                r_cur = r_mc
                c_cur = c_mc
        
        # Stage E: local search
        if time.time() < deadline:
            c_ls, r_ls = _local_search(c_cur, deadline, iters=2000)
            if r_ls > r_cur:
                r_cur = r_ls
                c_cur = c_ls

        if r_cur > best_r:
            best_r = r_cur
            best_c = c_cur.copy()

    # Final extra polish on the global best if time remains
    if best_c is not None and time.time() < deadline:
        def _final_polish(c, deadline):
            c = c.copy()
            c = np.clip(c, -0.499, 0.499)
            best = c.copy()
            best_r = _compute_radius(c)
            x0 = c.flatten()
            bounds = [(-0.499, 0.499)] * (N * 3)
            
            for t in (0.0001, 0.00005, 0.00002, 0.00001, 0.000005):
                if time.time() > deadline:
                    break
                try:
                    res = minimize(
                        lambda x: _smooth_obj_and_grad(x, t)[0],
                        x0,
                        jac=lambda x: _smooth_obj_and_grad(x, t)[1],
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12,
                                 'maxls': 100},
                    )
                    c = np.clip(res.x.reshape(N, 3), -0.499, 0.499)
                    r = _compute_radius(c)
                    if r > best_r:
                        best_r = r
                        best = c.copy()
                    x0 = c.flatten()
                except Exception:
                    pass
            return best, best_r
        
        c_pol, r_pol = _final_polish(best_c, deadline)
        if r_pol > best_r:
            best_r = r_pol
            best_c = c_pol
        
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(best_c, deadline, iters=2000)
            if r_mc > best_r:
                best_r = r_mc
                best_c = c_mc
        
        if time.time() < deadline:
            c_ls, r_ls = _local_search(best_c, deadline, iters=1500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

# EVOLVE-BLOCK-END

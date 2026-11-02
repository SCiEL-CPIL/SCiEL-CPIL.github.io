"""Improved packing: hybrid SA + L-BFGS polish for 100 equal spheres in unit cube."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
HALF = 0.5


def _compute_radius(c):
    """Return the max common radius for centers c (N,3) inside [-0.5,0.5]^3."""
    c = np.asarray(c)
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return 0.0
    # Wall distances
    wall = np.min(HALF - np.abs(c), axis=1)
    r_wall = float(np.min(wall))
    if r_wall <= 0:
        return 0.0
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    r_pair = float(np.min(d)) / 2.0
    return min(r_wall, r_pair)


def _objective(c_flat):
    """Minimize negative radius (i.e., maximize radius). c_flat is (3N,)."""
    c = c_flat.reshape(N, 3)
    r = _compute_radius(c)
    return -r


def _bounds():
    """Bounds for L-BFGS-B: each coordinate in [-0.5, 0.5]."""
    return [(-HALF, HALF)] * (3 * N)


def _sa_step(c, T, alpha=0.02):
    """One simulated annealing step: perturb one random point, accept if better or with prob."""
    i = np.random.randint(N)
    c_new = c.copy()
    # Random perturbation
    c_new[i] += np.random.normal(0, alpha, size=3)
    # Clamp to cube
    c_new[i] = np.clip(c_new[i], -HALF + 1e-9, HALF - 1e-9)
    
    r_old = _compute_radius(c)
    r_new = _compute_radius(c_new)
    
    if r_new >= r_old:
        return c_new, r_new
    else:
        delta = r_new - r_old
        if delta < 0:
            prob = np.exp(delta / max(T, 1e-12))
            if np.random.random() < prob:
                return c_new, r_new
        return c, r_old


def construct():
    t_start = time.time()
    deadline = t_start + 120.0  # 120s limit, use 150s wall-clock but be safe
    
    np.random.seed(42)
    
    # Initial: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Phase 1: L-BFGS polish from initial grid
    c_flat = c.flatten()
    res = minimize(_objective, c_flat, method='L-BFGS-B', bounds=_bounds(),
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    c = res.x.reshape(N, 3)
    r = _compute_radius(c)
    if r > best_r:
        best_r = r
        best_c = c.copy()
    
    if time.time() > deadline - 30:
        return best_c
    
    # Phase 2: Hybrid SA + periodic L-BFGS polish
    T_init = 0.01
    T_min = 0.0001
    cool_rate = 0.999
    alpha = 0.03
    
    T = T_init
    sa_iter = 0
    max_sa_iter = 200000
    polish_every = 500
    
    while sa_iter < max_sa_iter and time.time() < deadline - 15:
        # Batch of SA steps
        batch_size = 100
        for _ in range(batch_size):
            if time.time() > deadline - 15:
                break
            c, r = _sa_step(c, T, alpha)
            sa_iter += 1
        
        T = max(T * cool_rate, T_min)
        
        # Periodic L-BFGS polish
        if sa_iter % polish_every == 0:
            c_flat = c.flatten()
            try:
                res = minimize(_objective, c_flat, method='L-BFGS-B', bounds=_bounds(),
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                c = res.x.reshape(N, 3)
            except Exception:
                pass
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    # Final polish on best found
    c_flat = best_c.flatten()
    try:
        res = minimize(_objective, c_flat, method='L-BFGS-B', bounds=_bounds(),
                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        c = res.x.reshape(N, 3)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    except Exception:
        pass
    
    return best_c

"""Suite task: pack 100 equal spheres in the unit cube [-1/2,1/2]^3.
Entry point construct() -> (N,3) array of centres.
Score = min( min_{i<j} |c_i-c_j|/2 , min_i min_k (1/2 - |c_ik|) ).

Design:
  1) Vectorized distance computation for speed.
  2) Simulated annealing for global exploration (different search family).
  3) Analytical gradient polishing (L-BFGS-B) for high-precision local refinement.
  4) Progressive radius-target hard-violation energy to push past local minima.
  5) Wall-clock deadline guard (140s safety margin vs 150s kill).
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
HALF = 0.5
_DEADLINE = 140.0  # safety margin vs 150 s wall-clock kill
_rng = np.random.default_rng(42)


def _compute_radius(c):
    """Compute the maximum common radius for given centers."""
    c = np.asarray(c, dtype=float)
    n = c.shape[0]
    if n < 2:
        return float(np.min(0.5 - np.abs(c)))
    
    # Vectorized pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = float(np.min(dists))
    
    min_wall = float(np.min(0.5 - np.abs(c)))
    return min(0.5 * min_pair, min_wall)


def _energy_soft(x, K, beta):
    """
    E = -K * soft_min(pair_dists) + beta * sum(wall_pen^2)
    Vectorized implementation.
    """
    c = x.reshape(N, 3)
    
    # --- Wall penalty ---
    pen_out = np.maximum(np.abs(c) - HALF, 0.0)
    e_wall = beta * float(np.dot(pen_out.ravel(), pen_out.ravel()))
    g_wall = 2.0 * beta * pen_out * np.sign(c)
    
    # --- Repulsion: soft-min of pairwise distances ---
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Get unique upper-triangular pairs
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    
    dmin = float(np.min(d))
    if dmin < 1e-10:
        dmin = 1e-10
    
    # soft-min = -logsumexp(-beta * d) / beta
    scaled = -beta * d
    m = np.max(scaled)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled - m)))
    e_rep = K * log_sum_exp / beta
    
    # Gradient of soft-min w.r.t. d: w_k = exp(-beta*d_k) / sum(exp(-beta*d_j))
    w = np.exp(scaled - m)
    w /= np.sum(w)
    
    # Build gradient: for each pair (i,j), d(d_k)/d(c_j) = (c_j-c_i)/d_k
    g_rep = np.zeros_like(c)
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        wk = w[idx]
        d_k = d[idx]
        if d_k < 1e-10:
            continue
        unit_vec = (c[j] - c[i]) / d_k
        g_rep[j] += wk * unit_vec
        g_rep[i] -= wk * unit_vec
    
    # grad_obj = -K * d(soft_min)/d(c) + g_wall
    g_obj = -K * g_rep + g_wall
    
    return e_rep + e_wall, g_obj.ravel()


def _energy_hard(x, r_target, beta_wall=50.0):
    """
    Hard-violation energy: E = 0.5*sum max(0, 2r - d_ij)^2 + 0.5*beta_wall*sum max(0, r - clearance)^2
    Vectorized implementation.
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    
    e = 0.0
    g = np.zeros_like(c)
    
    # Pair violations
    viol = np.maximum(0.0, 2.0 * r_target - d)
    mask = viol > 0
    if np.any(mask):
        e += 0.5 * float(np.sum(viol[mask] ** 2))
        for idx in range(len(iu[0])):
            if not mask[idx]:
                continue
            i, j = iu[0][idx], iu[1][idx]
            d_k = d[idx]
            if d_k < 1e-10:
                continue
            v = viol[idx]
            unit_vec = (c[j] - c[i]) / d_k
            g[j] += v * unit_vec
            g[i] -= v * unit_vec
    
    # Wall violations
    clearance = HALF - np.abs(c)
    wviol = np.maximum(0.0, r_target - clearance)
    if np.any(wviol > 0):
        e += 0.5 * beta_wall * float(np.sum(wviol ** 2))
        g += beta_wall * wviol * np.sign(c)
    
    return e, g.ravel()


def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N].copy()


def _init_random(seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.49, 0.49, size=(N, 3))


def _init_jittered():
    return _init_grid() + _rng.normal(0, 0.012, size=(N, 3))


def _optimize_soft(c, K, beta, maxiter=400, ftol=1e-12, gtol=1e-10):
    """L-BFGS-B on soft-min energy."""
    x0 = c.ravel()
    try:
        res = minimize(_energy_soft, x0, args=(K, beta), method='L-BFGS-B',
                       jac=True, options={'maxiter': maxiter, 'maxls': 30,
                                          'ftol': ftol, 'gtol': gtol})
        cand = np.clip(res.x.reshape(N, 3), -HALF, HALF)
        return cand
    except Exception:
        return c


def _optimize_hard(c, r_target, maxiter=200, ftol=1e-12, gtol=1e-10):
    """L-BFGS-B on hard-violation energy."""
    x0 = c.ravel()
    try:
        res = minimize(_energy_hard, x0, args=(r_target, 50.0), method='L-BFGS-B',
                       jac=True, options={'maxiter': maxiter, 'maxls': 30,
                                          'ftol': ftol, 'gtol': gtol})
        cand = np.clip(res.x.reshape(N, 3), -HALF, HALF)
        return cand
    except Exception:
        return c


def _simulated_annealing(c0, r_target, t_start=0.05, t_end=0.001, n_iter=5000, deadline=None):
    """
    Simulated annealing to find a configuration that minimizes violations
    for a target radius r_target.
    """
    c = c0.copy()
    current_e, _ = _energy_hard(c.ravel(), r_target)
    best_c = c.copy()
    best_e = current_e
    
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / n_iter)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        t *= cooling
        
        # Generate a random perturbation
        idx = _rng.integers(0, N)
        delta = _rng.normal(0, t * 2.0, size=3)
        new_c = c.copy()
        new_c[idx] = np.clip(new_c[idx] + delta, -HALF, HALF)
        
        new_e, _ = _energy_hard(new_c.ravel(), r_target)
        
        # Accept if better, or with probability exp(-delta_e / t)
        delta_e = new_e - current_e
        if delta_e < 0 or _rng.random() < np.exp(-delta_e / max(t, 1e-10)):
            c = new_c
            current_e = new_e
            if current_e < best_e:
                best_c = c.copy()
                best_e = current_e
    
    return best_c


def _progressive_radius(c, r_start, r_end, n_steps=20, deadline=None):
    """Progressively increase target radius and minimize hard-violation energy."""
    current = c.copy()
    r_values = np.linspace(r_start, r_end, n_steps)
    for r in r_values:
        if deadline is not None and time.time() > deadline:
            break
        current = _optimize_hard(current, r, maxiter=100, ftol=1e-12, gtol=1e-10)
    return current


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE

    best = _init_grid()
    best_r = _compute_radius(best)

    inits = [_init_grid(), _init_random(1), _init_jittered(), _init_random(2)]
    K, beta = 1.0, 80.0

    # Phase 1: soft-min optimization from multiple starts
    for it, c in enumerate(inits):
        if time.time() > deadline:
            break
        c = np.clip(c, -0.49, 0.49)
        cand = _optimize_soft(c, K, beta, maxiter=400)
        r = _compute_radius(cand)
        if r > best_r + 1e-12:
            best, best_r = cand, r

    # Phase 2: Simulated annealing for global exploration
    if time.time() < deadline - 40.0:
        # Try SA from multiple starting points
        sa_starts = [best, _init_random(3), _init_random(4)]
        for sa_c in sa_starts:
            if time.time() > deadline - 30.0:
                break
            # Use a slightly conservative target to allow SA to converge
            r_target = best_r * 0.98
            sa_result = _simulated_annealing(sa_c, r_target, t_start=0.03, t_end=0.0005, 
                                            n_iter=3000, deadline=deadline - 30.0)
            r_sa = _compute_radius(sa_result)
            if r_sa > best_r + 1e-12:
                best, best_r = sa_result, r_sa

    # Phase 3: progressive radius refinement from best
    if time.time() < deadline - 25.0:
        r_start = best_r * 0.95
        r_end = best_r * 1.05
        n_steps = 15
        refined = _progressive_radius(best, r_start, r_end, n_steps=n_steps, deadline=deadline - 10.0)
        r_ref = _compute_radius(refined)
        if r_ref > best_r + 1e-12:
            best, best_r = refined, r_ref

    # Phase 4: high-precision polish
    if time.time() < deadline - 10.0:
        # Soft-min polish with high beta
        cand = _optimize_soft(best, K=1.0, beta=150.0, maxiter=2000, ftol=1e-15, gtol=1e-13)
        r = _compute_radius(cand)
        if r > best_r + 1e-14:
            best, best_r = cand, r

        # Hard-violation polish at current radius
        if time.time() < deadline - 5.0:
            r_target = best_r * 1.001
            cand2 = _optimize_hard(best, r_target, maxiter=1000, ftol=1e-15, gtol=1e-13)
            r2 = _compute_radius(cand2)
            if r2 > best_r + 1e-14:
                best, best_r = cand2, r2

    # Final validity guard
    best = np.clip(best, -0.4999999, 0.4999999)
    if not np.all(np.isfinite(best)):
        best = _init_grid()
    return best

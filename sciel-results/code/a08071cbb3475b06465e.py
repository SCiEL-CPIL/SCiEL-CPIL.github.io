"""Suite task: pack 100 equal spheres in the unit cube [-1/2,1/2]^3.
Entry point construct() -> (N,3) array of centres.
Score = min( min_{i<j} |c_i-c_j|/2 , min_i min_k (1/2 - |c_ik|) ).
Design:
  1) Multi-start L-BFGS-B on a repulsion-energy surrogate (soft-min of pairwise
     distances + quadratic wall penalty) with analytical gradients.
  2) Progressive radius-target hard-violation energy refinement to push past
     shallow local minima (inspired by the 1.095 solution).
  3) High-precision polish with very tight tolerances.
All under an explicit wall-clock deadline guard (150 s kill, 140 s safety).
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
HALF = 0.5
_DEADLINE = 140.0  # safety margin vs 150 s wall-clock kill
_rng = np.random.default_rng(12345)


def _radius(c):
    """Common radius: min of half min pairwise distance and min wall clearance."""
    c = np.asarray(c, dtype=float)
    n = c.shape[0]
    d = 1e30
    for i in range(n - 1):
        diff = c[i + 1:] - c[i]
        d = min(d, float(np.min(np.einsum('ij,ij->i', diff, diff))))
    if d > 1e29:
        return float(np.min(0.5 - np.abs(c)))
    return min(0.5 * np.sqrt(d), float(np.min(0.5 - np.abs(c))))


def _energy_soft(x, K, beta):
    """
    E = -K * soft_min(pair_dists) + beta * sum(wall_pen^2)
    soft_min via log-sum-exp for smoothness.
    """
    n = x.size
    c = x.reshape(N, 3)
    # --- wall penalty ---
    pen = np.maximum(HALF - np.abs(c), 0.0)  # clearance, 0 if inside
    # Actually we want to penalize going OUTSIDE: penalty when |c| > HALF
    pen_out = np.maximum(np.abs(c) - HALF, 0.0)
    e_wall = beta * float(np.dot(pen_out.ravel(), pen_out.ravel()))
    g_wall = 2.0 * beta * pen_out * np.sign(c)

    # --- repulsion: soft-min of pairwise distances ---
    d = np.zeros(N * (N - 1) // 2)
    g = np.zeros(n)
    idx = 0
    for i in range(N - 1):
        diff = c[i + 1:] - c[i]
        dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        d[idx:idx + len(dist)] = dist
        g[i + 1:] += diff / dist
        idx += len(dist)

    dmin = float(np.min(d))
    # soft-min = -logsumexp(-beta * d) / beta
    scaled = -beta * d
    m = np.max(scaled)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled - m)))
    e_rep = K * log_sum_exp / beta  # = -K * soft_min

    # gradient of soft-min w.r.t. d: w_k = exp(-beta*d_k) / sum(exp(-beta*d_j))
    w = np.exp(scaled - m)
    w /= np.sum(w)

    # d(soft_min)/d(c) = sum_k w_k * d(d_k)/d(c)
    # d(d_k)/d(c) for pair (i,j): (c_i - c_j)/d_k for c_i, (c_j - c_i)/d_k for c_j
    # We built g such that g accumulates diff/dist for c[i+1:] - c[i]
    # For soft-min gradient: sum_k w_k * (unit vector for pair k)
    # Rebuild: for each pair, contribution is w_k * (c_j - c_i)/d_k to c_j, etc.
    g_rep = np.zeros(n)
    idx = 0
    for i in range(N - 1):
        diff = c[i + 1:] - c[i]
        dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        wk = w[idx:idx + len(dist)]
        # d_k = |c_j - c_i|, d(d_k)/d(c_j) = (c_j - c_i)/d_k = diff/dist
        # d(d_k)/d(c_i) = -diff/dist
        # soft_min = sum w_k d_k, so d(soft_min)/d(c_j) += w_k * diff/dist
        # d(soft_min)/d(c_i) += -w_k * diff/dist
        contrib = wk[:, None] * (diff / dist)
        g_rep[i + 1:] += contrib
        g_rep[i] -= float(np.sum(contrib, axis=0))
        idx += len(dist)

    e_rep = -K * (-log_sum_exp / beta)  # = K * log_sum_exp / beta... wait
    # Let me redo: we want to MAXIMIZE soft_min, so minimize -soft_min
    # soft_min = -logsumexp(-beta*d)/beta
    # -soft_min = logsumexp(-beta*d)/beta
    # d(-soft_min)/d(c) = (1/beta) * d(logsumexp(-beta*d))/d(c)
    # = (1/beta) * sum_k [beta * exp(-beta*d_k)/sum(exp(-beta*d_j))] * (-d(d_k)/d(c))
    # = -sum_k w_k * d(d_k)/d(c)  where w_k = exp(-beta*d_k)/sum(...)
    # So gradient of objective = -K * (-sum_k w_k * d(d_k)/d(c)) = K * sum_k w_k * d(d_k)/d(c)
    # But we want to MINIMIZE, and the objective is -soft_min + wall
    # So grad = -K * d(soft_min)/d(c) + g_wall = -K * (sum w_k d(d_k)/d(c)) + g_wall
    # d(soft_min)/d(c) = sum_k w_k * d(d_k)/d(c)
    # For pair (i,j): d(d_k)/d(c_j) = (c_j-c_i)/d_k, d(d_k)/d(c_i) = -(c_j-c_i)/d_k
    # So d(soft_min)/d(c) has for c_j: +w_k*(c_j-c_i)/d_k, for c_i: -w_k*(c_j-c_i)/d_k
    # grad_obj = -K * that + g_wall
    g_obj = -K * g_rep + g_wall

    return e_rep + e_wall, g_obj.ravel()


def _energy_hard(x, r_target, beta_wall=50.0):
    """
    Hard-violation energy: E = 0.5*sum max(0, 2r - d_ij)^2 + 0.5*beta_wall*sum max(0, r - clearance)^2
    """
    c = x.reshape(N, 3)
    e = 0.0
    g = np.zeros_like(c)

    # Pair violations
    for i in range(N - 1):
        diff = c[i + 1:] - c[i]
        dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        viol = np.maximum(0.0, 2.0 * r_target - dist)
        mask = viol > 0
        if np.any(mask):
            e += 0.5 * float(np.sum(viol[mask] ** 2))
            # d(E)/d(c_j) = viol * (c_j - c_i)/dist, d(E)/d(c_i) = -viol*(c_j-c_i)/dist
            contrib = viol[mask, None] * (diff[mask] / dist[mask, None])
            g[i + 1:] += contrib
            g[i] -= float(np.sum(contrib, axis=0))

    # Wall violations
    clearance = HALF - np.abs(c)
    wviol = np.maximum(0.0, r_target - clearance)
    if np.any(wviol > 0):
        e += 0.5 * beta_wall * float(np.sum(wviol ** 2))
        # d(E)/d(c_ik) = beta_wall * wviol * sign(c_ik)
        g += beta_wall * wviol * np.sign(c)

    return e, g.ravel()


def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N].copy()


def _init_random():
    return _rng.uniform(-0.49, 0.49, size=(N, 3))


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
    best_r = _radius(best)

    inits = [_init_grid(), _init_random(), _init_jittered(), _init_random()]
    K, beta = 1.0, 80.0

    # Phase 1: soft-min optimization from multiple starts
    for it, c in enumerate(inits):
        if time.time() > deadline:
            break
        c = np.clip(c, -0.49, 0.49)
        cand = _optimize_soft(c, K, beta, maxiter=400)
        r = _radius(cand)
        if r > best_r + 1e-12:
            best, best_r = cand, r

    # Phase 2: progressive radius refinement from best
    if time.time() < deadline - 30.0:
        r_start = best_r * 0.95
        r_end = best_r * 1.05
        n_steps = 15
        refined = _progressive_radius(best, r_start, r_end, n_steps=n_steps, deadline=deadline - 10.0)
        r_ref = _radius(refined)
        if r_ref > best_r + 1e-12:
            best, best_r = refined, r_ref

    # Phase 3: high-precision polish
    if time.time() < deadline - 10.0:
        # Soft-min polish with high beta
        cand = _optimize_soft(best, K=1.0, beta=150.0, maxiter=2000, ftol=1e-15, gtol=1e-13)
        r = _radius(cand)
        if r > best_r + 1e-14:
            best, best_r = cand, r

        # Hard-violation polish at current radius
        if time.time() < deadline - 5.0:
            r_target = best_r * 1.001
            cand2 = _optimize_hard(best, r_target, maxiter=1000, ftol=1e-15, gtol=1e-13)
            r2 = _radius(cand2)
            if r2 > best_r + 1e-14:
                best, best_r = cand2, r2

    # Final validity guard
    best = np.clip(best, -0.4999999, 0.4999999)
    if not np.all(np.isfinite(best)):
        best = _init_grid()
    return best

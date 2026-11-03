"""Combined packing program: SLSQP linear-epigraph primary + multi-stage refinement."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

DEADLINE = 148.0  # safety margin before the 150s kill


def _true_score(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    
    # Wall distance
    r = np.min(0.5 - np.abs(c))
    
    # Pairwise distance
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    
    return r


def _grid_init():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _hex_init():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            pts.append(((j + 0.5) / m - 0.5, (i + 0.5) / m - 0.5))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _relax_hard_disk(c, R, iters=2000, eta0=0.1, eta1=0.01):
    """
    Hard-disk relaxation: push overlapping circles apart and keep them
    inside the square. Vectorized.
    """
    c = c.copy()
    c = np.clip(c, -0.5 + R, 0.5 - R)
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, 1e-12)
        
        # Force magnitude: push apart if d < 2R
        mag = np.maximum(2 * R - d, 0.0) / np.maximum(d, 1e-12)
        np.fill_diagonal(mag, 0.0)
        
        f = np.sum(mag[:, :, None] * diff, axis=1)
        
        # Wall forces
        # Push inward if outside [-0.5+R, 0.5-R]
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        
        c += eta * f
        c = np.clip(c, -0.5 + R, 0.5 - R)
    return c


def _soft_min_radius(c, t):
    """
    Differentiable surrogate: soft-min of all pairwise half-distances
    and wall distances.
    """
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()

    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()

    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    # Avoid overflow/underflow in exp
    s = np.exp(-(all_terms - m) / t)
    soft = m - t * np.log(s.sum())
    return soft


def _soft_min_grad(x, t):
    """
    Gradient of negative soft-min radius w.r.t. x (flattened centres).
    We minimize -soft_min_radius, so we return -grad(soft_min).
    """
    c = x.reshape(N, 2)
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()

    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()

    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    exps = np.exp(-(all_terms - m) / t)
    Z = exps.sum()

    grad = np.zeros_like(c)

    # Wall gradients: term = 0.5 - |c_ik|
    # d(soft)/dc_ik = (exps[i] / Z) * d(term)/dc_ik
    # d(term)/dc_ik = -sign(c_ik)
    sign_c = np.sign(c)
    # Wall part of exps is the first 2N elements
    w_exps = exps[:2 * N].reshape(N, 2)
    grad += (w_exps / Z) * (-sign_c)

    # Pair gradients: term_ij = ||c_i - c_j|| / 2
    # d(soft)/dc_i = sum_j (exps_ij / Z) * (c_i - c_j) / (2 * ||c_i - c_j||)
    dist = d.copy()
    np.fill_diagonal(dist, 1.0) # avoid div by zero
    
    # We need to sum over j for each i
    # exps for pairs is the remaining part
    p_exps = exps[2 * N:].reshape(N, N)
    
    # Compute weighted sum of (c_i - c_j) / dist_ij
    # p_exps[i,j] corresponds to term (i,j)
    # Note: cdist is symmetric, so d[i,j] == d[j,i].
    # The soft min includes both (i,j) and (j,i) in the sum?
    # Yes, `pair_flat` includes all i != j.
    # So we sum over all j != i.
    
    for i in range(N):
        # Mask out i itself
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        weights = p_exps[i, mask] / Z
        diffs = c[i] - c[mask]
        dists = dist[i, mask]
        
        # Vectorized sum: sum_j w_j * (c_i - c_j) / d_ij
        # weights shape (N-1,), diffs shape (N-1, 2), dists shape (N-1,)
        contrib = np.sum(weights[:, None] * diffs / dists[:, None], axis=0)
        grad[i] += contrib * 0.5 # because term is d/2

    # We are minimizing -soft_min, so return -grad
    return -grad.ravel()


def _slsqp_epigraph_polish(c, t0, time_budget=30.0):
    """
    SLSQP epigraph formulation: maximize r subject to
    - pairwise distances >= 2r
    - wall distances >= r
    
    Variables: x = [c_1x, c_1y, ..., c_Nx, c_Ny, r]
    Objective: minimize -r
    """
    c = np.array(c, dtype=float, copy=True)
    r_init = _true_score(c)
    if r_init <= 0:
        return c, 0.0

    def objective(x):
        return -x[-1]  # maximize r = x[-1]

    def objective_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    constraints = []
    
    # Pairwise constraints: ||c_i - c_j|| - 2r >= 0
    for i in range(N):
        for j in range(i + 1, N):
            def pair_con(x, i=i, j=j):
                ci = x[2 * i:2 * i + 2]
                cj = x[2 * j:2 * j + 2]
                r = x[-1]
                d = np.sqrt(np.sum((ci - cj) ** 2))
                return d - 2 * r
            
            def pair_grad(x, i=i, j=j):
                g = np.zeros_like(x)
                ci = x[2 * i:2 * i + 2]
                cj = x[2 * j:2 * j + 2]
                diff = ci - cj
                d = np.sqrt(np.sum(diff ** 2))
                if d > 1e-12:
                    g[2 * i:2 * i + 2] = diff / d
                    g[2 * j:2 * j + 2] = -diff / d
                g[-1] = -2.0
                return g
            
            constraints.append({'type': 'ineq', 'fun': pair_con, 'jac': pair_grad})

    # Wall constraints: 0.5 - |c_ik| - r >= 0
    for i in range(N):
        for k in range(2):
            def wall_con(x, i=i, k=k):
                c_ik = x[2 * i + k]
                r = x[-1]
                return 0.5 - np.abs(c_ik) - r
            
            def wall_grad(x, i=i, k=k):
                g = np.zeros_like(x)
                c_ik = x[2 * i + k]
                g[2 * i + k] = -np.sign(c_ik)
                g[-1] = -1.0
                return g
            
            constraints.append({'type': 'ineq', 'fun': wall_con, 'jac': wall_grad})

    x0 = np.concatenate([c.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(1e-6, 0.5)]
    
    best_x = x0.copy()
    best_r = r_init
    
    # Run SLSQP with multiple maxiter settings to allow deeper optimization
    maxiters_list = [200, 500, 1000, 2000]
    
    for maxiter in maxiters_list:
        if (time.time() - t0) > t0 + time_budget:
            break
        try:
            res = minimize(
                objective,
                x0,
                jac=objective_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                r_new = res.x[-1]
                c_new = res.x[:2 * N].reshape(N, 2)
                # Verify the result
                r_verified = _true_score(c_new)
                if r_verified > best_r:
                    best_r = r_verified
                    best_x = res.x.copy()
                    x0 = res.x.copy()
        except Exception:
            pass
    
    return best_x[:2 * N].reshape(N, 2), best_r


def construct():
    start_time = time.time()
    
    # 1. Initialization and Hard-Disk Relaxation
    # Try a few different initializations and relax them
    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(5):
        inits.append(rng.uniform(-0.45, 0.45, (N, 2)))
        
    best_c = None
    best_r = 0.0
    
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 40:
            break
            
        # Estimate a starting radius
        r_start = 0.05
        # Relax to find a feasible radius
        # We can try to grow the radius
        cur_c = c0
        cur_r = r_start
        for _ in range(10):
            cur_c = _relax_hard_disk(cur_c, cur_r, iters=500, eta0=0.1, eta1=0.01)
            s = _true_score(cur_c)
            if s >= cur_r - 1e-9:
                cur_r = cur_r * 1.1
            else:
                break
        
        if s > best_r:
            best_r = s
            best_c = cur_c

    if best_c is None:
        # Fallback
        best_c = _grid_init()
        best_r = _true_score(best_c)

    # 2. Soft-Min Polish (L-BFGS-B)
    # This smooths the layout and often increases the radius significantly
    # before the exact SLSQP step.
    if time.time() - start_time < DEADLINE - 30:
        # Try a few different temperature parameters
        for t in [0.01, 0.005, 0.001]:
            if time.time() - start_time > DEADLINE - 30:
                break
            try:
                res = minimize(
                    lambda x: -_soft_min_radius(x.reshape(N, 2), t),
                    best_c.ravel(),
                    jac=lambda x: _soft_min_grad(x, t),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
                )
                c_new = res.x.reshape(N, 2)
                s = _true_score(c_new)
                if s > best_r:
                    best_r = s
                    best_c = c_new
            except Exception:
                pass

    # 3. SLSQP Epigraph Polish
    # This is the most accurate and powerful step.
    # Allocate the remaining time to this.
    remaining_time = DEADLINE - 5 - (time.time() - start_time)
    if remaining_time > 5:
        best_c, best_r = _slsqp_epigraph_polish(best_c, time.time(), time_budget=remaining_time)

    # Final check
    final_r = _true_score(best_c)
    if final_r > best_r:
        best_r = final_r

    return best_c
# EVOLVE-BLOCK-END

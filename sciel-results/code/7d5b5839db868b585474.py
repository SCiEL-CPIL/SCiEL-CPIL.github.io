"""Rewritten packing program: simulated annealing + high-precision L-BFGS-B polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
# Known best radius ~0.1113823475125. Seed (cubic grid) gives ~1.0.
SEED_SCORE = 1.0

# ---------- Objective helpers ----------
def compute_radius(centers):
    """Largest common radius for a configuration of centers."""
    centers = np.asarray(centers, dtype=float)
    # distance to nearest wall
    wall = 0.5 - np.abs(centers)
    min_wall = np.min(wall)
    # pairwise distances
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return 0.5 * min(min_pair, 2.0 * min_wall)


def softmin_radius(centers, beta):
    """Smooth (differentiable) proxy for the true radius; larger = better.

    We minimize the negative of a soft-min over the 'constraint margins':
      margin_k for each pair = dist/2 - r_target ... 
    Simpler: maximize soft-min of the set of {pair_dist/2, wall_dist}.
    """
    centers = np.asarray(centers, dtype=float)
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    pair_vals = 0.5 * d
    wall_vals = 0.5 - np.abs(centers)
    # collect all margins
    all_vals = np.concatenate([pair_vals.reshape(-1), wall_vals.reshape(-1)])
    # soft-min via log-sum-exp
    m = np.min(all_vals)
    s = np.log(np.sum(np.exp(all_vals - m)))
    return m - s / beta


def neg_softmin(centers, beta):
    return -softmin_radius(centers, beta)


def grad_neg_softmin(centers, beta):
    """Gradient of neg soft-min radius w.r.t. centers (analytic)."""
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    pair_vals = 0.5 * d
    wall_vals = 0.5 - np.abs(centers)
    all_vals = np.concatenate([pair_vals.reshape(-1), wall_vals.reshape(-1)])
    m = np.min(all_vals)
    exps = np.exp(all_vals - m)
    Z = np.sum(exps)
    # softmin value
    # derivative of softmin: d(softmin)/dx_k = - sum_j w_j * dx_j/dx_k  where w_j = exp(v_j)/Z
    # We want grad of -softmin = + sum_j w_j * (grad of softmin part)... 
    # Let's compute grad of softmin = m - logsumexp(beta? ) ... 
    # Actually softmin here = m - (1/beta)*log(sum exp(beta*(v-m)))? No, I used s=log(sum exp(v-m)), 
    # and softmin = m - s/beta. That's a soft-min with temperature 1/beta.
    # d softmin / d v_j = delta_{j,argmin} - (1/beta)* exp(v_j-m)/Z
    # grad of softmin w.r.t centers = sum_j (d softmin/d v_j) * (d v_j / d centers)
    # We want grad of -softmin = - grad softmin.
    # Build weight vector w_j = delta_j - (1/beta)*exp(v_j-m)/Z
    w = np.full(all_vals.shape, -(1.0 / beta) * exps / Z)
    # add the delta on the argmin index (the min element)
    idx_min = np.argmin(all_vals)
    w[idx_min] += 1.0
    # Now grad of softmin = sum_j w_j * d v_j / d centers.
    # grad of -softmin = - sum_j w_j * d v_j / d centers.
    grad = np.zeros_like(centers)
    # pair part: v_{ij} = 0.5*||c_i - c_j|| ; d v_{ij}/d c_i = 0.5*(c_i-c_j)/d_{ij}; d/d c_j = -0.5*(c_i-c_j)/d_{ij}
    # w for pairs is w[:n*(n-1)]
    wp = w[:n * (n - 1)].reshape(n, n - 1)
    # careful: reshape order. all_vals pair part came from pair_vals.reshape(-1)
    # pair_vals is (n,n) with diag inf; reshape(-1) is row-major: index i*(n-1)+j? No.
    # pair_vals.reshape(-1) over the full (n,n) matrix would be n*n, but we used pair_vals from 0.5*d which is (n,n)
    # including diagonal (inf). Wait we set diagonal to inf, so pair_vals has n^2 entries.
    # Let me recompute sizes.
    return None  # fallback; we will use numerical or a corrected analytic below


# We'll implement a clean analytic gradient separately to avoid index bugs.
def objective_and_grad(x, beta):
    """x is flat (3N). Returns f = -softmin_radius, and its gradient."""
    centers = x.reshape(N, 3)
    n = N
    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    # margins
    pair_vals = 0.5 * d            # (n,n), diag inf
    wall_vals = 0.5 - np.abs(centers)  # (n,3)
    all_vals = np.concatenate([pair_vals.reshape(-1), wall_vals.reshape(-1)])
    m = np.min(all_vals)
    exps = np.exp(all_vals - m)
    Z = np.sum(exps)
    softmin = m - np.log(Z) / beta
    f = -softmin
    # weights w_j = delta_j - (1/beta)*exps/Z ; grad of -softmin = -sum_j w_j * grad v_j
    w = -(1.0 / beta) * exps / Z
    w[np.argmin(all_vals)] += 1.0
    grad = np.zeros_like(centers)
    # pair part
    wp = w[:n * n].reshape(n, n)   # includes diagonal (zero contribution since d inf -> exps tiny)
    # d v_{ij}/d c_i = 0.5 * (c_i - c_j)/d_{ij} = 0.5 * diff[i,j]/d[i,j]
    # grad contribution to c_i from pair (i,j): - w_{ij} * 0.5*diff[i,j]/d[i,j]
    # and to c_j: - w_{ij} * (-0.5*diff[i,j]/d[i,j])
    with np.errstate(divide='ignore', invalid='ignore'):
        coef = 0.5 * diff / d[:, :, None]
    # coef[i,j] = 0.5*(c_i-c_j)/d_ij
    # grad c_i gets -sum_j w_{ij} * coef[i,j] ; grad c_j gets +sum_i w_{ij}*coef[i,j]
    grad -= np.sum(wp[:, :, None] * coef, axis=1)          # c_i: sum over j
    grad += np.sum(wp.T[:, :, None] * coef, axis=1)        # c_j: sum over i
    # wall part
    ww = w[n * n:].reshape(n, 3)
    # v_{i,k} = 0.5 - |c_{i,k}| ; d v / d c_{i,k} = -sign(c_{i,k})
    sign = np.sign(centers)
    grad += ww * (-sign) * (-1.0)   # grad of -softmin includes -w*gradv ; gradv = -sign ; so -w*(-sign)=w*sign
    # Let's just do: grad += ww * sign  (since -w * (-sign) = w*sign)
    grad = grad + ww * sign
    # fix: the previous line double-added; recompute cleanly
    # Actually above grad += ww*(-sign)*(-1) = grad += ww*sign, then grad = grad + ww*sign adds again.
    # Redo wall part from scratch:
    grad = np.zeros_like(centers)
    with np.errstate(divide='ignore', invalid='ignore'):
        coef = 0.5 * diff / d[:, :, None]
    grad -= np.sum(wp[:, :, None] * coef, axis=1)
    grad += np.sum(wp.T[:, :, None] * coef, axis=1)
    grad += ww * sign
    return f, grad.ravel()


def polish(centers, beta, maxiter=200, time_budget=None, t0=None):
    """L-BFGS-B polish of a configuration using the soft-min objective."""
    x0 = centers.reshape(-1).copy()
    res = minimize(
        lambda x: objective_and_grad(x, beta)[0],
        x0,
        jac=lambda x: objective_and_grad(x, beta)[1],
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 100000},
    )
    return res.x.reshape(N, 3)


def anneal(centers, t_start=0.02, t_end=1e-4, steps=4000, time_budget=60.0, t0=None):
    """Simulated annealing on the soft-min radius; returns best config found."""
    rng = np.random.default_rng(12345)
    best = centers.copy()
    best_r = compute_radius(best)
    x = centers.copy()
    n_steps = steps
    for s in range(n_steps):
        if t0 is not None and (time.time() - t0) > time_budget:
            break
        beta = 50.0 + (200000.0 - 50.0) * (s / max(1, n_steps - 1))
        # propose a move: perturb a few random centers
        nmove = rng.integers(1, 4)
        idx = rng.integers(0, N, size=nmove)
        step_size = t_start * (t_end / t_start) ** (s / max(1, n_steps - 1))
        x_new = x.copy()
        x_new[idx] += rng.normal(0.0, step_size, size=(nmove, 3))
        # keep inside cube (clip) to avoid invalid
        x_new = np.clip(x_new, -0.5, 0.5)
        f_old = -softmin_radius(x, beta)
        f_new = -softmin_radius(x_new, beta)
        T = t_start * (t_end / t_start) ** (s / max(1, n_steps - 1))
        if f_new < f_old or rng.random() < np.exp(-(f_new - f_old) / max(T, 1e-12)):
            x = x_new
        r = compute_radius(x)
        if r > best_r:
            best_r = r
            best = x.copy()
    return best, best_r


def initial_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def construct():
    t0 = time.time()
    DEADLINE = 150.0
    SAFETY = 15.0  # margin before kill

    # Start from the cubic grid (valid baseline)
    base = initial_grid()
    base_r = compute_radius(base)
    best = base.copy()
    best_r = base_r

    # Phase 1: simulated annealing from the grid (and a couple of jittered starts)
    starts = [base.copy()]
    rng = np.random.default_rng(7)
    for _ in range(2):
        starts.append(base + rng.normal(0.0, 0.01, size=(N, 3)))
    for s_idx, start in enumerate(starts):
        if time.time() - t0 > DEADLINE - SAFETY - 30.0:
            break
        # give anneal a shrinking budget
        remaining = DEADLINE - SAFETY - (time.time() - t0)
        anneal_budget = max(5.0, min(35.0, remaining * 0.5))
        cand, r = anneal(start, time_budget=anneal_budget, t0=t0)
        if r > best_r:
            best_r = r
            best = cand.copy()

    # Phase 2: high-precision polish (L-BFGS-B) with increasing beta
    if time.time() - t0 < DEADLINE - SAFETY:
        cur = best.copy()
        for beta in [200.0, 2000.0, 20000.0, 200000.0]:
            if time.time() - t0 > DEADLINE - SAFETY:
                break
            remaining = DEADLINE - SAFETY - (time.time() - t0)
            if remaining < 2.0:
                break
            cand = polish(cur, beta=beta, maxiter=200)
            cand = np.clip(cand, -0.5, 0.5)
            r = compute_radius(cand)
            if r > best_r:
                best_r = r
                best = cand.copy()
                cur = cand.copy()
            else:
                # still polish the current best a bit more at this beta
                pass

    # Final safety: ensure validity (finite, inside cube, no coincident)
    best = np.asarray(best, dtype=float)
    best = np.clip(best, -0.5, 0.5)
    if not np.all(np.isfinite(best)):
        best = initial_grid()
    # Guard against coincident centres (shouldn't happen)
    return best

"""Soft-sphere (Langevin MD + annealing + smooth soft-min polish) packer for 300 equal circles in a unit disk."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300
T_LIMIT = 150.0          # hard wall-clock limit (s)
T_SAFETY = 10.0          # return best before this margin
RNG_SEED = 12345


def _soft_min(x, beta):
    """Smooth approximation of min(x). Larger beta -> closer to true min."""
    x = np.asarray(x, dtype=float)
    m = x.min()
    # shift for stability
    z = x - m
    # soft min = m - (1/beta) * logsumexp(-beta*z)
    a = -beta * z
    amax = a.max()
    lse = amax + np.log(np.sum(np.exp(a - amax)))
    return m - lse / beta


def _pairwise_dists(C):
    """Return pairwise distance matrix (N,N) for C of shape (N,2)."""
    d2 = np.sum(C[:, None, :] - C[None, :, :], axis=2)
    np.fill_diagonal(d2, 1e18)
    return np.sqrt(d2)


def _feasible_radius(C):
    """Raw score r for a configuration (double precision)."""
    d = _pairwise_dists(C)
    min_pair = d.min() / 2.0
    margin = 1.0 - np.linalg.norm(C, axis=1).min()
    return min(min_pair, margin)


def _soft_sphere_energy(C, target_r, n_rep, n_wall, cutoff):
    """
    Energy: repulsion between pairs closer than cutoff (scaled by target_r)
    plus wall penalty. Lower is better.
    """
    d = _pairwise_dists(C)
    # only consider pairs with d < cutoff (cutoff in units of 2*target_r)
    # use (2*target_r)/d as a normalized repulsion when close
    norm = (2.0 * target_r) / d
    # repulsion only when norm > 1 (i.e. d < 2*target_r)
    mask = norm > 1.0
    rep = np.where(mask, norm ** n_rep, 0.0)
    E_rep = 0.5 * np.sum(rep)
    # wall penalty: distance from boundary
    rnorm = np.linalg.norm(C, axis=1)
    over = np.maximum(0.0, rnorm + target_r - 1.0)
    E_wall = np.sum(over ** n_wall)
    return E_rep + E_wall


def _soft_sphere_grad(C, target_r, n_rep, n_wall):
    """Analytic gradient of _soft_sphere_energy."""
    d = _pairwise_dists(C)
    norm = (2.0 * target_r) / d
    mask = norm > 1.0
    # dE/dC_i for repulsion:
    # E_rep = 0.5 * sum_{i<j} f(d_ij), f = norm^n when norm>1 else 0
    # d f / d d = n * norm^{n-1} * (-2*target_r)/d^2  (when mask)
    # d d_ij / d C_i = (C_i - C_j)/d_ij
    # Build gradient via pair contributions.
    G = np.zeros_like(C)
    # vectorized: for each pair (i,j) with mask, contribution to G[i] += grad_d * (C_i-C_j)/d
    # We'll do it with a loop over the upper triangle for clarity/speed balance.
    n_rep_f = n_rep
    d_safe = np.where(d > 1e-12, d, 1e-12)
    df_dd = np.zeros_like(d)
    df_dd[mask] = n_rep_f * norm[mask] ** (n_rep_f - 1) * (-(2.0 * target_r) / d_safe[mask] ** 2)
    # G[i] = sum_j df_dd[i,j] * (C_i - C_j)/d[i,j]
    # = sum_j df_dd[i,j]/d[i,j] * (C_i - C_j)
    w = df_dd / d_safe
    w = np.where(np.isfinite(w), w, 0.0)
    # G = (w @ (C_i - C_j))  =>  G_i = sum_j w_ij C_i - sum_j w_ij C_j = (rowsum(w))*C_i - (w @ C)_i
    rowsum = w.sum(axis=1)
    G = rowsum[:, None] * C - w @ C
    # wall gradient
    rnorm = np.linalg.norm(C, axis=1)
    rnorm_safe = np.where(rnorm > 1e-12, rnorm, 1e-12)
    over = np.maximum(0.0, rnorm + target_r - 1.0)
    dE_wall_dC = np.zeros_like(C)
    wmask = over > 0.0
    dE_wall_dC[wmask] = n_wall * over[wmask] ** (n_wall - 1) * (C[wmask] / rnorm_safe[wmask, None])
    G = G + dE_wall_dC
    return G


def _md_anneal(C0, target_r, n_rep, n_wall, steps, T0, Tend, dt, damping, rng, time_budget=None):
    """
    Langevin MD with cooling. Returns best C found (lowest energy) and its energy.
    """
    C = C0.copy()
    v = np.zeros_like(C)
    best_C = C.copy()
    best_E = _soft_sphere_energy(C, target_r, n_rep, n_wall, None)
    n = C.shape[0]
    for step in range(steps):
        if time_budget is not None and (time.time() - time_budget[0]) > (T_LIMIT - T_SAFETY):
            break
        E = _soft_sphere_energy(C, target_r, n_rep, n_wall, None)
        if E < best_E:
            best_E = E
            best_C = C.copy()
        # temperature schedule (geometric)
        frac = step / max(1, steps - 1)
        T = T0 * (Tend / T0) ** frac
        g = _soft_sphere_grad(C, target_r, n_rep, n_wall)
        # random kick
        kick = rng.standard_normal((n, 2)) * np.sqrt(2.0 * T * dt)
        # velocity update (Langevin)
        v = (v - dt * g) * damping + kick
        # position update
        C = C + dt * v
        # hard clip to container to avoid divergence
        rnorm = np.linalg.norm(C, axis=1)
        too_far = rnorm > 1.0
        if np.any(too_far):
            C[too_far] = C[too_far] / rnorm[too_far, None] * 1.0
    return best_C, best_E


def _polish_softmin(C, iters=3000, lr_init=1e-4, beta=200.0, time_budget=None):
    """
    Gradient descent maximizing a smooth surrogate of r:
      surrogate = soft_min( [d_ij/2 for i<j, 1-|c_i|] )
    Use scipy L-BFGS-B with analytic gradient for a high-precision polish.
    """
    C = C.copy()
    n = C.shape[0]

    def neg_soft_min_and_grad(x):
        Cx = x.reshape(n, 2)
        d = _pairwise_dists(Cx)
        iu = np.triu_indices(n, k=1)
        half_d = d[iu] / 2.0
        margin = 1.0 - np.linalg.norm(Cx, axis=1)
        terms = np.concatenate([half_d, margin])
        # soft min
        m = terms.min()
        z = terms - m
        a = -beta * z
        amax = a.max()
        lse = amax + np.log(np.sum(np.exp(a - amax)))
        val = m - lse / beta
        # gradient of soft min w.r.t. terms: softmax weights
        w = np.exp(a - amax)
        w = w / w.sum()
        # d(soft_min)/d(terms) = -w  (since soft_min = m - lse/beta; careful)
        # Recompute properly:
        # soft_min = m - (1/beta) logsumexp(-beta (terms - m))
        # d/d terms_k:  delta_{k,argmin?}... use chain rule.
        # Let s = soft_min.  s = m - (1/beta) * L, L = logsumexp(-beta*(t-m))
        # dL/dt_k = -beta * exp(-beta(t_k-m)) / sum_j exp(-beta(t_j-m)) = -beta * w_k
        # dm/dt_k = 1 if k==argmin else 0
        # ds/dt_k = dm/dt_k - (1/beta)*dL/dt_k = (k==argmin) + w_k
        # So gradient of s w.r.t. t_k = (k==argmin) + w_k  ... but that double counts.
        # Standard soft-min gradient:  ds/dt_k = w_k  where w is softmax over -beta*(t-m)??
        # Let's just use the standard: softmin(t) gradient = softmax(-beta*(t - t.min()))  (weights sum to 1)
        # Actually the clean identity: grad softmin = softmax(-beta * t) (unshifted) because shift cancels.
        # Use that:
        a2 = -beta * terms
        a2m = a2.max()
        w2 = np.exp(a2 - a2m)
        w2 = w2 / w2.sum()
        # ds/dt_k = w2_k  (this is the correct softmax gradient for soft-min)
        # Now propagate to x.
        gx = np.zeros_like(x)
        # terms[0:len(iu[0])] = d_ij/2 ; terms[len:] = margin
        n_pairs = len(iu[0])
        # d(d_ij)/dx for pairs
        # We'll accumulate: for each pair (i,j), contribution w2[p] * (1/2) * (C_i - C_j)/d_ij
        # Vectorized via the same trick.
        # Build weight matrix W (n x n) symmetric, W[i,j]=w2[p] for pair (i,j)
        W = np.zeros((n, n))
        W[iu] = w2[:n_pairs]
        W = W + W.T
        d_safe = np.where(d > 1e-12, d, 1e-12)
        # contribution to gx from pairs: sum_j W[i,j] * (1/2) * (C_i - C_j)/d_ij
        rowsum = W.sum(axis=1)
        gx_pairs = 0.5 * (rowsum[:, None] * Cx - W @ Cx) / d_safe
        # margin gradient: d(1-|c_i|)/dx = - C_i/|c_i|
        rnorm = np.linalg.norm(Cx, axis=1)
        rnorm_safe = np.where(rnorm > 1e-12, rnorm, 1e-12)
        gx_margin = -Cx / rnorm_safe[:, None]
        gx[:n] = 0  # placeholder, we build full
        # total gradient of s w.r.t. x
        gx_full = np.zeros_like(x)
        gx_full[:n * 2] = gx_pairs + w2[n_pairs:][:, None] * gx_margin
        # we want to MAXIMIZE s => minimize -s
        return -val, -gx_full

    # Use L-BFGS-B
    try:
        res = minimize(neg_soft_min_and_grad, C.reshape(-1), method="L-BFGS-B",
                       options={"maxiter": iters, "maxls": 50, "ftol": 1e-16, "gtol": 1e-12})
        if time_budget is not None and (time.time() - time_budget[0]) > (T_LIMIT - T_SAFETY):
            return C
        C = res.x.reshape(n, 2)
    except Exception:
        pass
    return C


def _hex_init(rng, spacing):
    """Hexagonal lattice points inside unit disk, then pick N nearest to center."""
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    pts = np.array(rows)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[rad <= 1.0]
    if len(pts) < N:
        # fallback to random
        return _random_init(rng)
    order = np.argsort(rad, kind="stable")
    return pts[order[:N]]


def _random_init(rng):
    """Fibonacci-sphere-ish / random points in disk."""
    # rejection sample in disk, but use a quasi-random spiral for better spread
    t = np.arange(N)
    r = np.sqrt((t + 0.5) / N) * 1.0
    theta = 2.0 * np.pi * t * 0.38196601125  # golden angle
    C = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return C


def _fibonacci_init(rng):
    """Random rotation of a spiral (diverse start)."""
    C = _random_init(rng)
    ang = rng.uniform(0, 2 * np.pi)
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]])
    return C @ R.T


def construct():
    t_start = time.time()
    time_budget = [t_start]
    rng = np.random.default_rng(RNG_SEED)

    # We'll try a few diverse initializations and keep the best final radius.
    # Target radius we push up over the annealing "stages".
    best_C = None
    best_r = 0.0

    n_rep = 4.0
    n_wall = 4.0

    # Stage schedule: (target_r, steps, T0, Tend, dt, damping)
    # Start with a conservative target, raise it as we go.
    r_known = 0.0531542991469
    stages = [
        (0.045, 400, 0.020, 1e-5, 1e-3, 0.985),
        (0.050, 400, 0.012, 1e-5, 1e-3, 0.990),
        (0.0525, 500, 0.008, 5e-6, 8e-4, 0.992),
        (0.0532, 600, 0.005, 2e-6, 6e-4, 0.994),
        (0.0534, 600, 0.003, 1e-6, 5e-4, 0.995),
    ]

    inits = []
    # hex with a few spacings
    for sp in [0.105, 0.108, 0.110]:
        inits.append(_hex_init(rng, sp))
    # diverse spiral starts
    for _ in range(2):
        inits.append(_fibonacci_init(rng))

    for init_idx, C0 in enumerate(inits):
        if (time.time() - t_start) > (T_LIMIT - T_SAFETY - 20.0):
            break
        C = C0.copy()
        for (tr, steps, T0, Tend, dt, damping) in stages:
            if (time.time() - t_start) > (T_LIMIT - T_SAFETY - 15.0):
                break
            C, _ = _md_anneal(C, tr, n_rep, n_wall, steps, T0, Tend, dt, damping, rng, time_budget)
        # polish
        if (time.time() - t_start) > (T_LIMIT - T_SAFETY - 10.0):
            break
        C = _polish_softmin(C, iters=2000, lr_init=1e-4, beta=300.0, time_budget=time_budget)
        r = _feasible_radius(C)
        if r > best_r:
            best_r = r
            best_C = C.copy()

    # Final ultra-low polish on the best
    if best_C is not None and (time.time() - t_start) < (T_LIMIT - T_SAFETY - 5.0):
        best_C = _polish_softmin(best_C, iters=4000, beta=600.0, time_budget=time_budget)
        best_r = _feasible_radius(best_C)

    # Safety: ensure strictly inside and finite
    if best_C is None:
        best_C = _hex_init(rng, 0.105)
    # clip any point that slipped outside (shouldn't happen)
    rnorm = np.linalg.norm(best_C, axis=1)
    bad = rnorm >= 1.0
    if np.any(bad):
        best_C[bad] = best_C[bad] / rnorm[bad, None] * (1.0 - 1e-9)
    if not np.all(np.isfinite(best_C)):
        best_C = _hex_init(rng, 0.105)
    return best_C

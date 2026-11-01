"""Combined: soft-min gradient (L-BFGS-B) + simulated annealing + force-directed + greedy."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
# EVOLVE-BLOCK-START

def _compute_r(centres):
    """Exact max radius for centres in [-0.5, 0.5]^3 using cKDTree (fast)."""
    if not np.all(np.isfinite(centres)):
        return 0.0
    if centres.shape != (N, 3):
        return 0.0
    wall = np.min(0.5 - np.abs(centres))
    if wall < 0:
        return 0.0
    tree = cKDTree(centres)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        return wall
    d = centres[pairs[:, 0]] - centres[pairs[:, 1]]
    min_pair = np.sqrt(np.sum(d * d, axis=1)).min()
    return min(min_pair / 2.0, wall)


def _softmin_obj_grad(x, beta, iu):
    """Soft-min of the radius constraints with analytic gradient.

    Minimizing G = (1/beta) ln sum_k exp(-beta c_k)  ->  maximizes min_k c_k.
    """
    c = x.reshape(N, 3)
    wall = (0.5 - np.abs(c)).ravel()                      # 3N values
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair = (dists / 2.0).ravel()[iu]                       # M values
    vals = np.concatenate([wall, pair])

    z = -beta * vals
    z = z - z.max()
    e = np.exp(z)
    S = e.sum()
    w = e / S
    obj = (z.max() + np.log(S)) / beta

    grad = np.zeros_like(c)
    # walls: dG/dx = w_wall * sign(c)
    grad += w[:3 * N].reshape(N, 3) * np.sign(c)
    # pairs: dG/dx_i = -w_i * (c_i - c_j)/d_ij  summed over j
    wp = w[3 * N:]
    wmat = np.zeros((N, N))
    wmat[iu] = wp
    wmat[iu[1], iu[0]] = wp
    mask = dists > 1e-12
    inv = np.zeros_like(dists)
    inv[mask] = 1.0 / dists[mask]
    grad -= np.sum((wmat * inv)[:, :, None] * diff, axis=1)
    return obj, grad.ravel()


def _opt_softmin(x0, deadline, beta_start=20.0, beta_end=30000.0, n_steps=8):
    """Anneal beta upward, L-BFGS-B on the soft-min objective."""
    best_x = x0.copy()
    best_r = _compute_r(x0.reshape(N, 3))
    iu = np.triu_indices(N, k=1)
    bounds = [(-0.5, 0.5)] * (N * 3)
    for beta in np.geomspace(beta_start, beta_end, n_steps):
        if time.time() > deadline:
            break
        try:
            res = minimize(_softmin_obj_grad, best_x, args=(beta, iu), jac=True,
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 40})
            nx = np.clip(res.x, -0.5, 0.5)
            nr = _compute_r(nx.reshape(N, 3))
            if nr > best_r:
                best_r, best_x = nr, nx
        except Exception:
            pass
    return best_x, best_r


def _anneal(x0, deadline, beta=400.0):
    """Simulated annealing on the soft-min objective to escape local minima."""
    c = x0.reshape(N, 3).copy()
    best_c, best_r = c.copy(), _compute_r(c)
    iu = np.triu_indices(N, k=1)
    _, obj_c = _softmin_obj_grad(c.ravel(), beta, iu)
    T = 0.02
    rng = np.random.default_rng(1234)
    while time.time() < deadline:
        T *= 0.985
        scale = max(0.002, 0.012 * T / 0.02)
        cn = np.clip(c + rng.normal(0, scale, c.shape), -0.5, 0.5)
        try:
            obj_n, _ = _softmin_obj_grad(cn.ravel(), beta, iu)
        except Exception:
            continue
        d = obj_n - obj_c
        if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            c, obj_c = cn, obj_n
            r = _compute_r(c)
            if r > best_r:
                best_r, best_c = r, c.copy()
    return best_c.ravel(), best_r


def _force_refine(centres, max_iters=250):
    """Force-directed repulsion with wall constraints and adaptive step."""
    c = centres.copy().astype(np.float64)
    best_c, best_r = c.copy(), _compute_r(c)
    for it in range(max_iters):
        tree = cKDTree(c)
        pairs = tree.query_pairs(0.4, output_type='ndarray')
        F = np.zeros_like(c)
        if len(pairs):
            for i, j in pairs:
                d = c[i] - c[j]
                dist = np.sqrt((d * d).sum())
                if dist < 1e-12:
                    d, dist = np.random.randn(3) * 1e-6, 1e-6
                fm = 1.0 / (dist * dist + 1e-8)
                F[i] += fm * d / dist
                F[j] -= fm * d / dist
        for ax in range(3):
            over = c[:, ax] - 0.5
            m = over > -0.05
            if m.any():
                F[m, ax] -= 50.0 * np.clip(over[m], 0, None) + np.clip(0.05 + over[m], 0, None)
            under = c[:, ax] + 0.5
            mn = under < 0.05
            if mn.any():
                F[mn, ax] += 50.0 * np.clip(-under[mn], 0, None) + np.clip(0.05 - under[mn], 0, None)
        c += (0.015 * 0.997 ** it) * F
        c = np.clip(c, -0.499, 0.499)
        r = _compute_r(c)
        if r > best_r:
            best_r, best_c = r, c.copy()
    return best_c, best_r


def _greedy(centres, deadline, rng):
    """Greedy single-particle local search with random kicks."""
    c = centres.copy()
    best_c, best_r = c.copy(), _compute_r(c)
    step, kick, noimp, maxno = 0.008, 0.06, 0, 1500
    while time.time() < deadline:
        idx = rng.integers(0, N)
        cn = c.copy()
        cn[idx] = np.clip(cn[idx] + rng.normal(0, step, 3), -0.5, 0.5)
        rn = _compute_r(cn)
        if rn > best_r + 1e-12:
            c, best_c, best_r, noimp = cn, cn.copy(), rn, 0
        else:
            noimp += 1
        if noimp > maxno:
            ki = rng.choice(N, size=12, replace=False)
            c[ki] = np.clip(c[ki] + rng.normal(0, kick, (12, 3)), -0.5, 0.5)
            rc = _compute_r(c)
            if rc > best_r:
                best_r, best_c = rc, c.copy()
            noimp, step = 0, 0.01
    return best_c, best_r


def _init_grid(rng, scale=0.0):
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    t = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in t for y in t for x in t])[:N].copy()
    if scale > 0:
        g = np.clip(g + rng.normal(0, scale, g.shape), -0.49, 0.49)
    return g


def construct():
    start = time.time()
    deadline = start + 140.0
    rng = np.random.default_rng(42)

    base = _init_grid(rng, 0.0)
    c, best_r = _force_refine(base, 250)
    best_c = c.copy()

    for restart in range(5):
        if time.time() > deadline - 45.0:
            break
        if restart == 0:
            init = c.copy()
        elif restart == 1:
            init = np.clip(base + rng.normal(0, 0.02, (N, 3)), -0.49, 0.49)
        elif restart == 2:
            init = rng.uniform(-0.45, 0.45, (N, 3))
        elif restart == 3:
            init = np.clip(best_c + rng.normal(0, 0.03, (N, 3)), -0.49, 0.49)
        else:
            init = np.clip(best_c + rng.normal(0, 0.05, (N, 3)), -0.49, 0.49)

        c, r = _force_refine(init, 150)
        if r > best_r:
            best_r, best_c = r, c.copy()
        if time.time() > deadline - 35.0:
            break

        c, r = _opt_softmin(c.ravel(), min(time.time() + 25.0, deadline - 25.0))
        if r > best_r:
            best_r, best_c = r, c.reshape(N, 3).copy()
        if time.time() > deadline - 25.0:
            break

        c, r = _anneal(c, min(time.time() + 15.0, deadline - 15.0))
        if r > best_r:
            best_r, best_c = r, c.reshape(N, 3).copy()
        if time.time() > deadline - 15.0:
            break

        c, r = _greedy(c, min(time.time() + 12.0, deadline - 5.0), rng)
        if r > best_r:
            best_r, best_c = r, c.copy()

    fr = _compute_r(best_c)
    if fr > best_r:
        best_r = fr
    return best_c
# EVOLVE-BLOCK-END

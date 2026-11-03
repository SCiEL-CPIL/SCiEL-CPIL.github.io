"""Smooth-surrogate SGLD (Lennard-Jones-like repulsion) for packing 100 equal spheres
in the unit cube [-1/2, 1/2]^3, with a final L-BFGS-B polish.

The nonsmooth objective  r = min( min_{i<j} |c_i-c_j|/2 , min_i min_k (1/2-|c_ik|) )
is approximated by a smooth repulsion energy
    E = sum_{i<j} exp(-d_ij^2 / sigma^2)  +  sum_i sum_k exp(-w_ik^2 / tau^2)
whose gradient pushes particles apart and away from the walls.  We run Stochastic
Gradient Langevin Dynamics (gradient step + annealed noise) with an annealed kernel
width sigma, then polish each good basin with L-BFGS-B.  The returned point is always
the one with the largest EXACT radius seen, so validity is guaranteed.
"""
import time
import numpy as np

N = 100
HALF = 0.5
DEADLINE_MARGIN = 6.0   # safety margin before the 150 s kill


# ---------------- exact objective (the thing we actually score) ----------------
def _radius(pts):
    """Exact feasible common radius for a set of centres (vectorized)."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2))
    wall = HALF - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall)
    r = 0.5 * min(min_pair, 2.0 * min_wall)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return float(r)


# ---------------- smooth surrogate energy + gradient ----------------
def _energy_grad(pts, sigma, tau):
    """Return (E, grad) for the smooth repulsion energy.  O(N^2)."""
    n = pts.shape[0]
    diff = pts[:, None, :] - pts[None, :, :]          # (n,n,3)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 0.0)

    # pair term: exp(-d2/sigma^2);  d/d pts_i sum_{j!=i} exp(-d2/s^2)
    #   = -2/s^2 * sum_{j!=i} (pts_i - pts_j) exp(-d2/s^2)
    inv_s2 = 1.0 / (sigma * sigma)
    w = np.exp(-d2 * inv_s2)
    np.fill_diagonal(w, 0.0)
    grad_pair = -2.0 * inv_s2 * np.einsum('ij,ijk->ik', w, diff)
    E_pair = 0.5 * np.sum(w)

    # wall term: for each coordinate, w_ik = HALF - |c_ik| (>=0 if inside)
    #   f = exp(-w^2/tau^2);  d f/d c_ik = -2/tau^2 * w * sign(c_ik) * exp(-w^2/tau^2)
    inv_t2 = 1.0 / (tau * tau)
    wv = HALF - np.abs(pts)                            # (n,3)
    fw = np.exp(-wv * wv * inv_t2)
    sign = np.sign(pts)
    grad_wall = -2.0 * inv_t2 * wv * sign * fw
    E_wall = np.sum(fw)

    E = E_pair + E_wall
    grad = grad_pair + grad_wall
    return E, grad


# ---------------- initializers ----------------
def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _init_random(rng):
    return (rng.random((N, 3)) - 0.5) * 0.9


def _clamp(pts):
    return np.clip(pts, -HALF, HALF)


# ---------------- SGLD on the smooth surrogate ----------------
def _sgld(pts, rng, deadline, sigma0, sigma1, tau):
    """Run annealed SGLD on the smooth energy; return (best_pts, best_exact_r, best_E)."""
    pts = _clamp(pts.copy())
    E, _ = _energy_grad(pts, sigma0, tau)
    best_E = E
    best_pts = pts.copy()
    best_r = _radius(best_pts)

    n_steps = 4000
    lr0, lr1 = 0.02, 1e-4
    T0, T1 = 0.03, 1e-4

    for it in range(n_steps):
        if time.time() > deadline:
            break
        frac = it / max(1, n_steps - 1)
        sigma = sigma0 * (sigma1 / sigma0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        T = T0 * (T1 / T0) ** frac

        E, grad = _energy_grad(pts, sigma, tau)
        noise = np.sqrt(2.0 * lr * T) * rng.standard_normal((N, 3))
        pts = _clamp(pts - lr * grad + noise)

        # cheap surrogate check to decide whether to spend an exact evaluation
        E2, _ = _energy_grad(pts, sigma, tau)
        if E2 < best_E:
            best_E = E2
            r = _radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()

    # final exact eval of the current state
    r = _radius(pts)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()

    return best_pts, best_r, best_E


# ---------------- L-BFGS-B polish on the smooth surrogate ----------------
def _polish(pts, deadline):
    """A few L-BFGS-B passes to sharpen the basin; return best exact radius point."""
    from scipy.optimize import minimize
    pts = _clamp(pts.copy())
    best_r = _radius(pts)
    best_pts = pts.copy()
    for sigma in (0.06, 0.04, 0.03):
        if time.time() > deadline:
            break
        tau = 0.03

        def fun(x):
            e, _ = _energy_grad(x.reshape(N, 3), sigma, tau)
            return float(e)

        def gfun(x):
            _, g = _energy_grad(x.reshape(N, 3), sigma, tau)
            return g.ravel()

        try:
            res = minimize(fun, pts.ravel(), jac=gfun, method='L-BFGS-B',
                           bounds=[(-HALF, HALF)] * (N * 3),
                           options={'maxiter': 80, 'ftol': 1e-9, 'gtol': 1e-6})
            cand = _clamp(res.x.reshape(N, 3))
            r = _radius(cand)
            if r > best_r:
                best_r = r
                best_pts = cand
        except Exception:
            pass
    return best_pts, best_r


# ---------------- top level ----------------
def construct():
    t0 = time.time()
    deadline = 150.0 - DEADLINE_MARGIN

    rng = np.random.default_rng(20240517)

    # global best
    best_pts = _init_grid()
    best_r = _radius(best_pts)

    # initializations to try: grid + several random
    inits = [best_pts]
    for _ in range(5):
        inits.append(_init_random(rng))

    # SGLD parameters
    sigma0, sigma1 = 0.25, 0.02
    tau = 0.04

    for init in inits:
        if time.time() > deadline:
            break
        # quick polish of the raw init to get a good basin start
        p0, r0 = _polish(init, deadline)
        if r0 > best_r:
            best_r = r0
            best_pts = p0

        # SGLD from that polished start
        p1, r1, _ = _sgld(p0, rng, deadline, sigma0, sigma1, tau)
        if r1 > best_r:
            best_r = r1
            best_pts = p1

        # final polish of the SGLD result
        p2, r2 = _polish(p1, deadline)
        if r2 > best_r:
            best_r = r2
            best_pts = p2

    # Use spare time for extra SGLD restarts if available (conditional: only if
    # there is budget left AND it is expected to help escape local minima).
    guard = 0
    while time.time() < deadline - 3.0 and guard < 8:
        guard += 1
        start = _init_random(rng)
        p0, r0 = _polish(start, deadline)
        if r0 > best_r:
            best_r = r0
            best_pts = p0
        p1, r1, _ = _sgld(p0, rng, deadline, sigma0, sigma1, tau)
        if r1 > best_r:
            best_r = r1
            best_pts = p1
        p2, r2 = _polish(p1, deadline)
        if r2 > best_r:
            best_r = r2
            best_pts = p2

    # final safety: ensure the returned point is feasible and finite
    out = _clamp(np.asarray(best_pts, dtype=np.float64))
    if not np.all(np.isfinite(out)):
        out = _init_grid()
    return out

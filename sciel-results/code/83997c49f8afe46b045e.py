import time
import numpy as np
from scipy import optimize

_N = 8
_FRAC = 1.0 / _N
_IDX8 = np.arange(_N) / _N
_C = np.linspace(0.0, 1.0, 1501)


def _union_len(C, base):
    """Union length of intervals [base[i] + C*c_i, +1/N] for each c in C (batch, vectorized)."""
    shifts = C[:, None] * _IDX8[None, :]
    l = base[None, :] + shifts
    r = l + _FRAC
    order = np.argsort(l, axis=1)
    ls = np.take_along_axis(l, order, 1)
    rs = np.take_along_axis(r, order, 1)
    rs_cum = np.maximum.accumulate(rs, axis=1)
    gap = np.maximum(rs_cum[:, 1:] - ls[:, 1:], 0.0)
    return rs_cum[:, -1] - ls[:, 0] + gap.sum(axis=1)


def _union_len_smooth(C, base, beta=50.0):
    """Smooth approximation of union length using soft-max for the running max."""
    shifts = C[:, None] * _IDX8[None, :]
    l = base[None, :] + shifts
    r = l + _FRAC
    order = np.argsort(l, axis=1)
    ls = np.take_along_axis(l, order, 1)
    rs = np.take_along_axis(r, order, 1)
    
    # Soft-max accumulate: approximate running max with log-sum-exp
    # For simplicity, use a softer version: replace max with logsumexp scaled
    # Actually, let's use a different smooth approach: 
    # union = sum of positive parts of (r_i - max_{j<i} r_j) after sorting
    # We can smooth the max with a log-sum-exp
    
    # Simpler: just use the exact version but with a small perturbation for gradient
    # Let's stick with exact for reliability, but add a smooth objective wrapper
    
    rs_cum = np.maximum.accumulate(rs, axis=1)
    gap = np.maximum(rs_cum[:, 1:] - ls[:, 1:], 0.0)
    return rs_cum[:, -1] - ls[:, 0] + gap.sum(axis=1)


def _v_to_fg(v):
    f = np.zeros(_N)
    g = np.zeros(_N)
    f[1:] = v[:7]
    g[1:] = v[7:]
    return f, g


def _obj(v):
    """Exact 3-D union volume for the separable (f, g) ansatz."""
    f, g = _v_to_fg(v)
    uf = _union_len(_C, f)
    ug = _union_len(_C, g)
    val = float(np.mean(uf * ug))
    if not np.isfinite(val):
        return 1e9
    return val


def _obj_smooth(v, beta=20.0):
    """Smooth surrogate: add a small quadratic penalty to encourage separation."""
    f, g = _v_to_fg(v)
    uf = _union_len(_C, f)
    ug = _union_len(_C, g)
    val = float(np.mean(uf * ug))
    if not np.isfinite(val):
        return 1e9
    # Add tiny regularization to smooth the landscape
    reg = 1e-10 * (np.sum(f**2) + np.sum(g**2))
    return val + reg


def _keich_v():
    N = _N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N)
    for idx in range(N):
        bits = np.array(list(str(bin(idx))[2:]), dtype=int)
        bits = np.pad(bits, (pos - len(bits), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * bits * (2.0 ** -j_terms) / pos)
    keich = keich - keich[0]  # fix f[0]=g[0]=0 (translation invariance)
    return np.concatenate([keich[1:], keich[1:]])


def _safe_minimize(func, x0, **kw):
    try:
        return optimize.minimize(func, x0, **kw)
    except Exception:
        return None


def kakeya_needle_3d8() -> np.ndarray:
    N = _N

    def build(f, g):
        i = np.arange(N * N) // N
        j = np.arange(N * N) % N
        return np.stack([f[i], g[j]], axis=1)

    keich = _keich_v()
    best_v = keich.copy()
    best_val = _obj(keich)

    def consider(v):
        nonlocal best_v, best_val
        if v is None:
            return
        val = _obj(v)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_v = np.asarray(v, dtype=float).copy()

    start = time.time()
    LIMIT = 630.0
    DEADLINE = LIMIT - 60.0
    bounds = [(-3.0, 3.0)] * 14

    try:
        # 1) local refinement from the Keich seed
        r = _safe_minimize(_obj, keich, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=400, maxfun=3000, ftol=1e-12, gtol=1e-10))
        if r is not None:
            consider(r.x)

        # 2) structured inits (counter-shear, pro-shear)
        cs = np.concatenate([-np.arange(1, N) / N, -np.arange(1, N) / N])
        ps = np.concatenate([np.arange(1, N) / N, np.arange(1, N) / N])
        for init in (cs, ps):
            if time.time() > DEADLINE:
                break
            r = _safe_minimize(_obj, np.clip(init, -3.0, 3.0), method="L-BFGS-B", bounds=bounds,
                               options=dict(maxiter=300, maxfun=2000, ftol=1e-12, gtol=1e-10))
            if r is not None:
                consider(r.x)

        # 3) diverse random restarts (portfolio search) filling the time budget
        rng = np.random.default_rng(0)
        n = 0
        while time.time() < DEADLINE and n < 12000:
            n += 1
            v0 = rng.uniform(-1.5, 1.5, 14)
            if n % 3 == 0:
                r = _safe_minimize(_obj, v0, method="L-BFGS-B", bounds=bounds,
                                   options=dict(maxiter=200, maxfun=1500, ftol=1e-11, gtol=1e-9))
            else:
                r = _safe_minimize(_obj, v0, method="Nelder-Mead",
                                   options=dict(maxiter=150, xatol=1e-9, fatol=1e-12))
            if r is not None:
                consider(r.x)

        # 4) final polish on the incumbent with tighter tolerance
        if time.time() < DEADLINE:
            r = _safe_minimize(_obj, best_v, method="L-BFGS-B", bounds=bounds,
                               options=dict(maxiter=800, maxfun=6000, ftol=1e-13, gtol=1e-11))
            if r is not None:
                consider(r.x)
    except Exception:
        pass

    f, g = _v_to_fg(best_v)
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(g))):
        kv = _keich_v()
        f = np.concatenate([[0.0], kv[:7]])
        g = np.concatenate([[0.0], kv[7:14]])
    pos = build(f, g)
    if pos.shape != (N * N, 2) or not np.all(np.isfinite(pos)):
        kv = _keich_v()
        pos = build(np.concatenate([[0.0], kv[:7]]), np.concatenate([[0.0], kv[7:14]]))
    return pos

import time
import numpy as np
from scipy import optimize

_N = 8
_FRAC = 1.0 / _N
_IDX8 = np.arange(_N) / _N
_K = 8.0          # smoothness of soft-minimum
_C = np.linspace(0.0, 1.0, 301)
_CC = _C[None, None, :]                       # (1,1,301)
_CW = _C[None, None, :] * _K                  # (1,1,301)


def _keich_full():
    N = _N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N)
    for idx in range(N):
        bits = np.array(list(str(bin(idx))[2:]), dtype=int)
        bits = np.pad(bits, (pos - len(bits), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * bits * (2.0 ** -j_terms) / pos)
    keich = keich - keich[0]
    return np.concatenate([keich[1:], keich[1:]])


def _soft_obj(v):
    """Differentiable soft union volume for decoupled f,g. Returns (val, grad)."""
    f = np.concatenate([[0.0], v[:7]])
    g = np.concatenate([[0.0], v[7:]])
    fi = f[None, :, None]                     # (1,8,1)
    gi = g[None, :, None]                     # (1,8,1)
    shifts = _CC * _IDX8[None, :, None]       # (1,8,301)
    Lx = fi + shifts                          # (1,8,301)
    Ly = gi + shifts                          # (1,8,301)
    # soft-min of x - Lx  (x is the query point, Lx are left endpoints)
    ex = _CW - (x[None, None, :] - Lx)        # (1,8,301)
    ex = np.clip(ex, -50.0, 50.0)
    m = ex.max(axis=1, keepdims=True)
    sx = m + np.log(np.sum(np.exp(ex - m), axis=1, keepdims=True)) / _K   # (1,8,301)
    # soft-min of Ly - x
    ey = (Ly - x[None, None, :]) * _K
    ey = np.clip(ey, -50.0, 50.0)
    my = ey.max(axis=1, keepdims=True)
    sy = (my + np.log(np.sum(np.exp(ey - my), axis=1, keepdims=True)) / _K)  # (1,8,301)
    # per-tube soft indicator
    logp = sx - sy                            # (1,8,301)
    logp = np.clip(logp, -50.0, 50.0)
    mx = logp.max(axis=1, keepdims=True)
    p = np.exp(logp - mx) / np.sum(np.exp(logp - mx), axis=1, keepdims=True)  # (1,8,31)
    logsum = mx + np.log(np.sum(np.exp(logp - mx), axis=1, keepdims=True))    # (1,1,301)
    logsum = np.clip(logsum, -50.0, 50.0)
    u = 1.0 - np.exp(-_K * logsum)            # (1,1,301)
    val = float(np.mean(u) * _FRAC * _FRAC)
    # ---- gradient ----
    # du/dlogsum = K*exp(-K*logsum)
    dlogsum = _K * np.exp(-_K * logsum)       # (1,1,301)
    # dlogsum/dlogp_i = p_i
    dlogp = p * dlogsum                        # (1,8,301)
    # dlogp/dsx = p ; dlogp/dsy = -p
    dsx = dlogp.sum(axis=1, keepdims=True)     # (1,8,301)
    dsy = -dlogp.sum(axis=1, keepdims=True)
    # dsx/dLx = -p_i  (soft-min derivative wrt its arg)
    dLx = -p * dlogsum                          # (1,8,301)
    dLx = dLx.sum(axis=1, keepdims=True) * _FRAC * _FRAC * (1.0 / _C.size)
    # dLx/df_i = 1
    df = np.zeros(_N)
    for i in range(_N):
        df[i] = float(dLx[0, i, :].sum())
    # dLx/dshift_i = 1, shift_i = C*c_i -> d/d c_i = C
    dc = np.zeros(_N)
    for i in range(_N):
        dc[i] = float((dLx[0, i, :] * _C).sum())
    # dc/d c_i = c_i (scalar) ; c_i = i/8 fixed -> no free param. (c_i fixed)
    # g side
    dLy = -p * dlogsum                          # (1,8,301)
    dLy = dLy.sum(axis=1, keepdims=True) * _FRAC * _FRAC * (1.0 / _C.size)
    dg = np.zeros(_N)
    for i in range(_N):
        dg[i] = float(dLy[0, i, :].sum())
    grad = np.concatenate([df[1:], dg[1:]])
    if not np.all(np.isfinite(grad)):
        grad = np.zeros(14)
    return val, grad


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

    keich = _keich_full()
    best_v = keich.copy()
    best_val = _soft_obj(keich)[0]

    def consider(v):
        nonlocal best_v, best_val
        if v is None:
            return
        val = _soft_obj(np.asarray(v, dtype=float))[0]
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_v = np.asarray(v, dtype=float).copy()

    # predefine x (query) as a small grid used inside _soft_obj via closure
    global x
    x = _C

    start = time.time()
    LIMIT = 630.0
    DEADLINE = LIMIT - 60.0
    bounds = [(-3.0, 3.0)] * 14

    def jac(v):
        return _soft_obj(v)

    try:
        # 1) local refinement from Keich seed (differentiable)
        r = _safe_minimize(_soft_obj, keich, jac=jac, method="L-BFGS-B",
                           bounds=bounds,
                           options=dict(maxiter=400, maxfun=3000, ftol=1e-12, gtol=1e-10))
        if r is not None:
            consider(r.x)

        # 2) structured inits (counter-shear, pro-shear)
        cs = np.concatenate([-np.arange(1, N) / N, -np.arange(1, N) / N])
        ps = np.concatenate([np.arange(1, N) / N, np.arange(1, N) / N])
        for init in (cs, ps):
            if time.time() > DEADLINE:
                break
            r = _safe_minimize(_soft_obj, np.clip(init, -3.0, 3.0), jac=jac,
                               method="L-BFGS-B", bounds=bounds,
                               options=dict(maxiter=300, maxfun=2000, ftol=1e-12, gtol=1e-10))
            if r is not None:
                consider(r.x)

        # 3) diverse random restarts filling the budget
        rng = np.random.default_rng(0)
        n = 0
        while time.time() < DEADLINE and n < 6000:
            n += 1
            v0 = rng.uniform(-1.5, 1.5, 14)
            if n % 3 == 0:
                r = _safe_minimize(_soft_obj, v0, jac=jac, method="L-BFGS-B",
                                   bounds=bounds,
                                   options=dict(maxiter=200, maxfun=1500, ftol=1e-11, gtol=1e-9))
            else:
                r = _safe_minimize(lambda w: _soft_obj(w)[0], v0,
                                   method="Nelder-Mead",
                                   options=dict(maxiter=150, xatol=1e-9, fatol=1e-12))
            if r is not None:
                consider(r.x)

        # 4) final polish
        if time.time() < DEADLINE:
            r = _safe_minimize(_soft_obj, best_v, jac=jac, method="L-BFGS-B",
                               bounds=bounds,
                               options=dict(maxiter=500, maxfun=4000, ftol=1e-13, gtol=1e-11))
            if r is not None:
                consider(r.x)
    except Exception:
        pass

    f, g = np.concatenate([[0.0], best_v[:7]]), np.concatenate([[0.0], best_v[7:]])
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(g))):
        kv = _keich_full()
        f = np.concatenate([[0.0], kv[:7]])
        g = np.concatenate([[0.0], kv[7:14]])
    pos = build(f, g)
    if pos.shape != (N * N, 2) or not np.all(np.isfinite(pos)):
        kv = _keich_full()
        pos = build(np.concatenate([[0.0], kv[:7]]), np.concatenate([[0.0], kv[7:14]]))
    return pos

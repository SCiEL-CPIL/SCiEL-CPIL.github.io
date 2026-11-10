# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 620.0


def _mc_volume_fast(xv, yv):
    """Fast fixed-seed MC volume estimate for product tubes.

    xv: shape (8,) base x for i=0..7
    yv: shape (8,) base y for j=0..7
    Returns (V, valid).
    """
    if not (np.all(np.isfinite(xv)) and np.all(np.isfinite(yv))):
        return 0.0, False
    N = 8
    w = 1.0 / N
    # x_i(c) = xv[i] + c*i/N ; X0 = min_i xv[i], X1 = max_i xv[i] + (i+1)/N
    X0 = float(np.min(xv))
    X1 = float(np.max(xv + np.arange(N) * w))
    Y0 = float(np.min(yv))
    Y1 = float(np.max(yv + np.arange(N) * w))
    if (X1 - X0) <= 1e-9 or (Y1 - Y0) <= 1e-9:
        return 0.0, False
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, 500000)
    b = rng.uniform(Y0, Y1, 500000)
    c = rng.uniform(0.0, 1.0, 500000)
    # vectorized containment: point (a,b,c) in tube (i,j) iff
    # a in [xv[i]+c*i/N, xv[i]+c*i/N+w] and b in [yv[j]+c*j/N, yv[j]+c*j/N+w]
    ci = c * (np.arange(N) * w)[None, :]          # (M,8)
    xlo = xv[None, :] + ci                         # (M,8)
    xin = (a[:, None] >= xlo) & (a[:, None] <= xlo + w)
    xhit = xin.any(axis=1)                         # (M,)
    cj = c * (np.arange(N) * w)[None, :]
    ylo = yv[None, :] + cj
    yin = (b[:, None] >= ylo) & (b[:, None] <= ylo + w)
    yhit = yin.any(axis=1)
    frac = np.mean(xhit & yhit)
    V = frac * (X1 - X0) * (Y1 - Y0)
    if not np.isfinite(V) or V <= 0:
        return 0.0, False
    return float(V), True


def _keich_positions_1d():
    N = 8
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _objective(params):
    xv = params[:8]
    yv = params[8:]
    V, ok = _mc_volume_fast(xv, yv)
    if not ok:
        return 1e6
    return V


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    keich = _keich_positions_1d()
    best_params = np.concatenate([keich, keich])
    best_V, _ = _mc_volume_fast(keich, keich)
    if not np.isfinite(best_V) or best_V <= 0:
        best_V = 1e6

    def _try(params):
        nonlocal best_params, best_V
        V, ok = _mc_volume_fast(params[:8], params[8:])
        if ok and np.isfinite(V) and V < best_V:
            best_V = V
            best_params = params.copy()

    # Local optimization with L-BFGS-B on the (8,8) parameter vector.
    try:
        from scipy.optimize import minimize
        # Try a few starting points: keich, keich shifted, and a spread start.
        starts = [
            np.concatenate([keich, keich]),
            np.concatenate([keich + 0.05, keich - 0.05]),
            np.concatenate([np.linspace(0, 0.6, 8), np.linspace(0, 0.6, 8)]),
            np.concatenate([np.linspace(-0.2, 0.4, 8), np.linspace(-0.2, 0.4, 8)]),
        ]
        for s in starts:
            if time.time() > _DEADLINE - 30:
                break
            res = minimize(_objective, s, method="L-BFGS-B",
                           options={"maxiter": 400, "maxfun": 20000,
                                    "ftol": 1e-12, "gtol": 1e-8})
            if time.time() > _DEADLINE - 30:
                break
            if res.success or np.isfinite(res.fun):
                _try(res.x)
    except Exception:
        pass

    # Final safety: ensure finite valid result.
    xv = best_params[:8]
    yv = best_params[8:]
    V, ok = _mc_volume_fast(xv, yv)
    if not ok:
        xv = keich.copy()
        yv = keich.copy()

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = xv[i_flat]
    base_y = yv[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END

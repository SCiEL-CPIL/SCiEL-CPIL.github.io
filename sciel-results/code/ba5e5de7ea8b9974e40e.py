# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 600.0 - 25.0  # safety margin before the 600s kill
_N = 8
_S = 1.0 / _N  # tube width


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _union_len_1d(pos, N=_N, s=_S):
    """Exact total length of the union of the N sheared 1-D intervals.

    pos: (N,) base positions. Interval k = [pos[k]+c*k/N, pos[k]+s+c*k/N], c in [0,1].
    """
    n = N
    idx = np.arange(n, dtype=np.float64)
    s = np.float64(s)
    L = pos + idx * (s / n)          # left endpoints
    R = L + s                        # right endpoints
    order = np.argsort(L, kind="stable")
    Ls = L[order]
    Rs = R[order]
    total = np.float64(0.0)
    cur_l = Ls[0]
    cur_r = Rs[0]
    for k in range(1, n):
        l = Ls[k]
        r = Rs[k]
        if l <= cur_r + 1e-15:
            if r > cur_r:
                cur_r = r
        else:
            total += cur_r - cur_l
            cur_l = l
            cur_r = r
    total += cur_r - cur_l
    return float(total)


def _softmin(a, b, t):
    """Smooth min(a, b) via log-sum-exp (t = temperature)."""
    m = np.minimum(a, b)
    return m - t * np.log1p(np.exp(-(a - b) / t))


def _softmax(a, b, t):
    """Smooth max(a, b) via log-sum-exp."""
    m = np.maximum(a, b)
    return m + t * np.log1p(np.exp(-(a - b) / t))


def _objective_1d(pos, t):
    """Smooth surrogate for the 1-D union volume (X-range * P_x). Lower is better."""
    n = _N
    idx = np.arange(n, dtype=np.float64)
    s = np.float64(_S)
    L = pos + idx * (s / n)
    R = L + s
    order = np.argsort(L, kind="stable")
    Ls = L[order]
    Rs = R[order]
    cur_l = Ls[0]
    cur_r = Rs[0]
    total = np.float64(0.0)
    for k in range(1, n):
        l = Ls[k]
        r = Rs[k]
        if l <= cur_r + 1e-9:
            cur_r = _softmax(cur_r, r, t)
        else:
            total = total + (cur_r - cur_l)
            cur_l = l
            cur_r = r
    total = total + (cur_r - cur_l)
    X1 = _softmax(Rs, np.float64(0.0), t)
    X0 = _softmin(Ls, np.float64(0.0), t)
    return float((X1 - X0) * total)


def _exact_1d_volume(pos):
    """Exact 1-D factor: X-range * P_x = (X1-X0) * union_length."""
    n = _N
    idx = np.arange(n, dtype=np.float64)
    s = np.float64(_S)
    L = pos + idx * (s / n)
    R = L + s
    ulen = _union_len_1d(pos)
    X0 = float(min(0.0, float(L.min())))
    X1 = float(max(0.0, float(R.max())))
    return (X1 - X0) * ulen


def _keich_init(N=_N):
    """Keich binary construction (the seed) as a starting point."""
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    return keich.copy()


def _optimize_1d(pos0):
    """Optimize one 1-D coordinate set (N positions) to minimize the 1-D factor."""
    pos0 = np.asarray(pos0, dtype=np.float64).copy()
    best = pos0.copy()
    best_val = _exact_1d_volume(pos0)
    n = _N

    # --- Phase 1: L-BFGS-B on the smooth surrogate ---
    for t in (0.05, 0.02, 0.008, 0.003, 0.001):
        if _time_left() < 2.0:
            break
        try:
            from scipy.optimize import minimize
            res = minimize(
                _objective_1d,
                best,
                args=(t,),
                method="L-BFGS-B",
                jac="2-point",
                options={"maxiter": 400, "maxfun": 60000, "ftol": 1e-12, "gtol": 1e-10},
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _exact_1d_volume(res.x)
                if v < best_val - 1e-14:
                    best = res.x.copy()
                    best_val = v
        except Exception:
            pass

    # --- Phase 2: Nelder-Mead polish on the exact objective ---
    for trial in range(3):
        if _time_left() < 3.0:
            break
        try:
            from scipy.optimize import minimize
            res = minimize(
                _exact_1d_volume,
                best,
                method="Nelder-Mead",
                options={"maxiter": 4000, "maxfev": 200000, "xatol": 1e-11,
                         "fatol": 1e-13, "adaptive": True},
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _exact_1d_volume(res.x)
                if v < best_val - 1e-14:
                    best = res.x.copy()
                    best_val = v
        except Exception:
            pass

    return best, best_val


def kakeya_needle_3d8() -> np.ndarray:
    """
    3-D discrete Kakeya needle, N = 8. Returns (64, 2) base positions.

    Exploits the product structure: the union is U_x x U_y, so the volume is
    (X1-X0)*(Y1-Y0)*P_x*P_y. We optimize the 8 x-positions and 8 y-positions
    independently (each an 8-D problem) using a smooth log-sum-exp surrogate
    (L-BFGS-B) followed by exact Nelder-Mead polishing, which is far faster and
    more reliable than optimizing all 128 coordinates jointly.
    """
    if _time_left() <= 0.0:
        # Fallback: return the seed construction immediately.
        keich = _keich_init(_N)
        i_flat, j_flat = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
        i_flat = i_flat.ravel()
        j_flat = j_flat.ravel()
        return np.stack([keich[i_flat], keich[j_flat]], axis=1)

    # Optimize x (depends on i) and y (depends on j) from the Keich seed.
    keich = _keich_init(_N)
    x_pos, x_val = _optimize_1d(keich)
    y_pos, y_val = _optimize_1d(keich)

    i_flat, j_flat = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    i_flat = i_flat.ravel()
    j_flat = j_flat.ravel()
    positions = np.stack([x_pos[i_flat], y_pos[j_flat]], axis=1)

    # Validate: finite, and the MC volume must be positive with box area > 1e-9.
    if not np.all(np.isfinite(positions)):
        keich = _keich_init(_N)
        return np.stack([keich[i_flat], keich[j_flat]], axis=1)

    try:
        V = _mc_volume(positions)
        if not np.isfinite(V) or V <= 0.0:
            keich = _keich_init(_N)
            return np.stack([keich[i_flat], keich[j_flat]], axis=1)
    except Exception:
        pass

    return positions


def _mc_volume(positions):
    """Reproduce the evaluator's fixed-sample Monte Carlo volume estimate."""
    N = _N
    k = np.arange(N * N)
    i = (k // N).astype(np.float64)
    j = (k % N).astype(np.float64)
    xk = positions[:, 0]
    yk = positions[:, 1]
    X0 = float(xk.min())
    X1 = float((xk + (i + 1) / N).max())
    Y0 = float(yk.min())
    Y1 = float((yk + (j + 1) / N).max())
    area = (X1 - X0) * (Y1 - Y0)
    if area <= 1e-9:
        return 0.0
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, 500000)
    b = rng.uniform(Y0, Y1, 500000)
    c = rng.uniform(0.0, 1.0, 500000)
    inside = np.zeros(a.shape, dtype=bool)
    for kk in range(N * N):
        ix = xk[kk] + c * i[kk] / N
        iy = yk[kk] + c * j[kk] / N
        inside |= (a >= ix) & (a <= ix + _S) & (b >= iy) & (b <= iy + _S)
    return float(inside.mean()) * area


# EVOLVE-BLOCK-END

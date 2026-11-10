# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_LIMIT = 600.0
_MARGIN = 40.0


def _ok():
    return (time.time() - _START) < (_LIMIT - _MARGIN)


def _union_len_np(lo, hi):
    """Length of the union of intervals [lo[t], hi[t]]."""
    order = np.argsort(lo)
    lo = lo[order]
    hi = hi[order]
    total = 0.0
    cl = lo[0]
    ch = hi[0]
    for t in range(1, len(lo)):
        if lo[t] > ch:
            total += ch - cl
            cl = lo[t]
            ch = hi[t]
        elif hi[t] > ch:
            ch = hi[t]
    total += ch - cl
    return total


def _exact_volume(f, g, n_c=256):
    """Exact 3-D union volume for product structure x=f(i), y=g(j)."""
    N = len(f)
    idx = np.arange(N)
    cs = np.linspace(0.0, 1.0, n_c)
    vals = np.empty(n_c)
    for k in range(n_c):
        c = cs[k]
        lo_x = f + c * idx / N
        hi_x = f + (1.0 + idx) / N
        lo_y = g + c * idx / N
        hi_y = g + (1.0 + idx) / N
        vals[k] = _union_len_np(lo_x, hi_x) * _union_len_np(lo_y, hi_y)
    h = 1.0 / (n_c - 1)
    return h * (0.5 * vals[0] + vals[1:-1].sum() + 0.5 * vals[-1])


def _keich(N):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(bin(idx)[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    keich = _keich(N)

    # Objective over 16 params: first 8 = f (x by i), next 8 = g (y by j).
    def obj(fg):
        return _exact_volume(fg[:N], fg[N:])

    best = np.concatenate([keich, keich])
    best_v = obj(best)

    rng = np.random.default_rng(0)
    # Random restarts with Nelder-Mead; keep best.
    for restart in range(40):
        if not _ok():
            break
        if restart == 0:
            x0 = best.copy()
        else:
            x0 = best + rng.normal(0.0, 0.05, size=2 * N)
        try:
            from scipy.optimize import minimize
            res = minimize(obj, x0, method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
            if res.success or res.fun < best_v:
                if np.all(np.isfinite(res.x)):
                    best = res.x
                    best_v = res.fun
        except Exception:
            pass

    # Final safety: if optimization produced nonsense, fall back to Keich.
    f = best[:N]
    g = best[N:]
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(g))):
        f = keich
        g = keich

    # Guard: ensure a non-degenerate box.
    x_lo = np.min(f)
    x_hi = np.max(f + (1.0 + np.arange(N)) / N)
    y_lo = np.min(g)
    y_hi = np.max(g + (1.0 + np.arange(N)) / N)
    if (x_hi - x_lo) <= 1e-9 or (y_hi - y_lo) <= 1e-9:
        f = keich
        g = keich

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    base_x = f[i_grid.flatten()]
    base_y = g[j_grid.flatten()]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END

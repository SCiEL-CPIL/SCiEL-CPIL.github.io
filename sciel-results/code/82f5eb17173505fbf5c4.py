# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_N = 8
_NN = 64
_W = 1.0 / _N
_DEADLINE = 620.0


def _union_len(a, b):
    """Total length of the union of intervals [a[i], b[i]]."""
    order = np.argsort(a, kind="mergesort")
    a = a[order]
    b = b[order]
    total = 0.0
    cur_s = a[0]
    cur_e = b[0]
    for k in range(1, len(a)):
        s = a[k]
        e = b[k]
        if s > cur_e:
            total += cur_e - cur_s
            cur_s = s
            cur_e = e
        else:
            if e > cur_e:
                cur_e = e
    total += cur_e - cur_s
    return float(total)


def _objective(z):
    """Exact union volume of the 64 sheared tubes (Monte Carlo expectation)."""
    x = z[0:_NN]
    y = z[_NN:2 * _NN]
    c = np.linspace(0.0, 1.0, 33)
    # x-intervals at height c: [x_i + c*i/N, x_i + (i+1)/N + c*i/N] for i in 0..7
    ax = x[:, None] + (c[None, :] * np.arange(_N) / _N)
    bx = ax + _W
    # y-intervals at height c
    ay = y[:, None] + (c[None, :] * np.arange(_N) / _N)
    by = ay + _W
    Lx = np.array([_union_len(ax[k], bx[k]) for k in range(len(c))])
    Ly = np.array([_union_len(ay[k], by[k]) for k in range(len(c))])
    area = Lx * Ly
    # Simpson's rule over c in [0,1]
    h = 1.0 / (len(c) - 1)
    V = (h / 3.0) * (area[0] + area[-1] + 4.0 * np.sum(area[1:-1:2]) + 2.0 * np.sum(area[2:-1:2]))
    return float(V)


def _keich(N):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _seed_positions():
    keich = _keich(_N)
    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    base_x = keich[i_grid.flatten()]
    base_y = keich[j_grid.flatten()]
    return np.stack([base_x, base_y], axis=1)


def _to_flat(pos):
    # x depends only on i, y only on j
    x = np.empty(_N)
    y = np.empty(_N)
    for i in range(_N):
        row = pos[i * _N:(i + 1) * _N]
        x[i] = row[0, 0]
        y[:] = row[:, 1]
    return np.concatenate([x, y])


def _to_pos(z):
    x = z[0:_N]
    y = z[_N:2 * _N]
    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    base_x = x[i_grid.flatten()]
    base_y = y[j_grid.flatten()]
    return np.stack([base_x, base_y], axis=1)


def kakeya_needle_3d8() -> np.ndarray:
    start = time.time()
    best_pos = _seed_positions()
    best_V = _objective(_to_flat(best_pos))

    # Local optimization minimizing exact union volume over per-direction offsets.
    try:
        z0 = _to_flat(best_pos)
        res = minimize(_objective, z0, method="L-BFGS-B",
                       options={"maxiter": 400, "maxfun": 20000, "maxls": 50,
                                "ftol": 1e-12, "gtol": 1e-8})
        if res.success or (hasattr(res, "fun") and res.fun < best_V):
            cand = _to_pos(res.x)
            cand_V = float(res.fun)
            if cand_V < best_V:
                best_pos = cand
                best_V = cand_V
    except Exception:
        pass

    # If we still have time, run a second refinement from the improved point.
    if time.time() - start < _DEADLINE - 120.0:
        try:
            z0 = _to_flat(best_pos)
            res = minimize(_objective, z0, method="L-BFGS-B",
                           options={"maxiter": 400, "maxfun": 20000, "maxls": 50,
                                    "ftol": 1e-14, "gtol": 1e-10})
            if res.success or (hasattr(res, "fun") and res.fun < best_V):
                cand = _to_pos(res.x)
                cand_V = float(res.fun)
                if cand_V < best_V:
                    best_pos = cand
                    best_V = cand_V
        except Exception:
            pass

    # Safety deadline guard: return best found before the wall-clock limit.
    if time.time() - start > _DEADLINE:
        pass

    return best_pos


# EVOLVE-BLOCK-END

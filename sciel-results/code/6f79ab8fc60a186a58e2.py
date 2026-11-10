# EVOLVE-BLOCK-START
import time

import numpy as np


def _kakeya_1d_union_length(x, N, n=20000):
    """Union length of the N sheared 1-D intervals for base positions x.

    Interval i at height c: [x_i + c*i/N, x_i + 1/N + c*i/N].
    Vectorised Monte-Carlo estimate over a fine 1-D sample of c.
    """
    c = np.linspace(0.0, 1.0, n)
    lo = x[None, :] + c[:, None] * (np.arange(N)[None, :] / N)
    hi = lo + 1.0 / N
    order = np.argsort(lo, axis=1)
    lo_s = np.take_along_axis(lo, order, axis=1)
    hi_s = np.take_along_axis(hi, order, axis=1)
    start = lo_s[:, 0]
    cur = hi_s[:, 0]
    for k in range(1, N):
        nxt = hi_s[:, k]
        cur = np.maximum(cur, nxt)
    return float(np.mean(cur - start))


def _kakeya_1d_volume(x, N):
    """2-D volume of the union of the N sheared rectangles (base x_i, width 1/N)."""
    return _kakeya_1d_union_length(x, N) * (1.0 / N)


def _nelder_mead(f, x0, max_iter=60, step=0.03, alpha=1.0, gamma=2.0, rho=0.5,
                 sigma=2.0, tol=1e-8):
    """Compact Nelder-Mead minimiser for a scalar function of a 1-D vector."""
    n = len(x0)
    simplex = [list(x0)]
    f0 = f(x0)
    for i in range(n):
        v = list(x0)
        v[i] = v[i] + step
        simplex.append(v)
    fv = [f0] + [f(np.array(v)) for v in simplex[1:]]
    for _ in range(max_iter):
        order = np.argsort(fv)
        simplex = [simplex[k] for k in order]
        fv = [fv[k] for k in order]
        if abs(fv[-1] - fv[0]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - np.array(simplex[-1]))
        fr = f(xr)
        if fr < fv[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1], fv[-1] = list(xe), fe
            else:
                simplex[-1], fv[-1] = list(xr), fr
        elif fr < fv[-2]:
            simplex[-1], fv[-1] = list(xr), fr
        else:
            xc = centroid + rho * (np.array(simplex[-1]) - centroid)
            fc = f(xc)
            if fc < fv[-1]:
                simplex[-1], fv[-1] = list(xc), fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][k] + sigma * (simplex[i][k] - simplex[0][k])
                                  for k in range(n)]
                    fv[i] = f(np.array(simplex[i]))
    best = np.argmin(fv)
    return np.array(simplex[best]), fv[best]


def _optimize_1d(N, n_mc, deadline, seed):
    """Deterministic multi-restart local search for the 1-D sheared-interval layout."""
    rng = np.random.default_rng(seed)
    f = lambda x: _kakeya_1d_union_length(x, N, n_mc)

    # Initial layout: staircase x_i = (N-1-i)/N (span 1 + (N-1)/N).
    base = np.array([(N - 1 - i) / N for i in range(N)], dtype=np.float64)

    best_x = base.copy()
    best_v = f(best_x)

    # (1) Coordinate descent
    for _ in range(400):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            x = best_x.copy()
            x[i] -= 0.02
            v = f(x)
            if v < best_v - 1e-12:
                best_x, best_v = x, v
                improved = True
            x = best_x.copy()
            x[i] += 0.02
            v = f(x)
            if v < best_v - 1e-12:
                best_x, best_v = x, v
                improved = True
        if not improved:
            break

    # (2) Nelder-Mead polish
    if time.time() <= deadline:
        cand, cv = _nelder_mead(f, best_x, max_iter=80)
        if cv < best_v - 1e-12:
            best_x, best_v = cand, cv

    # (3) Multi-restart with random perturbations
    for _ in range(40):
        if time.time() > deadline:
            break
        x0 = best_x + rng.normal(0.0, 0.05, size=N)
        cand, cv = _nelder_mead(f, x0, max_iter=50)
        if cv < best_v - 1e-12:
            best_x, best_v = cand, cv

    return best_x, best_v


def kakeya_needle_3d8() -> np.ndarray:
    """Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).

    Returns base positions (x_k, y_k) of shape (64, 2). Row k is tube (i, j)
    with i = k // 8, j = k % 8. The 2-D union volume equals the product of two
    1-D union lengths (one per coordinate), so each is optimised independently.
    A deadline guard returns the best layout found before the wall-clock limit.
    """
    N = 8
    start = time.time()
    deadline = start + 540.0  # safety margin under the 600 s cap

    # Baseline staircase layout (always valid).
    base = np.array([(N - 1 - i) / N for i in range(N)], dtype=np.float64)

    # Try the improved 1-D layout; fall back to baseline if the search is cut short.
    try:
        x1d, _ = _optimize_1d(N, n_mc=20000, deadline=deadline, seed=12345)
        if not np.all(np.isfinite(x1d)):
            x1d = base
    except Exception:
        x1d = base

    # Product construction: x depends only on i, y only on j.
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    positions = np.stack([x1d[i_grid], x1d[j_grid]], axis=1).reshape(-1, 2)

    if not np.all(np.isfinite(positions)):
        positions = np.stack([base[i_grid], base[j_grid]], axis=1).reshape(-1, 2)

    return positions


# EVOLVE-BLOCK-END

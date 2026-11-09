# EVOLVE-BLOCK-START
import time
import numpy as np


def _exact_volume(pos, n=120):
    """Exact (grid-approximated) union volume of the 64 sheared tubes.

    The union volume equals the integral over c in [0,1] of the 2-D union area
    A(c) of the N^2 sheared squares at height c. We integrate A(c) over a uniform
    c-grid; since each square's x-interval depends only on (i,c) and its y-interval
    only on (j,c), the 2-D union area at fixed c is exactly the product of the 1-D
    x-union length and the 1-D y-union length (product-of-unions identity).
    """
    N = 8
    pos = np.asarray(pos, dtype=np.float64)
    i = (np.arange(N * N) // N).astype(np.float64)
    j = (np.arange(N * N) % N).astype(np.float64)
    base_x = pos[:, 0]
    base_y = pos[:, 1]
    w = 1.0 / N

    c = (np.arange(n) + 0.5) / n
    cc = c[:, None]                                   # (n,1)
    sx = base_x[None, :] + cc * i[None, :]            # (n, 64)
    sy = base_y[None, :] + cc * j[None, :]            # (n, 64)

    def union_len(s, w):
        lo = s
        hi = s + w
        order = np.argsort(lo, axis=1)
        lo = np.take_along_axis(lo, order, axis=1)
        hi = np.take_along_axis(hi, order, axis=1)
        cov = np.zeros_like(lo)
        cur = -np.inf
        for t in range(s.shape[1]):
            h = hi[:, t]
            if h > cur:
                cov += np.maximum(h - np.maximum(lo[:, t], cur), 0.0)
                cur = h
        return cov

    Lx = union_len(sx, w)
    Ly = union_len(sy, w)
    return float(np.sum(Lx * Ly) / n)


def _nm_minimize(f, x0, maxiter, maxfev, deadline):
    """Deterministic Nelder-Mead with a wall-clock deadline guard."""
    x = np.array(x0, dtype=np.float64)
    n = x.size
    simplex = [x.copy()]
    for k in range(1, n + 1):
        v = x.copy()
        v[(k - 1) % n] += 0.05
        simplex.append(v)
    vals = [f(s) for s in simplex]
    fev = n + 1
    best = min(vals)
    best_x = simplex[int(np.argmin(vals))]
    for _ in range(maxiter):
        if time.time() >= deadline or fev >= maxfev:
            break
        order = np.argsort(vals)
        simplex = [simplex[k] for k in order]
        vals = [vals[k] for k in order]
        xc = np.mean(simplex[:-1], axis=0)
        xr = xc + (xc - simplex[-1])
        fr = f(xr); fev += 1
        if fr < vals[0]:
            xs = xc + 2.0 * (xr - xc)
            fs = f(xs); fev += 1
            if fs < fr:
                simplex[-1] = xs; vals[-1] = fs
            else:
                simplex[-1] = xr; vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr; vals[-1] = fr
        else:
            xc = np.mean(simplex[:-1], axis=0)
            xc2 = xc + 0.5 * (simplex[-1] - xc)
            fc = f(xc2); fev += 1
            if fc < vals[-1]:
                simplex[-1] = xc2; vals[-1] = fc
            else:
                for k in range(1, n + 1):
                    simplex[k] = 0.5 * (simplex[0] + simplex[k])
                    vals[k] = f(simplex[k]); fev += 1
        m = int(np.argmin(vals))
        if vals[m] < best:
            best = vals[m]; best_x = simplex[m].copy()
    return best_x, best


def _refine(pos, deadline, maxiter, maxfev):
    """Local refinement of a 64x2 layout, returning the best layout found."""
    x0 = pos.reshape(-1).copy()
    best_x, best_v = _nm_minimize(_exact_volume, x0, maxiter, maxfev, deadline)
    if time.time() < deadline - 2.0:
        # second pass from the refined point to escape shallow local minima
        x2, v2 = _nm_minimize(_exact_volume, best_x, maxiter, maxfev, deadline)
        if v2 < best_v:
            best_x, best_v = x2, v2
    return best_x.reshape(64, 2), best_v


def kakeya_needle_3d8() -> np.ndarray:
    """64 sheared-tube Kakeya layout minimizing the union volume (score ~ 1/V).

    Breaks the baseline's product structure by optimizing the full 2-D base
    layout with a deterministic multi-start Nelder-Mead pipeline on the exact
    union volume, guarded by a wall-clock deadline.
    """
    start = time.time()
    deadline = start + 540.0          # safety margin well inside the 600 s limit
    N = 8

    # --- baseline product layout (Keich binary construction) as a safe seed ---
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    ig, jg = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    base_x = keich[ig.flatten()]
    base_y = keich[jg.flatten()]
    best = np.stack([base_x, base_y], axis=1).copy()
    best_v = _exact_volume(best)

    # --- deterministic multi-start over a family of non-product layouts ---
    starts = [best.reshape(-1).copy()]
    rng = np.random.default_rng(0)
    for s in range(1, 8):
        starts.append(rng.uniform(-1.0, 1.5, size=128))
    for idx in range(N):
        starts.append(keich[idx] * np.ones(128))   # constant-x rows

    for x0 in starts:
        if time.time() >= deadline - 30.0:
            break
        x, v = _refine(x0.reshape(64, 2), deadline, maxiter=350, maxfev=900)
        if v < best_v:
            best_v = v
            best = x.copy()

    # --- final polish on the best layout with the remaining budget ---
    if time.time() < deadline - 5.0:
        x, v = _refine(best, deadline, maxiter=250, maxfev=700)
        if v < best_v:
            best_v = v
            best = x.copy()

    # --- sanity guard: fall back to the baseline if anything went wrong ---
    if (not np.all(np.isfinite(best))) or best_v <= 0.0:
        return np.stack([base_x, base_y], axis=1)
    return best


# EVOLVE-BLOCK-END

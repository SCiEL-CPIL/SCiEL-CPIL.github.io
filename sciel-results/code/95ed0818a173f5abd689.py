import numpy as np
import time


def _keich(n=64):
    """Classical Keich construction (n a power of 2): a good convex start."""
    k = int(round(np.log2(n)))
    j = np.arange(1, k + 1)
    pos = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bits = list(bin(i)[2:])
        bits = [0] * (k - len(bits)) + bits
        eps = np.array(bits, dtype=np.float64)
        pos[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return pos


def _area_grad(x, n, M, a, y, w):
    """Exact union-area (mean over heights) and its subgradient w.r.t. x."""
    # L_i(y) = x_i + a_i * y   -> (M, n)
    L = x[None, :] + a[None, :] * y[:, None]
    order = np.argsort(L, axis=1)
    Ls = np.take_along_axis(L, order, axis=1)          # sorted left endpoints
    d = np.diff(Ls, axis=1)                            # (M, n-1) consecutive gaps
    mn = np.minimum(d, w[:, None])
    area = (w + mn.sum(axis=1)).mean()                 # mean union length
    # subgradient of min(d_j, w): 1 if d_j < w else 0
    s = (d < w[:, None]).astype(np.float64)
    # dU/dL_p (sorted position p) = s_{p-1} - s_p
    gL = np.zeros((M, n), dtype=np.float64)
    gL[:, 1:] += s
    gL[:, :n - 1] -= s
    # scatter from sorted position back to original index
    gx = np.zeros((M, n), dtype=np.float64)
    np.put_along_axis(gx, order, gL, axis=1)
    grad = gx.mean(axis=0)                             # (n,)
    return area, grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 600.0                              # safety margin vs 630 s kill
    n = 64
    M = 20000
    a = (np.arange(n) + 1) / n
    m = np.arange(M)
    y = (m + 0.5) / M
    w = (1.0 - y) / n

    def run(x0, budget, c=3.0):
        x = np.asarray(x0, dtype=np.float64).copy()
        sum_x = np.zeros(n, dtype=np.float64)
        best_x = x.copy()
        best_area, _ = _area_grad(x, n, M, a, y, w)
        k = 0
        while time.time() < budget:
            k += 1
            area, grad = _area_grad(x, n, M, a, y, w)
            if area < best_area - 1e-15:
                best_area, best_x = area, x.copy()
            sum_x += x
            if k % 200 == 0:                            # Polyak average checkpoint
                avg = sum_x / k
                aa, _ = _area_grad(avg, n, M, a, y, w)
                if aa < best_area - 1e-15:
                    best_area, best_x = aa, avg.copy()
            t = min(c / np.sqrt(k), 1.0)
            x = np.clip(x - t * grad, -5.0, 5.0)
        return best_x, best_area

    try:
        # Keich start gets the bulk of the budget (it is already near-optimal).
        bx, ba = run(_keich(n), deadline - 45.0, c=3.0)
        best_x, best_area = bx, ba

        # A few random restarts with the remaining time (convex -> same min, but
        # different finite-horizon landing points; keep the best).
        rng = np.random.default_rng(12345)
        for _ in range(3):
            if time.time() > deadline - 20.0:
                break
            bx2, ba2 = run(rng.uniform(-1.0, 1.0, n), deadline - 20.0, c=3.0)
            if ba2 < best_area:
                best_x, best_area = bx2, ba2

        # Final low-step polish from the current best to squeeze a little more.
        if time.time() < deadline - 5.0:
            bx3, ba3 = run(best_x, deadline - 5.0, c=0.5)
            if ba3 < best_area:
                best_x, best_area = bx3, ba3

        # Validate before returning.
        fa, _ = _area_grad(best_x, n, M, a, y, w)
        if (not np.isfinite(fa)) or fa <= 0 or (not np.all(np.isfinite(best_x))):
            best_x = _keich(n)
        return best_x.astype(np.float64)
    except Exception:
        return _keich(n).astype(np.float64)


# EVOLVE-BLOCK-END

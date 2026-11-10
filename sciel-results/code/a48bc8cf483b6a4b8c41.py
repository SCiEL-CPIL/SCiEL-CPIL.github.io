# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_FRACTION = 0.90  # safety margin
_START_TIME = None


def _keich_positions(n=64):
    """Classical Keich construction for n a power of 2."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return positions


def _union_area_and_subgrad(x, n=64, n_y=20000):
    """
    Compute the quadrature union area and its subgradient w.r.t. x.
    x: shape (n,)
    Returns (area, subgrad) where subgrad has shape (n,).
    """
    a = (np.arange(n, dtype=np.float64) + 1.0) / n  # a_i
    w = 1.0 / n
    y = (np.arange(n_y, dtype=np.float64) + 0.5) / n_y  # (n_y,)

    # L matrix: L[i, m] = x_i + a_i * y_m  -> shape (n, n_y)
    L = x[:, None] + a[:, None] * y[None, :]  # (n, n_y)

    # Sort L along axis 0 for each y
    order = np.argsort(L, axis=0)  # (n, n_y)
    Ls = np.take_along_axis(L, order, axis=0)  # (n, n_y) sorted

    # gaps between consecutive sorted L
    gaps = Ls[1:, :] - Ls[:-1, :]  # (n-1, n_y)
    below = gaps < w  # (n-1, n_y)

    # subgradient of union length w.r.t. each sorted position
    # For each pair p (between sorted pos p and p+1), if gap < w:
    #   dU/dL_sorted[p] += -1, dU/dL_sorted[p+1] += +1
    # Build dU/dL_sorted: shape (n, n_y)
    dL_sorted = np.zeros_like(Ls)
    dL_sorted[:-1, :] += -1.0 * below
    dL_sorted[1:, :] += 1.0 * below

    # Map back to original index i: dU/dx_i = dU/dL_i
    dL_orig = np.zeros_like(L)
    np.put_along_axis(dL_orig, order, dL_sorted, axis=0)

    subgrad = dL_orig.mean(axis=1)  # mean over y -> (n,)

    # union length per y
    union_len = w + np.sum(np.minimum(gaps, w), axis=0)  # (n_y,)
    area = union_len.mean()

    return area, subgrad


def kakeya_needle_2d_positions64() -> np.ndarray:
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    n = 64
    n_y = 20000
    deadline = _START_TIME + 630.0 * _DEADLINE_FRACTION

    x = _keich_positions(n)
    best_x = x.copy()
    best_area = _union_area_and_subgrad(x, n, n_y)[0]

    # Adaptive subgradient step
    # Initial step size heuristic: scale by typical x range / iterations
    step = 0.05
    min_step = 1e-7
    max_iter = 200000

    for it in range(max_iter):
        if time.time() > deadline:
            break

        area, subgrad = _union_area_and_subgrad(x, n, n_y)
        if area < best_area:
            best_area = area
            best_x = x.copy()

        # Normalize subgradient for stable step
        sg_norm = np.linalg.norm(subgrad)
        if sg_norm < 1e-12:
            break
        direction = subgrad / sg_norm

        # Try a few step sizes (line search over a few scales)
        improved = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            cand = x - step * scale * direction
            cand_area, _ = _union_area_and_subgrad(cand, n, n_y)
            if cand_area < area:
                x = cand
                improved = True
                break

        if not improved:
            step *= 0.5
            if step < min_step:
                break
        else:
            # occasionally grow step back
            if it % 50 == 0 and step < 0.05:
                step = min(0.05, step * 1.5)

    # Final check: return best
    if np.any(~np.isfinite(best_x)) or best_area <= 0:
        return _keich_positions(n)
    return best_x


# EVOLVE-BLOCK-END

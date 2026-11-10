# EVOLVE-BLOCK-START
import numpy as np


def _entropy_of_projection(P, r):
    """Shannon entropy (bits) of the projected distribution x + r*y.

    P is indexed as P[y, x]. Zero-probability outcomes are ignored.
    """
    n = P.shape[0]
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    vals = xs[None, :] + r * ys[:, None]          # shape (n, n), vals[y, x]
    # Pool by value (values are integers here, so exact grouping is safe).
    uniq, inv = np.unique(vals, return_inverse=True)
    pooled = np.bincount(inv.ravel(), weights=P.ravel(), minlength=len(uniq))
    pooled = pooled[pooled > 0.0]
    return float(-(pooled * np.log2(pooled)).sum())


def _ratio(P):
    D = _entropy_of_projection(P, -1.0)
    M = max(_entropy_of_projection(P, 0.0),
            _entropy_of_projection(P, 1.0),
            _entropy_of_projection(P, 2.0))
    if M <= 1e-9:
        return -np.inf
    return D / M


def _refine(P, iters=600, step=0.02):
    """Coordinate-ascent style local search on log-weights to raise the ratio."""
    n = P.shape[0]
    w = np.log(P + 1e-300).copy()
    best = P.copy()
    best_r = _ratio(P)
    for _ in range(iters):
        # pick a random cell, perturb its log-weight, renormalize
        i, j = np.random.randint(n), np.random.randint(n)
        old = w[i, j]
        w[i, j] = old + np.random.uniform(-step, step)
        expw = np.exp(w - w.max())
        P2 = expw / expw.sum()
        r2 = _ratio(P2)
        if r2 > best_r + 1e-12:
            best_r = r2
            best = P2.copy()
        else:
            w[i, j] = old
    return best


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Construct a 10x10 joint distribution P[y, x] maximizing the arithmetic
    Kakeya entropy ratio D / M where D = H(x - y) and
    M = max(H(x), H(x + y), H(x + 2y)).

    Approach: place two tight 4x4 clusters in the top-left and bottom-right
    corners. Within each cluster x and y are concentrated (low marginal
    entropies), but the two clusters lie on widely separated diagonals so the
    x - y projection spreads over many values (high diagonal entropy). A short
    local refinement pass then nudges the ratio higher.
    """
    n = 10
    P = np.zeros((n, n), dtype=np.float64)

    # Cluster A: top-left 4x4 (rows/cols 0..3)
    P[0:4, 0:4] = 1.0
    # Cluster B: bottom-right 4x4 (rows/cols 6..9)
    P[6:10, 6:10] = 1.0

    P = P / P.sum()

    # Local refinement to squeeze a little more ratio out.
    np.random.seed(0)
    P = _refine(P, iters=400, step=0.015)

    # Safety: clip negatives, renormalize, ensure validity.
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / s

    return P


# EVOLVE-BLOCK-END

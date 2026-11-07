# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Build a heuristic matrix H where H[i, j] rewards placing item j after item i
    in the same bin by measuring how "dead" the residual gap becomes.

    For each ordered pair (i, j):
      - If d_i + d_j > capacity, H[i,j] = 1e-9 (cannot coexist).
      - Otherwise, residual = capacity - d_i - d_j.
        We count how many items k (k != i, k != j) satisfy d_k <= residual.
        A smaller count means the residual is too small for most remaining items
        (a "dead gap"), which is desirable.
        H[i,j] = 1 / (count + 1)
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)

    # Pair sums and residuals
    pair_sum = demands[:, None] + demands[None, :]  # (n, n)
    residual = capacity - pair_sum                    # (n, n)

    # For each pair (i, j), count items k (k != i, k != j) with d_k <= residual[i, j].
    # We'll compute count_all[i, j] = #{k : d_k <= residual[i, j]} first,
    # then subtract 1 if d_i <= residual[i, j] (since k != i),
    # and subtract 1 if d_j <= residual[i, j] (since k != j).

    # count_all: (n, n, n) -> sum over axis 2 gives (n, n)
    # But n=120, so 120^3 = 1.728M booleans, fine.
    fits_all = (residual[:, :, None] >= demands[None, None, :])  # (n, n, n)
    count_all = fits_all.sum(axis=2)  # (n, n)

    # Subtract self-contributions
    # d_i <= residual[i, j]?  -> demands[i] <= residual[i, j]
    # This is a boolean (n, n) where row i, col j
    self_i = (demands[:, None] <= residual)   # (n, n): True if d_i <= residual[i,j]
    self_j = (demands[None, :] <= residual)   # (n, n): True if d_j <= residual[i,j]

    count = count_all - self_i.astype(float) - self_j.astype(float)
    count = np.maximum(count, 0.0)  # guard against numerical issues

    # Heuristic: inverse of (count + 1)
    heuristic = 1.0 / (count + 1.0)

    # Pairs that don't fit get minimal value
    heuristic = np.where(residual >= 0, heuristic, 1e-9)

    # Safety: ensure positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END

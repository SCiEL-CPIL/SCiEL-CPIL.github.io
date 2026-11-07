"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Safety margin (seconds) before the 330 s wall-clock kill, so we always
# return a valid matrix even under unexpected slowness.
_DEADLINE_MARGIN_S = 5.0


def heuristics(demands, capacity):
    """
    Computes a heuristic matrix for the ant colony optimizer.

    Strategy (combined family):
      Primary driver  -- tight packing: 1/(remaining + 1) strongly rewards
                         pairs that fill the bin with little leftover space,
                         which directly reduces fragmentation and bin count.
      Secondary driver -- normalized pair sum: pair_sum / capacity provides a
                         smooth baseline gradient so that even moderate fits
                         (where the inverse-remaining term is flat) still carry
                         a consistent preference for larger, more useful items.

    eta[i, j] = 1/(remaining + 1) + 0.5 * (pair_sum / capacity)  if pair fits
    eta[i, j] = 1e-9                                              otherwise
    """
    start_time = time.time()

    d = np.asarray(demands, dtype=np.float64)
    n = len(d)

    # Pair sums and fit mask
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity

    # Remaining space for fitting pairs (large sentinel elsewhere)
    remaining = np.where(fits, capacity - pair_sums, capacity)

    # Build the matrix, defaulting to the non-positive floor of 1e-9
    eta = np.full((n, n), 1e-9, dtype=np.float64)

    if np.any(fits):
        tight_reward = 1.0 / (remaining[fits] + 1.0)
        normalized_sum = pair_sums[fits] / capacity
        eta[fits] = tight_reward + 0.5 * normalized_sum

    # Enforce the contract: any non-positive entry becomes 1e-9
    eta[eta <= 0.0] = 1e-9

    # Guard against non-finite values (defensive; should not occur)
    if not np.all(np.isfinite(eta)):
        eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
        eta[eta <= 0.0] = 1e-9

    # Deadline guard: the computation above is O(n^2) and finishes in
    # microseconds, but if something unexpected stalls, we still return a
    # valid finite matrix with a safety margin before the wall-clock limit.
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE_MARGIN_S:
        # The matrix is already fully computed and finite at this point;
        # nothing further to do. This branch is purely a safety net.
        pass

    return eta
# EVOLVE-BLOCK-END

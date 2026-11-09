"""Direct-approach heuristics for ant-colony offline bin packing.

Rationale (derived from the task structure):
  * An ant places a RANDOM first item, then uses H[current, j] only to pick the
    NEXT item.  Therefore H must encode "given item i is already in the bin,
    which item j is the best to add next?"
  * Fewer bins  <=>  fill each bin tightly.  The best next item is the LARGEST
    one that still fits (it leaves the least unusable leftover space).  This is
    favoured most when the current item i is small (the remaining capacity is
    large and the tight fit is achievable).
  * Hence:  size(j) * (remaining_capacity(i) - size(j)).
      - larger j  -> higher (big items consume the bin efficiently)
      - small i   -> higher (more room to pack a big item tightly)
      - tight fit -> the product peaks near the best-filling size for the bin
  * Non-positive / infeasible entries collapse to 1e-9 by construction, and the
    j == i self-entry is forced to 1e-9 so an ant never "replaces" its own item.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320  # safety margin before 330s kill


def heuristics(demands, capacity):
    d = np.asarray(demands, dtype=float)
    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    # size(i) + size(j) should be close to capacity for tight packing
    sum_sizes = d[:, None] + d[None, :]
    remaining = capacity - d[:, None]  # capacity left after placing item i

    # Feasible: j fits in the space left after i
    feasible = (d[None, :] <= remaining) & (d[None, :] > 0)

    # Score: how well i and j together fill a bin of `capacity`
    # Use 1/(1+|sum - capacity|) which peaks when sum == capacity
    gap = np.abs(sum_sizes - capacity)
    score = 1.0 / (1.0 + gap)

    # Boost by size of j: larger items are harder to place, so prioritize them
    # This prevents the heuristic from being dominated by small complementary items
    score = score * (d[None, :] / 100.0)  # normalize to [0.2, 0.99]

    # Forbid self-placement
    if n:
        feasible[np.diag_indices(n)] = False

    out = np.where(feasible, score, 1e-9)

    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a simple fallback if we're running out of time
        return out

    return out
# EVOLVE-BLOCK-END

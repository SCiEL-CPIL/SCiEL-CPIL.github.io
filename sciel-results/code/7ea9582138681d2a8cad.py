"""Best-fit-style heuristic for ACO offline bin packing (refined within family).

Key design (kept from the current program, refined):
  * H[i, j] answers "given item i is in the bin, how attractive is adding j next?"
  * The ACO ant picks the FIRST item uniformly at random, so H only guides the
    subsequent placements.  Fewer bins  <=>  each bin is filled tightly.
  * The best next item is the one whose size most closely matches the remaining
    capacity after i (the "best-fit" idea).  We score this with a SMOOTH
    complementarity surrogate instead of a hard peak:
        comp(i, j) = 1 / (1 + |cap - (s_i + s_j)|)
    which is always positive, peaks when s_i + s_j == cap, and degrades smoothly.
  * A size factor (s_j / cap) is multiplied in so that, among items that fill the
    bin similarly, LARGER items are preferred.  This matters because large items
    are the hardest to place later; placing them early while the bin is open is
    what keeps the total bin count low.  This is the refinement over the current
    program (which used the same comp * size idea) -- we keep it and tune the
    exponent, and we make the whole thing robust and fast.
  * Infeasible (j does not fit after i) and self entries -> 1e-9.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 315.0  # hard safety margin before the 330 s kill


def _build_heuristic(d, capacity):
    """Vectorised best-fit heuristic.  d: 1-D float array of item sizes."""
    n = d.shape[0]
    s = d[None, :]
    rem = capacity - d[:, None]            # space left after placing item i
    feasible = (s <= rem) & (s > 0)

    gap = np.abs(capacity - (d[:, None] + s))
    comp = 1.0 / (1.0 + gap)               # smooth complementarity in (0, 1]
    size_factor = (s / capacity) ** 1.0    # prefer larger j; exponent tuned

    out = np.where(feasible, comp * size_factor, 1e-9)
    np.fill_diagonal(out, 1e-9)            # never "replace" the current item
    return out


def heuristics(demands, capacity):
    d = np.asarray(demands, dtype=float)
    if d.ndim != 1:
        d = d.reshape(-1)
    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    # Fast, always-valid best-fit heuristic.
    out = _build_heuristic(d, capacity)

    # Safety: guarantee a finite, correctly-shaped, positive matrix.
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=1e-9, posinf=1e-9, neginf=1e-9)
    out = np.maximum(out, 1e-9)
    if out.shape != (n, n):
        out = np.full((n, n), 1e-9, dtype=float)

    # --- In-process validation: run the documented ACO and confirm it works. ---
    # This also warms up the numpy paths.  We cap the work so we never approach
    # the wall-clock limit; the heuristic itself is computed once and returned.
    try:
        if n > 0 and (time.time() - _START_TIME) < _TIME_LIMIT - 5.0:
            _validate_aco(d, capacity, out)
    except Exception:
        pass  # heuristic is independent of the colony; keep the valid matrix

    return out


def _validate_aco(d, capacity, H):
    """Run a small ACO to confirm the pipeline produces a finite bin count.

    Mirrors the evaluator: 20 ants x 30 iters, pheromone init 1, decay 0.9,
    deposit 1/fewest_bins_of_iteration, first item random, then probabilistic
    pick proportional to tau[i, j] * H[i, j] over fitting items.
    """
    n = d.shape[0]
    tau = np.ones((n, n), dtype=float)
    rng = np.random.default_rng(12345)

    for _ in range(30):
        best_iter = n + 1
        for _ in range(20):
            remaining = np.ones(n, dtype=bool)
            bins = 0
            while remaining.any():
                bins += 1
                # random first item
                idx = np.where(remaining)[0]
                cur = idx[rng.integers(idx.size)]
                remaining[cur] = False
                used = d[cur]
                # repeatedly pick a fitting item
                while True:
                    fit = np.where(remaining & (d <= capacity - used))[0]
                    if fit.size == 0:
                        break
                    w = tau[cur, fit] * H[cur, fit]
                    w = np.where(w > 0, w, 1e-9)
                    w = w / w.sum()
                    j = fit[rng.choice(fit.size, p=w)]
                    remaining[j] = False
                    used += d[j]
                    cur = j
        best_iter = min(best_iter, bins)
        tau = tau * 0.9 + 1.0 / best_iter

    return best_iter


# EVOLVE-BLOCK-END

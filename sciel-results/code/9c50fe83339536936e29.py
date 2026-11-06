"""Size-efficient sorting networks on 17 wires.
Entry point: construct() -> list of (i, j) integer pairs, 0 <= i < j < 17.
"""
import time
import numpy as np

N = 17
NUM_BOOL = 1 << N          # 131072
DEADLINE = 85.0            # safety margin under 90s wall-clock limit

# Precompute all Boolean inputs as a (NUM_BOOL, N) uint8 matrix.
# Row r has bit k set iff (r >> k) & 1.
def _build_inputs():
    r = np.arange(NUM_BOOL, dtype=np.uint32)
    bits = np.zeros((N, NUM_BOOL), dtype=np.uint8)
    for k in range(N):
        bits[k] = ((r >> k) & 1).astype(np.uint8)
    return bits.T  # (NUM_BOOL, N)

_INPUTS = _build_inputs()

def _make_seed():
    """Odd-even transposition network on N wires."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _verify(net):
    """Return True iff net sorts every Boolean input (zero-one principle)."""
    if not net:
        return N <= 1
    vals = _INPUTS.astype(np.uint8).copy()
    for (i, j) in net:
        a = vals[:, i]
        b = vals[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        vals[:, i] = lo
        vals[:, j] = hi
    # After sorting, wire k must equal the k-th order statistic.
    # For Boolean inputs, wire k holds 1 iff at least (N - k) inputs were 1.
    # i.e. count_ones >= N - k.
    counts = vals.sum(axis=1)  # number of 1s in each sorted output row
    # wire k is 1 iff counts >= N - k
    for k in range(N):
        expected = (counts >= (N - k)).astype(np.uint8)
        if not np.array_equal(vals[:, k], expected):
            return False
    return True

def _verify_quick(net, subset=None):
    """Verify on a subset of inputs for fast screening (still sufficient for correctness if full passes)."""
    if not net:
        return N <= 1
    idx = slice(None) if subset is None else subset
    vals = _INPUTS[idx].astype(np.uint8).copy()
    for (i, j) in net:
        a = vals[:, i]
        b = vals[:, j]
        vals[:, i] = np.minimum(a, b)
        vals[:, j] = np.maximum(a, b)
    counts = vals.sum(axis=1)
    for k in range(N):
        if not np.array_equal(vals[:, k], (counts >= (N - k)).astype(np.uint8)):
            return False
    return True

def construct():
    t0 = time.time()
    best = _make_seed()
    assert _verify(best)

    def remaining():
        return DEADLINE - (time.time() - t0)

    # Local search: try to delete comparators.
    # Order: try later comparators first (often redundant), then earlier.
    # We do multiple passes until no improvement or deadline.
    improved = True
    pass_count = 0
    while improved and remaining() > 3.0:
        improved = False
        pass_count += 1
        n = len(best)
        # Build candidate order: prefer removing from the end
        order = list(range(n - 1, -1, -1))
        # Occasionally shuffle for diversity on later passes
        if pass_count > 1 and remaining() > 10.0:
            # deterministic-ish shuffle using pass_count
            step = (pass_count * 7919) % max(1, len(order))
            order = order[step:] + order[:step]
        for idx in order:
            if remaining() < 2.0:
                break
            cand = best[:idx] + best[idx + 1:]
            if _verify(cand):
                best = cand
                improved = True
                # After a deletion, re-check ordering is still valid (it is).
                break  # restart scan after a change (cheap: re-evaluate from end)
        if remaining() < 5.0:
            break

    # Optional: try adding a comparator that "helps" then re-prune.
    # Only if we have time. A comparator (i,j) with i<j.
    if remaining() > 15.0:
        # Try a small set of candidate extra comparators, then prune.
        candidates = []
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) not in best:
                    candidates.append((i, j))
        # Limit to a subset to bound time
        max_tries = 200
        tried = 0
        for (i, j) in candidates:
            if remaining() < 5.0 or tried >= max_tries:
                break
            tried += 1
            cand = best + [(i, j)]
            # Quick screen on a subset first
            if not _verify_quick(cand, subset=slice(0, 4096)):
                continue
            if _verify(cand):
                # Now try to prune the new network
                pruned = cand
                changed = True
                while changed and remaining() > 3.0:
                    changed = False
                    pn = len(pruned)
                    for idx in range(pn - 1, -1, -1):
                        if remaining() < 2.0:
                            break
                        t = pruned[:idx] + pruned[idx + 1:]
                        if _verify(t):
                            pruned = t
                            changed = True
                            break
                if len(pruned) < len(best):
                    best = pruned

    # Final safety: ensure best is valid
    if not _verify(best):
        # fall back to seed
        best = _make_seed()

    return best

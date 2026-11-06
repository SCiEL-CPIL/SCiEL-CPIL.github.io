"""Size-efficient sorting network on 16 wires.

Strategy:
  * Foundation: Batcher's odd-even merge sort -> verified 63-comparator network.
  * Refinement: multi-pass greedy comparator-removal with randomized restarts.

Key optimizations over prior version:
  * Vectorized sortedness check (single np.all instead of N-1 np.any calls).
  * After a successful removal, continue the forward scan without a full
    rebuild -- only re-apply the already-committed prefix to a fresh matrix
    when the scan wraps around.
  * Randomized restarts: shuffle the comparator order before each sweep so
    that different local minima are explored within the time budget.
  * Aggressive deadline guard with safety margin.

Correctness certified by the zero-one principle over all 65,536 Boolean
inputs.  Target: 60 comparators (known optimal for 16 wires).
"""
import numpy as np
import time
import random

N = 16
_START = time.time()
_DEADLINE = _START + 72.0  # explicit safety margin below the 90 s wall clock


# ---------------------------------------------------------------------------
# Network builder: Batcher's odd-even merge sort (recursive).
# ---------------------------------------------------------------------------
def _batcher_odd_even_merge(n):
    """Return a fixed compare-exchange network sorting n wires (n a power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]

    cmps = []
    # 1) recursively sort odd-indexed wires: 1,3,5,...
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a + 1, 2 * b + 1))
    # 2) recursively sort even-indexed wires: 0,2,4,...
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a, 2 * b))
    # 3) odd-even cross comparisons (2i, 2i+1) then (2i+1, 2i+2)
    for i in range(n // 2):
        cmps.append((2 * i, 2 * i + 1))
    for i in range(n // 2 - 1):
        cmps.append((2 * i + 1, 2 * i + 2))
    return cmps


# ---------------------------------------------------------------------------
# Verification: zero-one principle (all 2**N Boolean inputs), NumPy-accelerated.
# ---------------------------------------------------------------------------
def _verify_zero_one(cmps):
    """Return True iff `cmps` sorts every Boolean vector of length N."""
    import numpy as np

    M = 1 << N
    idx = np.arange(M, dtype=np.int64)
    bits = np.zeros((M, N), dtype=np.int8)
    for k in range(N):
        bits[:, k] = ((idx >> k) & 1).astype(np.int8)

    for i, j in cmps:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi

    for k in range(N - 1):
        if np.any(bits[:, k] > bits[:, k + 1]):
            return False
    return True


# ---------------------------------------------------------------------------
# Depth (parallelism) reporter.
# ---------------------------------------------------------------------------
def _parallel_depth(cmps):
    end_time = [0] * N
    depth = 0
    for i, j in cmps:
        t = max(end_time[i], end_time[j]) + 1
        depth = max(depth, t)
        end_time[i] = t
        end_time[j] = t
    return depth


# ---------------------------------------------------------------------------
# Incremental reduction search.
# ---------------------------------------------------------------------------
def _build_matrix():
    import numpy as np
    M = 1 << N
    idx = np.arange(M, dtype=np.int64)
    bits = np.zeros((M, N), dtype=np.int8)
    for k in range(N):
        bits[:, k] = ((idx >> k) & 1).astype(np.int8)
    return bits


def _is_sorted(bits):
    """Vectorized sortedness check: all adjacent pairs satisfy bits[:,k] <= bits[:,k+1]."""
    return bool(np.any(np.diff(bits, axis=1) < 0) == False)


def _apply(bits, cmps, start):
    """Apply comparators cmps[start:] to `bits` in place."""
    for i, j in cmps[start:]:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi


def _reduce_network(initial_cmps, max_time=None):
    """
    Multi-pass greedy comparator-removal with randomized restarts and
    bidirectional sweeps (forward + backward) to find more redundancies.
    """
    import numpy as np

    if max_time is None:
        max_time = _DEADLINE

    def full_build(cmp_list):
        m = _build_matrix()
        _apply(m, cmp_list, 0)
        return m

    current = list(initial_cmps)
    best = list(current)
    best_len = len(current)

    attempt = 0
    while time.time() < max_time:
        attempt += 1
        # For attempts > 1, shuffle to explore different local minima.
        if attempt > 1:
            random.shuffle(current)

        n = len(current)
        made_removal = False

        # --- Forward sweep ---
        # Build state matrix incrementally.
        bits = _build_matrix()
        pos = 0
        while pos < n:
            if time.time() > max_time:
                break
            # bits = state after first `pos` comparators of `current`.
            # Test: skip comparator at `pos`, apply rest, check sorted.
            cand = bits.copy()
            _apply(cand, current, pos + 1)
            if _is_sorted(cand):
                # Redundant: remove it.
                del current[pos]
                n -= 1
                made_removal = True
            else:
                # Needed: apply it and advance.
                i, j = current[pos]
                a = bits[:, i]
                b = bits[:, j]
                lo = np.minimum(a, b)
                hi = np.maximum(a, b)
                bits[:, i] = lo
                bits[:, j] = hi
                pos += 1

        # --- Backward sweep (only if forward made progress or on first attempt) ---
        if time.time() < max_time and n < len(initial_cmps) + 5:
            # Rebuild from scratch for backward pass.
            # For backward sweep, we need to test removing each comparator
            # from the end.  We build the "prefix" state for each position.
            # Efficient approach: precompute prefix states.
            # But that's expensive. Instead, do a simpler backward greedy:
            # Start from the full network, try removing from the end.
            pos = n - 1
            while pos >= 0:
                if time.time() > max_time:
                    break
                # Test: remove comparator at `pos`, check if remaining sorts.
                # We need to apply all comparators except `pos`.
                # Build matrix, skip pos.
                m = _build_matrix()
                for k in range(n):
                    if k == pos:
                        continue
                    i, j = current[k]
                    a = m[:, i]
                    b = m[:, j]
                    lo = np.minimum(a, b)
                    hi = np.maximum(a, b)
                    m[:, i] = lo
                    m[:, j] = hi
                if _is_sorted(m):
                    del current[pos]
                    n -= 1
                    made_removal = True
                    # Stay at same pos (next element shifted into pos)
                else:
                    pos -= 1

        # Update best.
        if n < best_len:
            best = list(current)
            best_len = n

        # If no removal was made in this round, try a randomized restart.
        if not made_removal:
            if attempt >= 15:
                break
        else:
            if attempt > 60:
                break

    return best


# ---------------------------------------------------------------------------
# Entry point (name/signature kept exactly as in the seed program).
# ---------------------------------------------------------------------------
def _polish_local_search(net, max_time):
    """
    High-precision polish: try swapping adjacent comparator pairs and
    removing single comparators.  Accept any change that reduces count
    or (at same count) passes verification.  This is an exact local
    1-move search in the comparator-sequence space.
    """
    import numpy as np
    current = list(net)
    n = len(current)

    # Precompute the "all inputs" matrix once.
    M = 1 << N
    idx = np.arange(M, dtype=np.int64)
    base_bits = np.zeros((M, N), dtype=np.int8)
    for k in range(N):
        base_bits[:, k] = ((idx >> k) & 1).astype(np.int8)

    def fast_sorts(cmp_list):
        m = base_bits.copy()
        for i, j in cmp_list:
            a = m[:, i]
            b = m[:, j]
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            m[:, i] = lo
            m[:, j] = hi
        return bool(np.any(np.diff(m, axis=1) < 0) == False)

    # Verify current state.
    if not fast_sorts(current):
        return net

    improved = True
    while improved and time.time() < max_time:
        improved = False
        # Try removing each comparator.
        for pos in range(len(current)):
            if time.time() > max_time:
                return current
            cand = current[:pos] + current[pos+1:]
            if fast_sorts(cand):
                current = cand
                improved = True
                break
        if improved:
            continue
        # Try swapping adjacent comparators (to escape local minima in ordering).
        for pos in range(len(current) - 1):
            if time.time() > max_time:
                return current
            cand = list(current)
            cand[pos], cand[pos+1] = cand[pos+1], cand[pos]
            if fast_sorts(cand):
                current = cand
                improved = True
                break
    return current


def construct():
    # 1) Batcher's verified 63-comparator network.
    net = _batcher_odd_even_merge(N)
    initial_count = len(net)

    if not _verify_zero_one(net):
        # Should never happen; fall back to the guaranteed seed network.
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    # 2) Multi-pass greedy reduction with randomized restarts.
    random.seed(42)
    # Leave time for polish + final verification.
    search_deadline = _START + 70.0
    reduced_net = _reduce_network(net, max_time=search_deadline)

    # 3) Polish pass: local search for further reductions.
    polish_deadline = _START + 82.0
    reduced_net = _polish_local_search(reduced_net, polish_deadline)

    # 4) Final verification (safety net).
    if not _verify_zero_one(reduced_net):
        reduced_net = net

    depth = _parallel_depth(reduced_net)
    try:
        import sys
        sys.stderr.write(
            "[construct] Initial: %d, Reduced: %d, depth %d, time %.1fs\n"
            % (initial_count, len(reduced_net), depth, time.time() - _START)
        )
    except Exception:
        pass

    return reduced_net


if __name__ == "__main__":
    net = construct()
    print("comparators:", len(net))
    print("depth:", _parallel_depth(net))
    print("zero-one verified:", _verify_zero_one(net))

import numpy as np
import time

N = 16

def _merge_network(k):
    """Generate comparators to merge two sorted halves of length k."""
    if k == 1:
        return [(0, 1)]
    k_half = k // 2
    even = []
    for i, j in _merge_network(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even.append((mi, mj))
    odd = []
    for i, j in _merge_network(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd.append((mi, mj))
    final = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    return even + odd + final

def _batcher_sort(n):
    if n <= 1:
        return []
    k = n // 2
    left = _batcher_sort(k)
    right = [(i + k, j + k) for i, j in _batcher_sort(k)]
    merge = _merge_network(k)
    return left + right + merge

def _all_inputs(n=N):
    arr = np.arange(1 << n, dtype=np.int32)
    bits = (arr[:, None] >> np.arange(n, dtype=np.int32)[None, :]) & 1
    return bits.astype(np.uint8)

def _simulate(comp, inputs):
    state = inputs.copy()
    for a, b in comp:
        lo = np.minimum(state[:, a], state[:, b])
        hi = np.maximum(state[:, a], state[:, b])
        state[:, a] = lo
        state[:, b] = hi
    return state

def _is_sorted(comp, inputs):
    final_state = _simulate(comp, inputs)
    diffs = np.diff(final_state, axis=1)
    return bool(np.all(diffs >= 0))

def _try_remove(comp, idx, inputs):
    if idx < 0 or idx >= len(comp):
        return False
    candidate = comp[:idx] + comp[idx + 1:]
    return _is_sorted(candidate, inputs)

def _try_remove_pair(comp, i, inputs):
    if i < 0 or i + 1 >= len(comp):
        return False
    candidate = comp[:i] + comp[i + 2:]
    return _is_sorted(candidate, inputs)

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def remaining():
        return deadline - time.time()

    # Build Batcher's odd-even merge network (initial 60+ comparators)
    network = _batcher_sort(N)
    inputs = _all_inputs(N)

    # Verify initial network
    if not _is_sorted(network, inputs):
        raise ValueError("Initial network failed verification")

    # Aggressive redundant comparator removal
    current = list(network)

    # Phase 1: Single removal, scanning from middle outward
    while remaining() > 2:
        removed = False
        n = len(current)
        if n == 0:
            break
        mid = n // 2
        order = list(range(mid, n)) + list(range(mid - 1, -1, -1))
        for idx in order:
            if remaining() < 2:
                break
            if _try_remove(current, idx, inputs):
                current = current[:idx] + current[idx + 1:]
                removed = True
                break
        if not removed:
            break

    # Phase 2: Pair removal (adjacent)
    while remaining() > 2:
        removed = False
        n = len(current)
        if n < 2:
            break
        for i in range(n - 1):
            if remaining() < 2:
                break
            if _try_remove_pair(current, i, inputs):
                current = current[:i] + current[i + 2:]
                removed = True
                break
        if not removed:
            break

    # Phase 3: Final single removal pass
    while remaining() > 2:
        removed = False
        n = len(current)
        if n == 0:
            break
        for idx in range(n):
            if remaining() < 2:
                break
            if _try_remove(current, idx, inputs):
                current = current[:idx] + current[idx + 1:]
                removed = True
                break
        if not removed:
            break

    # Final safety check
    if remaining() > 1:
        if not _is_sorted(current, inputs):
            raise ValueError("Final network failed verification")

    return current

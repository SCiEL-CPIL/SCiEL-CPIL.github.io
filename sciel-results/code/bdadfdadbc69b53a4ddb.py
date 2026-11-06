"""Size-efficient 16-wire sorting network: bitonic baseline + vectorized greedy pruning."""
import time
import numpy as np

N = 16
MAX_COMPARATORS = 4096
DEADLINE = 90.0
MARGIN = 10.0


def _bitonic_network(n):
    """Batcher's bitonic sorting network for n wires (n a power of 2)."""
    comps = []
    for size in range(2, n + 1, 2):
        for k in range(1, size, 2):
            for start in range(0, n, 2 * size):
                comps += [(start + i, start + i + k) for i in range(k)]
    return comps


def _build_index_matrix(comps):
    """For each comparator, the (src, dst) indices into the input matrix."""
    idx = np.empty((len(comps), 2, N), dtype=np.int64)
    for t, (a, b) in enumerate(comps):
        for w in range(N):
            idx[t, 0, w] = a if w == b else w
            idx[t, 1, w] = b if w == a else w
    return idx


def _is_sorting(comps, idx):
    """Zero-one principle: check all 2**N Boolean inputs sort to 00...011...1."""
    if not comps:
        return True
    inputs = np.zeros((N, 1 << N), dtype=np.uint8)
    for w in range(N):
        inputs[w] = (np.arange(1 << N, dtype=np.uint16) >> w) & 1
    target = np.zeros(1 << N, dtype=np.uint8)
    target[N:] = 1
    cur = inputs
    for t in range(len(comps)):
        cur = np.minimum(cur[idx[t, 0]], cur[idx[t, 1]])
    return np.array_equal(cur[N - 1], target)


def _time_left():
    return DEADLINE - time.time() > MARGIN


def construct():
    t0 = time.time()
    comps = _bitonic_network(N)
    idx = _build_index_matrix(comps)
    assert _is_sorting(comps, idx), "baseline must sort"

    # Pass 1: drop comparators that never fire on any Boolean input (no-ops).
    inputs = np.zeros((N, 1 << N), dtype=np.uint8)
    for w in range(N):
        inputs[w] = (np.arange(1 << N, dtype=np.uint16) >> w) & 1
    cur = inputs
    keep = []
    for t in range(len(comps)):
        if not _time_left():
            break
        new = np.minimum(cur[idx[t, 0]], cur[idx[t, 1]])
        if np.array_equal(new, cur):
            continue  # no-op, drop
        cur = new
        keep.append(t)
    comps = [comps[t] for t in keep]
    idx = idx[keep]

    # Pass 2: greedy forward deletion.
    changed = True
    while changed and _time_left():
        changed = False
        for t in range(len(comps)):
            if not _time_left():
                break
            cand = comps[:t] + comps[t + 1:]
            cand_idx = np.concatenate([idx[:t], idx[t + 1:]], axis=0)
            if _is_sorting(cand, cand_idx):
                comps = cand
                idx = cand_idx
                changed = True
                break

    comps = comps[:MAX_COMPARATORS]
    return comps


if __name__ == "__main__":
    net = construct()
    print("comparators:", len(net))
    _idx = _build_index_matrix(net)
    print("sorts:", _is_sorting(net, _idx))

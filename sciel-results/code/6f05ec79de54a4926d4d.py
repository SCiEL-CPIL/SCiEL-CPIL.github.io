import time
import numpy as np

N = 18
NUM_MASKS = 1 << N

def _build_inputs():
    """Return an (NUM_MASKS, N) uint8 array where row m is the bit pattern of m."""
    masks = np.arange(NUM_MASKS, dtype=np.uint32)
    bits = np.zeros((NUM_MASKS, N), dtype=np.uint8)
    for i in range(N):
        bits[:, i] = (masks >> i) & 1
    return bits

def _verify_network(network, inputs):
    """Vectorized check: returns True iff network sorts every 0-1 input.
    inputs: (NUM_MASKS, N) uint8 array."""
    x = inputs.copy()
    for i, j in network:
        xi = x[:, i].copy()
        xj = x[:, j].copy()
        # min to i, max to j
        x[:, i] = np.minimum(xi, xj)
        x[:, j] = np.maximum(xi, xj)
    # sorted iff x[:,k] <= x[:,k+1] for all k
    for k in range(N - 1):
        if np.any(x[:, k] > x[:, k + 1]):
            return False
    return True

def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _batches_odd_even_merge(n):
    """Batcher's odd-even merge sort network for n (power-of-2 friendly) inputs.
    Returns list of (i,j) comparators."""
    if n <= 1:
        return []
    # Use Batcher's odd-even merge sort recursively.
    def _merge(a, b, k):
        # merge two sorted sequences of length k starting at a and b
        # odd-even merge network
        if k == 1:
            return [(a, b)]
        # recursively merge odd and even sub-sequences
        comps = []
        comps += _merge(a, b, k // 2)
        comps += _merge(a + 1, b + 1, k // 2)
        # then the cross comparators
        for i in range(k // 2):
            comps.append((a + i, b + i))
        return comps

    def _sort(n, start=0):
        if n == 1:
            return []
        if n == 2:
            return [(start, start + 1)]
        h = n // 2
        comps = []
        comps += _sort(h, start)
        comps += _sort(h, start + h)
        comps += _merge(start, start + h, h)
        return comps

    return _sort(n)

def _local_remove(network, inputs, deadline):
    """Repeatedly try to remove each comparator; return improved network."""
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # iterate backwards so removals don't shift indices
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 2:
                return network
            test = network[:idx] + network[idx + 1:]
            if _verify_network(test, inputs):
                network = test
                improved = True
                break  # restart scan from top after a successful removal
    return network

def _swap_search(network, inputs, deadline):
    """Try swapping adjacent comparators to enable further removals."""
    cur = network
    cur_len = len(cur)
    found = True
    while found and (time.time() - start_time) < deadline:
        found = False
        for idx in range(len(cur) - 1):
            if (time.time() - start_time) > deadline - 2:
                return cur
            cand = cur[:idx] + [cur[idx + 1], cur[idx]] + cur[idx + 2:]
            # verify then try removals
            if _verify_network(cand, inputs):
                cand2 = _local_remove(cand, inputs, deadline)
                if len(cand2) < cur_len:
                    cur = cand2
                    cur_len = len(cur)
                    found = True
                    break
    return cur

def construct():
    global start_time
    start_time = time.time()
    deadline = start_time + 82  # safety margin under 90s

    inputs = _build_inputs()

    # Candidate seeds: odd-even transposition (153) and Batcher merge (~111)
    seeds = []
    oet = _odd_even_transposition()
    seeds.append(oet)
    try:
        bm = _batches_odd_even_merge(N)
        if len(bm) < len(oet):
            seeds.append(bm)
    except Exception:
        pass

    best = None
    best_len = 10**9

    for seed in seeds:
        if (time.time() - start_time) > deadline - 15:
            break
        # verify seed
        if not _verify_network(seed, inputs):
            continue
        cur = _local_remove(seed, inputs, deadline)
        if len(cur) < best_len:
            best = cur
            best_len = len(cur)
        # second phase: swap search on the best found so far
        if (time.time() - start_time) < deadline - 5:
            cur2 = _swap_search(cur, inputs, deadline)
            if len(cur2) < best_len:
                best = cur2
                best_len = len(cur2)

    # If everything failed, fall back to the odd-even transposition
    if best is None:
        best = _odd_even_transposition()

    return best

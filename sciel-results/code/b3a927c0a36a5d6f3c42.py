import time
import numpy as np

N = 16
TOTAL_ZERO_ONE = 1 << N

T_START = time.time()
DEADLINE = T_START + 82.0

# Precompute the input bits matrix once: (65536, 16) uint8
_INPUT_BITS = (np.arange(TOTAL_ZERO_ONE, dtype=np.int64)[:, None] >> np.arange(N)) & 1
# Precompute popcount of each input index (number of 1s)
_POPCOUNT = np.array([bin(i).count('1') for i in range(TOTAL_ZERO_ONE)], dtype=np.int32)

def _verify(comp_list):
    if len(comp_list) == 0:
        return True
    comps = np.asarray(comp_list, dtype=np.int64)
    bits = _INPUT_BITS.copy()
    n = len(comps)
    # Apply comparators in chunks to reduce Python loop overhead
    CH = 64
    for s in range(0, n, CH):
        chunk = comps[s:s + CH]
        ii = chunk[:, 0]
        jj = chunk[:, 1]
        a = bits[:, ii]
        b = bits[:, jj]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, ii] = lo
        bits[:, jj] = hi
    # A 0/1 network sorts iff wire 0 (the minimum) equals the AND of all inputs.
    # AND of inputs = 1 only when all inputs are 1, i.e. popcount == N.
    # So wire 0 must equal (popcount == N).
    return bool(np.array_equal(bits[:, 0], (_POPCOUNT == N).astype(np.uint8)))

def _batcher_oe_merge_sort(n):
    def oe_merge(left, right):
        comps = []
        h = len(left)
        if h == 1:
            comps.append((left[0], right[0]))
            return comps
        le = [left[k] for k in range(0, h, 2)]
        lo = [left[k] for k in range(1, h, 2)]
        re = [right[k] for k in range(0, h, 2)]
        ro = [right[k] for k in range(1, h, 2)]
        comps += oe_merge(le, re)
        comps += oe_merge(lo, ro)
        # Fixed: Compare left[k] and right[k-1] for k=1, 3, ...
        for k in range(1, h, 2):
            comps.append((left[k], right[k - 1]))
        return comps

    def sort_block(wires):
        m = len(wires)
        if m == 1:
            return []
        h = m // 2
        left = wires[:h]
        right = wires[h:]
        comps = []
        comps += sort_block(left)
        comps += sort_block(right)
        comps += oe_merge(left, right)
        return comps

    return sort_block(list(range(n)))

def _remove_redundant(comp_list):
    cur = list(comp_list)
    rng = np.random.default_rng(0)
    # Repeated passes; each pass tries deletions in a shuffled order.
    # Stop when a full pass removes nothing (local optimum) or deadline.
    while time.time() < DEADLINE:
        n = len(cur)
        if n == 0:
            break
        order = rng.permutation(n)
        changed = False
        for pos in order:
            if time.time() >= DEADLINE:
                break
            if pos >= len(cur):
                continue
            trial = cur[:pos] + cur[pos + 1:]
            if _verify(trial):
                cur = trial
                changed = True
        if not changed:
            break
    return cur

def construct():
    network = _batcher_oe_merge_sort(N)

    if not _verify(network):
        # Fallback
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]

    best = network
    best = _remove_redundant(best)

    if not _verify(best):
        best = network

    return [(int(i), int(j)) for (i, j) in best]

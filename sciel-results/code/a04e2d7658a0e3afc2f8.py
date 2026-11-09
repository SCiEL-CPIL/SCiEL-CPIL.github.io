"""Size-efficient sorting network on 17 wires: Batcher seed + structured local search."""
import time
import numpy as np

N = 17
DEADLINE = 80.0          # wall-clock safety margin (evaluator kills at 90s)
_START = time.time()

# ---------------------------------------------------------------------------
# Bitmask verifier (zero-one principle). A network is correct iff for every
# 0/1 input the output is non-decreasing.
# ---------------------------------------------------------------------------
def _bitmasks(N):
    n = 1 << N
    masks = np.arange(n, dtype=np.int64)
    cols = np.zeros((n, N), dtype=np.int8)
    for j in range(N):
        cols[:, j] = (masks >> (N - 1 - j)) & 1
    return cols

def _verify(cols, net):
    """Return True iff net sorts every 0/1 input (non-decreasing output)."""
    c = cols
    for (i, j) in net:
        a = c[:, i]; b = c[:, j]
        lo = np.minimum(a, b); hi = np.maximum(a, b)
        c[:, i] = lo; c[:, j] = hi
    # check monotone: for every adjacent pair c[:,k] <= c[:,k+1]
    return bool(np.all(c[:, :-1] <= c[:, 1:]))

def _depth(net):
    d = [0] * N
    for (i, j) in net:
        nd = d[i] + 1
        if nd < d[j]:
            nd = d[j] + 1
        d[i] = nd; d[j] = nd
    return max(d)

# ---------------------------------------------------------------------------
# Batcher odd-even mergesort (correct for any N).
# ---------------------------------------------------------------------------
def _batcher(N):
    if N <= 1:
        return []
    if N == 2:
        return [(0, 1)]
    if N % 2 == 0:
        h = N // 2
        net = _batcher(h)
        net += _batcher(h)
        net += _odd_even_merge(N, 0)
    else:
        h = (N - 1) // 2
        net = _batcher(h)
        net += _batcher(h + 1)
        net += _odd_even_merge(N, 1)
    return net

def _odd_even_merge(N, phase):
    comps = []
    comps += _merge_pass(N, phase)
    i = 2
    while i < N:
        comps += _odd_even_merge(i, phase)
        i *= 2
    return comps

def _merge_pass(N, phase):
    comps = []
    i = phase
    while i < N - 1:
        comps.append((i, i + 1))
        i += 2
    return comps

# ---------------------------------------------------------------------------
# Local search: deletion + swap, structured by Batcher layer.
# ---------------------------------------------------------------------------
def _layers(net):
    d = [0] * N
    layers = {}
    for idx, (i, j) in enumerate(net):
        nd = d[i] + 1
        if nd < d[j]:
            nd = d[j] + 1
        d[i] = nd; d[j] = nd
        layers.setdefault(nd, []).append(idx)
    return layers

def construct():
    cols = _bitmasks(N)

    net = _batcher(N)
    if not _verify(cols, net):
        # Fallback: odd-even transposition (guaranteed correct).
        net = [(i, i + 1) for ph in range(N) for i in range(ph % 2, N - 1, 2)]

    best = list(net)
    best_valid = _verify(cols, best)
    best_len = len(best)

    def try_delete(net):
        # Try to remove each comparator; keep removals that stay valid.
        changed = True
        while changed:
            changed = False
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > DEADLINE:
                    break
                cand = net[:idx] + net[idx + 1:]
                if _verify(cols, cand):
                    net = cand
                    changed = True
                    break
        return net

    def try_swap(net):
        # Try swapping adjacent comparators (different wires) that stay valid.
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for idx in range(len(net) - 1):
                a = net[idx]; b = net[idx + 1]
                if a[0] == b[0] or a[0] == b[1] or a[1] == b[0] or a[1] == b[1]:
                    continue
                cand = net[:idx] + [b, a] + net[idx + 2:]
                if _verify(cols, cand):
                    net = cand
                    changed = True
        return net

    # Phase 1: aggressive deletion (biggest wins, cheap to find).
    net = try_delete(net)
    if len(net) < best_len:
        best = list(net); best_len = len(net)

    # Phase 2: interleaved swap + deletion to escape local minima.
    while time.time() < DEADLINE:
        net = try_swap(net)
        net = try_delete(net)
        if len(net) < best_len:
            best = list(net); best_len = len(net)
        # If we hit the proven lower bound, stop.
        if best_len <= 63:
            break

    return best

# ---------------------------------------------------------------------------
# Module-level sanity (kept cheap; construct() is the real work).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _cols = _bitmasks(N)
    _net = construct()
    print("comparators:", len(_net), "depth:", _depth(_net),
          "valid:", _verify(_cols, _net))

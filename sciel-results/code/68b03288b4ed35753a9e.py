import time
import random
import numpy as np

_N = 16
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before the 150 s kill

# Precompute the list of (i,j) edges in lexicographic order
_EDGES = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_M = len(_EDGES)  # 120
_EDGE_IDX = {(i, j): k for k, (i, j) in enumerate(_EDGES)}

# Precompute, for each of the 1820 quadruples, the 6 edge indices (as a 1820x6 int array)
_QUAD_EDGE_IDX = None

def _build_quads():
    global _QUAD_EDGE_IDX
    from itertools import combinations
    quads = list(combinations(range(_N), 4))
    arr = np.empty((len(quads), 6), dtype=np.int64)
    for q, (a, b, c, d) in enumerate(quads):
        pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        for k, p in enumerate(pairs):
            arr[q, k] = _EDGE_IDX[p]
    _QUAD_EDGE_IDX = arr

_build_quads()
_Q = _QUAD_EDGE_IDX.shape[0]  # 1820

def _is_valid(colours):
    """colours: int array length _M. Return True if every K4 has >=5 distinct colours."""
    # For each colour c, build a boolean matrix B_c of shape (16,17):
    # B_c[v, c] = 1 iff vertex v is incident to an edge of colour c.
    # A K4 has <5 colours iff there is a colour c and a set of vertices S
    # (the 4 vertices of the K4) such that all 6 edges of the K4 are in colours
    # that are all "covered" by <=4 colours. Equivalent: for some K4, the set of
    # colours on its 6 edges has size <=4.
    # Fast test: for each K4, gather its 6 colours and count distinct.
    cols = colours[_QUAD_EDGE_IDX]  # shape (1820, 6)
    # Count distinct per row efficiently:
    # Sort each row and count runs.
    s = np.sort(cols, axis=1)
    # distinct count = 1 + number of positions where s[:,k] != s[:,k-1]
    diff = (s[:, 1:] != s[:, :-1])
    distinct = 1 + diff.sum(axis=1)
    return bool(np.all(distinct >= 5))

def _count_distinct(colours):
    return int(np.max(colours) + 1)

def construct():
    rng = random.Random(12345)
    # Start from the known valid 17-colouring
    colours = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int64)
    assert _is_valid(colours)
    best = colours.copy()
    best_k = _count_distinct(best)  # 17

    # Palette compression helper
    def compress(c):
        used = np.unique(c)
        m = len(used)
        if m == c.max() + 1:
            return c, m
        remap = np.zeros(c.max() + 1, dtype=np.int64)
        for new, old in enumerate(used):
            remap[old] = new
        return remap[c], m

    colours, k = compress(colours)
    best_k = k

    # Candidate colours to try are those currently used (0..k-1) plus maybe a fresh one
    # We'll try only colours in [0, k-1] to keep palette size non-increasing,
    # and occasionally try a new colour index k (to enable rearrangement) but that
    # increases palette, so we only accept if it leads to a later reduction.
    # Simpler: only try existing colours; if stuck, do a random restart with a new seed.

    attempts = 0
    while time.time() < _DEADLINE:
        # Pick a random edge
        e = rng.randrange(_M)
        old = colours[e]
        # Try each existing colour != old
        for new in range(k):
            if new == old:
                continue
            colours[e] = new
            if _is_valid(colours):
                # Valid; palette size unchanged (still <= k). Accept.
                attempts += 1
                # Occasionally compress to tighten k
                if attempts % 200 == 0:
                    colours, k2 = compress(colours)
                    if k2 < k:
                        k = k2
                        if k < best_k:
                            best_k = k
                            best = colours.copy()
            else:
                colours[e] = old
        # End of trying all colours for this edge
        # Small chance to do a random single-edge random-colour move to escape local minima
        if rng.random() < 0.05:
            e = rng.randrange(_M)
            old = colours[e]
            new = rng.randrange(k + 1)  # allow one new colour
            colours[e] = new
            if _is_valid(colours):
                colours, k2 = compress(colours)
                k = k2
                if k < best_k:
                    best_k = k
                    best = colours.copy()
            else:
                colours[e] = old
                # Re-compress in case palette shrank elsewhere (it didn't)
        # Check deadline more frequently
        if time.time() > _DEADLINE:
            break

    # Final: ensure best is valid and compressed
    best, k_best = compress(best)
    assert _is_valid(best)
    return best.tolist()

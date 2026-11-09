"""Direct exhaustive search using bitmask for speed."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency as bitmasks for fast popcount-based degree computation
_N = P['n']
_adj_mask = [0] * _N
for (u, v) in P['edges']:
    _adj_mask[u] |= (1 << v)
    _adj_mask[v] |= (1 << u)

# Precompute popcount for all 17-bit masks
_popcount_table = [0] * (1 << _N)
for _i in range(1, 1 << _N):
    _popcount_table[_i] = _popcount_table[_i >> 1] + (_i & 1)

def _popcount(x):
    return _popcount_table[x]

def _distinct_degrees_for_mask(mask):
    """Count distinct degrees in induced subgraph defined by bitmask `mask`."""
    degs = set()
    m = mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        d = _popcount(_adj_mask[v] & mask)
        degs.add(d)
        m ^= lb
    return len(degs)

def construct():
    # The search space is 2^17 = 131,072 masks.
    # This is small enough to enumerate fully without a time guard in the inner loop,
    # ensuring we don't miss the optimum due to time-check overhead or early break.
    best_score = -1
    best_mask = (1 << 8) - 1  # fallback: vertices 0-7

    # Enumerate all subsets of size 8 to 17 using bitmasks
    for mask in range(1 << _N):
        sz = _popcount(mask)
        if sz < 8:
            continue
        score = _distinct_degrees_for_mask(mask)
        if score > best_score:
            best_score = score
            best_mask = mask

    # Convert best_mask to list of Python ints
    result = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        result.append(int(v))
        m ^= lb
    result.sort()
    return result
# EVOLVE-BLOCK-END

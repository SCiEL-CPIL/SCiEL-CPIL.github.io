"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_min_v = int(P['minimum_vertices'])

# Precompute adjacency as bitmask for fast degree computation
_adj_mask = [0] * _N
for _a, _b in P['edges']:
    _adj_mask[_a] |= (1 << _b)
    _adj_mask[_b] |= (1 << _a)

# Precompute bit_count for all 2^17 masks (131072 entries)
_popcount = [0] * (1 << _N)
for _i in range(1, 1 << _N):
    _popcount[_i] = _popcount[_i >> 1] + (_i & 1)

_deadline = time.time() + 80.0  # safety margin under 90s wall-clock limit


def _distinct_degrees_mask(mask):
    """Return the number of distinct degrees within the induced subgraph represented by bitmask."""
    degs = set()
    m = mask
    while m:
        low = m & (-m)
        v = low.bit_length() - 1
        deg = _popcount[mask & _adj_mask[v]]
        degs.add(deg)
        m ^= low
    return len(degs)


def construct():
    global _deadline
    best_score = 0
    best_mask = 0

    # Enumerate every vertex subset of size >= minimum_vertices
    for k in range(_min_v, _N + 1):
        if time.time() > _deadline:
            break
        for combo in itertools.combinations(range(_N), k):
            if time.time() > _deadline:
                break
            mask = 0
            for v in combo:
                mask |= (1 << v)
            score = _distinct_degrees_mask(mask)
            if score > best_score:
                best_score = score
                best_mask = mask

    # Convert best_mask to list of ints
    best_subset = []
    m = best_mask
    while m:
        low = m & (-m)
        v = low.bit_length() - 1
        best_subset.append(int(v))
        m ^= low

    # Fallback if something went wrong
    if len(best_subset) < _min_v:
        best_subset = list(range(_min_v))

    return best_subset
# EVOLVE-BLOCK-END

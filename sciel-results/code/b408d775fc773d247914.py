"""Rewritten program: deadline-guarded randomized local search for max distinct degrees."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_EDGES = P['edges']
_MIN = P['minimum_vertices']

# Precompute bitset adjacency for fast induced-degree computation.
_ADJ = [0] * _N
for _u, _v in _EDGES:
    _ADJ[_u] |= (1 << _v)
    _ADJ[_v] |= (1 << _u)


def _distinct(mask):
    """Number of distinct degrees in the induced subgraph selected by bit-mask."""
    bits = []
    m = mask
    while m:
        lsb = m & (-m)
        bits.append(lsb.bit_length() - 1)
        m ^= lsb
    k = len(bits)
    if k < _MIN:
        return -1
    degs = set()
    for i in range(k):
        vi = bits[i]
        d = bin(_ADJ[vi] & mask).count('1')
        degs.add(d)
    return len(degs)


def construct():
    deadline = time.time() + 80.0  # safety margin under 90s wall-clock kill
    best_mask = 0
    best_score = -1

    def consider(mask):
        nonlocal best_mask, best_score
        s = _distinct(mask)
        if s > best_score:
            best_score = s
            best_mask = mask
        return s

    # --- Greedy seed construction (fast, deterministic) ---
    mask = 0
    for v in range(_N):
        mask |= (1 << v)
        if time.time() > deadline:
            break
    consider(mask)

    # --- Randomized local search polish ---
    rng = random.Random(12345)

    def local_search(mask):
        nonlocal best_mask, best_score
        cur = _distinct(mask)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            cur_bits = []
            m = mask
            while m:
                lsb = m & (-m)
                cur_bits.append(lsb.bit_length() - 1)
                m ^= lsb
            # Try flips: remove one, add one, or swap one
            for i in range(len(cur_bits)):
                if time.time() > deadline:
                    break
                vi = cur_bits[i]
                # Remove vi
                m2 = mask ^ (1 << vi)
                if _distinct(m2) > cur:
                    mask = m2
                    cur = _distinct(mask)
                    improved = True
            if improved:
                continue
            # Add a non-member
            for v in range(_N):
                if time.time() > deadline:
                    break
                if mask & (1 << v):
                    continue
                m2 = mask | (1 << v)
                if _distinct(m2) > cur:
                    mask = m2
                    cur = _distinct(mask)
                    improved = True
            if improved:
                continue
            # Swap: remove vi, add vj
            for i in range(len(cur_bits)):
                if time.time() > deadline:
                    break
                vi = cur_bits[i]
                for vj in range(_N):
                    if time.time() > deadline:
                        break
                    if vj == vi or (mask & (1 << vj)):
                        continue
                    m2 = (mask ^ (1 << vi)) | (1 << vj)
                    if _distinct(m2) > cur:
                        mask = m2
                        cur = _distinct(mask)
                        improved = True
                        break
                if improved:
                    break
        consider(mask)
        return mask

    # Restart with random masks, polish each
    restarts = 0
    while time.time() < deadline - 2.0:
        restarts += 1
        if restarts % 200 == 0:
            rng.seed((restarts * 2654435761) & 0xFFFFFFFF)
        # Build a random mask of size 8..17
        size = rng.randint(_MIN, _N)
        perm = list(range(_N))
        rng.shuffle(perm)
        rmask = 0
        for v in perm[:size]:
            rmask |= (1 << v)
        local_search(rmask)

    # Final high-precision polish on the best mask found
    if time.time() < deadline - 1.0:
        local_search(best_mask)

    # Convert to a list of Python ints (exact types)
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        result.append(int(lsb.bit_length() - 1))
        m ^= lsb
    if len(result) < _MIN:
        # Fallback: ensure at least _MIN distinct vertices
        result = [int(v) for v in range(_MIN)]
    return result
# EVOLVE-BLOCK-END

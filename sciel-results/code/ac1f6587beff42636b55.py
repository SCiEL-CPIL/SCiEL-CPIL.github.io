"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {
    'n': 17,
    'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16],
              [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16],
              [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15],
              [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16],
              [4, 5], [4, 6], [4, 8], [4, 12], [4, 13],
              [5, 6], [5, 7], [5, 9], [5, 13], [5, 14],
              [6, 7], [6, 8], [6, 10], [6, 14], [6, 15],
              [7, 8], [7, 9], [7, 11], [7, 15], [7, 16],
              [8, 9], [8, 10], [8, 12], [8, 16],
              [9, 10], [9, 11], [9, 13],
              [10, 11], [10, 12], [10, 14],
              [11, 12], [11, 13], [11, 15],
              [12, 13], [12, 14], [12, 16],
              [13, 14], [13, 15],
              [14, 15], [14, 16],
              [15, 16]],
    'homogeneous_cap': 4,
    'minimum_vertices': 8,
}


def _adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _diversity(mask, adj):
    """Number of distinct induced degrees for the vertex set encoded by mask."""
    degs = set()
    for i in range(P['n']):
        if mask & (1 << i):
            degs.add(len(adj[i]) & mask if False else bin(mask & adj_mask[i]).count('1'))
    return len(degs)


# Precompute adjacency masks for faster popcount
def _init_masks():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    masks = []
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= 1 << j
        masks.append(m)
    return masks

_ADJ_MASKS = _init_masks()


def _diversity_fast(mask):
    degs = set()
    for i in range(P['n']):
        if mask & (1 << i):
            degs.add(bin(mask & _ADJ_MASKS[i]).count('1'))
    return len(degs)


def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    n = P['n']
    min_v = P['minimum_vertices']

    best_cnt = 0
    best_mask = (1 << min_v) - 1  # first min_v vertices as fallback

    # Iterate masks in a clever order: start from larger subsets (higher potential diversity)
    # but we want to be fast. Let's just do a full scan with bitmask.
    # Total masks: 2^17 = 131072. Filter: popcount >= min_v.
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
        # Quick popcount check
        pc = bin(mask).count('1')
        if pc < min_v:
            continue
        cnt = _diversity_fast(mask)
        if cnt > best_cnt:
            best_cnt = cnt
            best_mask = mask
            if best_cnt == n:
                break
            if best_cnt >= pc:
                # Can't do better with this mask size, but could with larger
                pass

    # Convert best_mask to list of ints
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(int(i))

    return result
# EVOLVE-BLOCK-END

"""Rewritten program: exhaustive search over vertex subsets to maximize distinct induced degrees."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _distinct_degrees(subset, adj):
    sub_set = set(subset)
    degs = set()
    for v in subset:
        d = sum(1 for u in adj[v] if u in sub_set)
        degs.add(d)
    return len(degs)


def construct():
    start = time.time()
    deadline = start + 85.0
    n = P['n']
    min_v = P['minimum_vertices']
    adj = _build_adjacency()

    # Precompute bitmask adjacency for speed
    adj_mask = [0] * n
    for v in range(n):
        m = 0
        for u in adj[v]:
            m |= (1 << u)
        adj_mask[v] = m

    best_count = 0
    best_subset = None

    # Enumerate all subsets of size >= min_v
    for size in range(min_v, n + 1):
        if time.time() > deadline:
            break
        for combo in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
            mask = 0
            for v in combo:
                mask |= (1 << v)
            degs = set()
            for v in combo:
                d = bin(adj_mask[v] & mask).count('1')
                degs.add(d)
            cnt = len(degs)
            if cnt > best_count:
                best_count = cnt
                best_subset = list(combo)
            # Early stop: max possible distinct degrees for a size-k induced subgraph is k
            # (degrees range 0..k-1). If we already hit k, we can't beat it for this size.
            if best_count == size:
                # Can't exceed size for this size; but larger sizes could exceed.
                # Only break inner if size is max we care about; continue outer.
                pass

    if best_subset is None:
        best_subset = list(range(min_v))

    # Ensure Python ints and a list
    return [int(v) for v in best_subset]


# Precompute once at import to avoid repeated work; but construct() is self-contained.
# EVOLVE-BLOCK-END

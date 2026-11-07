"""Optimized prefix-minimal perfect-matching search with deadline guard."""
P = {'n': 18, 'edge_order': [[0, 1, 6], [4, 13, 14], [2, 3, 16], [9, 10, 17], [6, 14, 17], [11, 12, 16], [3, 10, 17], [10, 12, 14], [3, 9, 13], [0, 7, 13], [3, 14, 16], [1, 5, 8], [0, 9, 12], [9, 13, 17], [4, 11, 17], [4, 8, 13], [3, 14, 17], [3, 11, 15], [11, 13, 17], [8, 11, 12], [4, 9, 13], [2, 7, 9], [3, 6, 14], [0, 3, 5], [5, 7, 16], [2, 9, 16], [6, 7, 9], [1, 7, 8], [3, 10, 13], [3, 10, 14], [1, 10, 15], [2, 4, 14], [4, 11, 14], [6, 7, 10], [5, 6, 12], [3, 13, 17], [0, 5, 7], [5, 7, 11], [0, 2, 11], [9, 15, 16], [0, 2, 14], [5, 15, 17], [0, 3, 7], [10, 12, 15], [7, 9, 11], [4, 7, 17], [4, 11, 16], [3, 6, 17], [7, 11, 14], [1, 3, 8], [1, 10, 16], [3, 9, 16], [7, 8, 11], [7, 10, 14], [4, 10, 14], [3, 14, 15], [4, 14, 17], [4, 16, 17], [9, 12, 15], [0, 2, 4], [7, 9, 15], [10, 12, 17], [7, 8, 13], [8, 11, 15], [8, 16, 17], [5, 11, 13], [1, 11, 14], [0, 6, 10], [0, 8, 14], [5, 6, 16], [0, 6, 17], [1, 16, 17], [7, 11, 16], [2, 11, 16], [0, 1, 13], [6, 7, 14], [4, 7, 11], [6, 11, 14], [1, 7, 14], [1, 4, 8], [5, 9, 15], [8, 9, 15], [3, 11, 13], [2, 11, 14], [3, 12, 13], [2, 4, 8], [12, 14, 15], [1, 12, 17], [1, 14, 17], [6, 15, 16], [0, 11, 13], [2, 5, 15], [11, 12, 17], [5, 15, 16], [2, 6, 15], [5, 6, 15], [5, 8, 15], [0, 7, 9], [1, 4, 14], [9, 14, 15], [2, 4, 10], [3, 7, 15], [5, 7, 17], [1, 10, 14], [6, 8, 17], [1, 2, 15], [1, 14, 15], [2, 5, 17], [7, 9, 16], [10, 13, 17], [5, 9, 14], [0, 12, 13], [0, 13, 14], [7, 10, 12], [2, 11, 15], [4, 5, 10], [10, 16, 17], [4, 14, 16], [1, 8, 15], [5, 6, 14], [7, 11, 15], [3, 11, 12], [7, 9, 17], [4, 7, 14], [1, 5, 15], [3, 7, 12], [4, 5, 12], [3, 4, 15], [1, 6, 10], [1, 8, 10], [6, 10, 13], [9, 11, 17], [10, 14, 15], [2, 3, 5], [0, 5, 6], [4, 11, 12], [2, 6, 16], [2, 8, 10], [2, 3, 11], [9, 10, 11], [2, 9, 17], [0, 12, 16], [0, 6, 13], [1, 9, 13], [5, 6, 9], [4, 5, 17], [2, 7, 10], [2, 3, 7], [1, 7, 11], [3, 7, 17], [0, 2, 7], [3, 5, 10], [4, 7, 15], [1, 2, 11], [6, 8, 10], [2, 4, 6], [0, 2, 13], [2, 3, 17], [3, 7, 16], [3, 4, 10], [1, 6, 7], [6, 10, 12], [4, 6, 7], [7, 10, 17], [8, 13, 17], [1, 12, 15], [0, 8, 15], [2, 5, 7], [1, 3, 15], [2, 4, 11], [4, 6, 8], [9, 14, 16], [0, 13, 16], [0, 9, 16], [8, 10, 14], [3, 5, 12], [2, 9, 13], [2, 7, 12], [0, 1, 11], [1, 7, 13], [1, 4, 7], [0, 9, 17], [0, 1, 14], [8, 13, 14], [0, 2, 5], [1, 10, 12], [8, 12, 13], [1, 14, 16], [3, 8, 12], [5, 7, 14], [1, 5, 9], [5, 13, 16], [0, 1, 15], [0, 16, 17], [3, 5, 6], [5, 8, 11], [0, 3, 13], [0, 12, 14], [5, 6, 8], [5, 7, 12], [4, 5, 8], [0, 10, 11], [1, 8, 11], [1, 13, 16], [3, 12, 16], [3, 6, 8], [0, 2, 17], [9, 10, 12], [2, 5, 8], [1, 12, 14], [8, 9, 13], [0, 7, 8], [0, 14, 17], [4, 10, 13], [9, 12, 13], [5, 11, 15], [13, 14, 15], [1, 9, 10], [2, 7, 15], [6, 9, 10], [0, 12, 15], [12, 13, 16], [0, 9, 14], [6, 13, 15], [1, 4, 10], [4, 7, 16], [12, 15, 17], [2, 7, 14], [0, 4, 5], [3, 16, 17], [13, 15, 16], [5, 7, 10], [6, 7, 16], [6, 15, 17], [6, 12, 13], [1, 13, 15], [0, 4, 8], [7, 13, 15], [4, 9, 11], [6, 7, 13], [0, 8, 16], [5, 6, 11], [2, 4, 17], [3, 13, 14], [10, 11, 13], [2, 11, 12], [6, 7, 17], [4, 13, 15], [3, 5, 13], [0, 9, 15], [3, 13, 16], [4, 12, 15], [12, 16, 17], [2, 6, 12], [7, 8, 12], [0, 7, 17], [2, 10, 12], [8, 9, 16], [8, 10, 13], [6, 9, 16], [6, 13, 17], [1, 3, 13], [6, 11, 12], [6, 13, 14], [2, 4, 9], [0, 10, 15], [0, 14, 16], [1, 9, 17], [4, 8, 10], [3, 6, 13], [4, 6, 17], [1, 2, 8], [2, 10, 17], [2, 8, 16], [0, 7, 16], [0, 10, 17], [2, 14, 15], [6, 7, 11], [10, 11, 16], [6, 9, 11], [1, 8, 17], [1, 7, 9], [0, 6, 9], [10, 14, 17], [2, 5, 10], [4, 8, 9], [7, 8, 15], [2, 12, 17], [9, 10, 14], [8, 12, 17], [6, 7, 12], [5, 8, 14], [6, 9, 14], [5, 12, 14], [3, 15, 17], [10, 13, 16], [8, 14, 15], [7, 11, 17], [5, 10, 13], [5, 11, 14], [6, 12, 15], [2, 10, 15], [0, 5, 15], [12, 14, 17], [11, 13, 16], [2, 13, 16], [2, 9, 15], [0, 15, 17], [0, 6, 11], [2, 8, 13], [0, 1, 4], [4, 12, 13], [5, 9, 12], [0, 9, 10], [1, 9, 11], [7, 15, 16], [0, 1, 8], [5, 7, 9], [0, 1, 9], [11, 12, 14], [1, 8, 14], [6, 8, 12], [5, 8, 17], [3, 8, 15], [3, 5, 17], [2, 5, 11], [0, 3, 12], [5, 14, 16], [7, 14, 17], [6, 9, 12], [9, 10, 15], [7, 9, 13], [9, 11, 16], [3, 7, 11], [3, 9, 12], [3, 4, 7], [1, 5, 17], [0, 1, 3], [5, 10, 12], [2, 3, 8], [7, 13, 14], [0, 7, 12], [9, 13, 15], [3, 11, 14], [0, 4, 16], [12, 13, 15], [7, 9, 14], [2, 10, 16], [7, 9, 10], [3, 10, 12], [14, 15, 16], [5, 12, 16], [1, 3, 10], [4, 10, 15], [5, 6, 13], [1, 7, 10], [6, 9, 13], [2, 9, 11], [1, 7, 15], [1, 7, 16], [3, 10, 16], [1, 6, 13], [0, 9, 13], [5, 8, 13], [11, 13, 15], [2, 4, 15], [10, 11, 14], [13, 16, 17], [3, 5, 7], [0, 8, 13], [3, 8, 17], [2, 5, 6], [3, 4, 6], [0, 1, 10], [3, 8, 10], [0, 3, 8], [2, 13, 17], [0, 4, 15], [2, 7, 13], [3, 12, 17], [8, 10, 11], [4, 8, 12], [10, 12, 13], [11, 12, 15], [0, 4, 10], [8, 10, 17], [4, 5, 11], [0, 10, 16], [0, 6, 7], [3, 10, 11], [1, 15, 17], [3, 8, 13], [3, 6, 9], [4, 15, 17], [2, 4, 13], [4, 8, 11], [1, 10, 17], [0, 2, 12], [0, 7, 10], [0, 1, 16], [0, 7, 14], [4, 5, 6], [4, 5, 9], [2, 6, 10], [2, 6, 17], [0, 7, 15], [4, 5, 14], [3, 7, 10], [2, 16, 17], [5, 8, 12], [6, 11, 16], [6, 10, 15], [2, 15, 17], [4, 6, 14], [0, 10, 12], [10, 15, 17], [1, 6, 16], [0, 12, 17], [6, 11, 13], [1, 3, 9], [7, 12, 16], [3, 5, 14], [6, 10, 11], [3, 11, 16], [1, 4, 12], [7, 15, 17], [4, 9, 17], [7, 8, 17], [1, 8, 13], [3, 6, 12], [1, 13, 17], [2, 9, 10], [7, 9, 12], [1, 8, 9], [2, 8, 17], [0, 11, 14], [3, 6, 11], [2, 5, 12], [11, 15, 16], [1, 9, 12], [2, 7, 16], [5, 8, 10], [2, 6, 14], [9, 13, 14], [5, 9, 10], [2, 12, 14], [0, 8, 11], [8, 12, 16], [3, 4, 8], [7, 11, 13], [5, 14, 15], [2, 4, 16], [1, 2, 10], [11, 14, 15], [6, 8, 15], [10, 11, 15], [6, 8, 16], [11, 12, 13], [8, 11, 17], [11, 15, 17], [2, 3, 13], [1, 5, 11], [0, 5, 16], [4, 6, 15], [14, 16, 17], [0, 9, 11], [0, 3, 10], [0, 2, 3], [3, 10, 15], [10, 12, 16], [2, 11, 13], [4, 5, 7], [10, 15, 16], [8, 14, 17], [1, 13, 14], [2, 6, 8], [1, 2, 17], [2, 4, 5], [9, 13, 16], [4, 10, 12], [7, 14, 16], [5, 7, 13], [3, 4, 16], [2, 12, 15], [0, 3, 4], [6, 7, 15], [8, 11, 14], [5, 9, 13], [1, 2, 9], [0, 8, 12], [1, 5, 16], [0, 5, 14], [8, 10, 16], [1, 10, 13], [10, 11, 12], [4, 15, 16], [5, 13, 15], [1, 10, 11], [13, 14, 16], [2, 10, 14], [1, 5, 7], [6, 14, 16], [4, 6, 11], [2, 3, 12], [7, 14, 15], [7, 10, 16], [4, 12, 17], [0, 8, 10], [0, 7, 11], [1, 9, 14], [8, 13, 16], [4, 13, 16], [1, 8, 12], [1, 6, 14], [9, 11, 14], [8, 10, 12], [1, 3, 5], [0, 3, 14], [0, 4, 7], [2, 13, 14], [5, 11, 17], [1, 2, 16], [3, 7, 8], [2, 11, 17], [5, 13, 14], [3, 5, 11], [4, 8, 14], [1, 3, 17], [0, 14, 15], [5, 8, 9], [0, 15, 16], [0, 11, 12], [0, 2, 16], [2, 3, 9], [2, 5, 13], [8, 12, 15], [9, 10, 13], [7, 10, 13], [0, 6, 12], [3, 11, 17], [1, 5, 13], [3, 7, 9], [0, 1, 2], [1, 12, 13], [4, 9, 10], [1, 4, 6], [8, 9, 11], [3, 5, 15], [2, 14, 17], [2, 9, 14], [2, 8, 11], [5, 12, 13], [7, 10, 11], [9, 12, 16], [5, 12, 17], [0, 5, 10], [10, 14, 16], [5, 10, 15], [8, 9, 17], [1, 6, 15], [1, 8, 16], [6, 13, 16], [4, 5, 15], [8, 13, 15], [6, 10, 14], [3, 7, 13], [3, 4, 5], [0, 2, 9], [5, 6, 7], [8, 15, 17], [0, 11, 16], [5, 7, 15], [7, 12, 15], [6, 8, 9], [0, 1, 12], [6, 8, 11], [0, 1, 7], [0, 5, 8], [0, 13, 15], [0, 3, 11], [0, 4, 17], [9, 15, 17], [5, 11, 16], [9, 16, 17], [4, 11, 15], [8, 12, 14], [3, 5, 16], [3, 5, 8], [0, 6, 16], [2, 3, 14], [4, 6, 12], [4, 12, 16], [6, 8, 13], [5, 9, 17], [5, 9, 11], [3, 7, 14], [8, 9, 12], [0, 5, 17], [4, 7, 10], [3, 9, 14], [3, 9, 17], [1, 2, 4], [11, 14, 16], [0, 2, 10], [1, 3, 6], [2, 8, 15], [5, 8, 16], [3, 4, 9], [2, 8, 14], [2, 5, 9], [1, 4, 5], [4, 6, 13], [9, 12, 17], [7, 12, 17], [3, 6, 16], [10, 13, 14], [5, 10, 17], [6, 11, 17], [5, 12, 15], [4, 12, 14], [2, 7, 17], [3, 12, 14], [0, 3, 15], [1, 4, 16], [6, 11, 15], [0, 4, 9], [0, 10, 14], [0, 10, 13], [4, 9, 16], [0, 6, 8], [0, 4, 14], [4, 9, 14], [2, 14, 16], [13, 14, 17], [0, 3, 9], [2, 8, 12], [7, 16, 17], [6, 14, 15], [3, 13, 15], [4, 10, 16], [9, 11, 15], [4, 7, 9], [4, 10, 11], [1, 2, 5], [6, 7, 8], [0, 3, 17], [3, 9, 15], [0, 13, 17], [3, 12, 15], [5, 6, 17], [4, 5, 13], [6, 10, 17], [1, 5, 10], [1, 2, 6], [6, 12, 14], [11, 13, 14], [1, 11, 17], [0, 6, 14], [0, 2, 8], [1, 2, 7], [1, 6, 8], [1, 3, 12], [3, 4, 11], [7, 12, 13], [12, 13, 14], [4, 7, 13], [2, 5, 14], [1, 2, 3], [7, 8, 10], [0, 4, 12], [0, 4, 6], [2, 3, 4], [8, 14, 16], [1, 3, 16], [1, 15, 16], [3, 15, 16], [0, 5, 13], [1, 9, 16], [6, 9, 15], [5, 9, 16], [1, 5, 14], [7, 13, 16], [1, 4, 15], [10, 11, 17], [1, 3, 4], [2, 12, 16], [0, 11, 17], [4, 9, 12], [5, 13, 17], [2, 6, 13], [4, 9, 15], [5, 7, 8], [8, 11, 13], [1, 9, 15], [9, 10, 16], [0, 11, 15], [0, 2, 6], [0, 1, 17], [1, 6, 17], [9, 11, 13], [2, 9, 12], [4, 7, 8], [1, 2, 12], [1, 11, 16], [4, 11, 13], [1, 11, 12], [4, 8, 15], [2, 3, 15], [2, 15, 16], [2, 10, 13], [0, 5, 12], [13, 15, 17], [3, 4, 13], [5, 16, 17], [9, 11, 12], [2, 10, 11], [3, 4, 14], [6, 10, 16], [8, 10, 15], [4, 5, 16], [4, 6, 10], [2, 3, 6], [8, 9, 14], [9, 12, 14], [5, 10, 11], [12, 14, 16], [0, 8, 9], [6, 16, 17], [4, 6, 9], [15, 16, 17], [7, 12, 14], [3, 5, 9], [1, 7, 12], [0, 8, 17], [0, 5, 11], [12, 13, 17], [6, 12, 16], [5, 10, 14], [1, 3, 11], [3, 9, 11], [7, 8, 16], [1, 4, 13], [1, 6, 9], [6, 9, 17], [8, 15, 16], [1, 3, 7], [0, 1, 5], [6, 12, 17], [3, 4, 17], [8, 11, 16], [11, 16, 17], [3, 8, 9], [7, 13, 17], [1, 6, 12], [1, 2, 13], [4, 13, 17], [0, 4, 11], [2, 6, 9], [2, 6, 11], [5, 6, 10], [7, 8, 9], [2, 3, 10], [0, 4, 13], [3, 6, 10], [6, 8, 14], [2, 7, 11], [4, 7, 12], [10, 13, 15], [1, 5, 12], [5, 14, 17], [2, 4, 7], [1, 4, 17], [4, 14, 15], [3, 8, 16], [12, 15, 16], [7, 8, 14], [2, 4, 12], [9, 14, 17], [3, 8, 14], [1, 4, 11], [0, 5, 9], [4, 8, 17], [1, 11, 15], [1, 4, 9], [0, 3, 16], [4, 8, 16], [0, 3, 6], [3, 8, 11], [4, 10, 17], [1, 7, 17], [8, 9, 10], [3, 4, 12], [2, 8, 9], [5, 11, 12], [1, 12, 16], [3, 9, 10], [2, 7, 8], [1, 3, 14], [4, 6, 16], [2, 5, 16], [1, 6, 11], [2, 13, 15], [5, 10, 16], [14, 15, 17], [0, 2, 15], [3, 6, 15], [2, 12, 13], [11, 14, 17], [2, 6, 7], [1, 2, 14], [1, 5, 6], [7, 10, 15], [7, 11, 12], [0, 6, 15], [3, 6, 7], [1, 11, 13]]}


def construct():
    """Return six disjoint triples covering 0..17 minimizing largest one-based position
    in P['edge_order']."""
    import time

    start = time.time()
    deadline = start + 70.0

    edge_order = P['edge_order']
    n = int(P.get('n', 18))
    m = len(edge_order)
    full_mask = (1 << n) - 1

    # Precompute canonical key and mask for each triple
    tri_key = [tuple(sorted(tuple(int(x) for x in t))) for t in edge_order]
    masks = []
    for t in edge_order:
        mm = 0
        for x in t:
            mm |= 1 << int(x)
        masks.append(mm)

    # Index by sorted canonical key to the first occurrence index (one-based will be idx+1)
    first_pos = {}
    for idx, key in enumerate(tri_key):
        if key not in first_pos:
            first_pos[key] = idx

    # Helper: find one-based position for a triple (as list/tuple) or its sorted tuple
    def pos_of(triple):
        key = tuple(sorted(tuple(int(x) for x in triple)))
        if key in first_pos:
            return first_pos[key] + 1
        return None

    def valid_match(triples):
        # triples: list of 6 lists of 3 ints
        if not isinstance(triples, (list, tuple)):
            return False
        if len(triples) != 6:
            return False
        seen = 0
        for t in triples:
            if not isinstance(t, (list, tuple)):
                return False
            if len(t) != 3:
                return False
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in t):
                return False
            vals = [int(x) for x in t]
            if len(set(vals)) != 3:
                return False
            mm = 0
            for x in vals:
                if x < 0 or x >= n:
                    return False
                mm |= 1 << x
            if mm & seen:
                return False
            seen |= mm
        return seen == full_mask

    def value_of(triples):
        vals = []
        for t in triples:
            p = pos_of(t)
            if p is None:
                return float('inf')
            vals.append(p)
        return max(vals)

    def time_ok():
        return time.time() < deadline

    # Build vertex -> list of candidate indices for full set, ordered by position
    vertex_to_idx = [[] for _ in range(n)]
    for idx in range(m):
        t = edge_order[idx]
        for x in t:
            vertex_to_idx[int(x)].append(idx)

    # Precompute bitcount lookup for 18-bit masks (fast)
    bitcount_table = [0] * (1 << n)
    for i in range(1, 1 << n):
        bitcount_table[i] = bitcount_table[i >> 1] + (i & 1)

    # For a given k (first k candidates only), check existence
    def exists_in_prefix(k):
        if k < 6:
            return None

        cand_indices = list(range(k))
        cand_masks = [masks[i] for i in cand_indices]

        # local vertex -> cand indices restricted to prefix
        local_v2i = [[] for _ in range(n)]
        for local_i, ci in enumerate(cand_indices):
            t = edge_order[ci]
            for x in t:
                local_v2i[int(x)].append(local_i)

        # Precompute for each vertex in prefix: which candidates cover it, sorted by position
        # (already in order since we iterate ci in increasing order)

        # Memoization: used_mask -> failure (we only memo failures)
        memo_fail = set()

        def dfs(used, path):
            if time.time() > deadline:
                return None

            count = bitcount_table[used] // 3
            if count == 6:
                if used == full_mask:
                    return path[:]
                return None

            if used in memo_fail:
                return None

            # find smallest unused vertex
            v = 0
            while v < n and (used >> v) & 1:
                v += 1
            if v == n:
                memo_fail.add(used)
                return None

            # Try candidates for vertex v, in order of their position (earliest first)
            # local_v2i[v] is already in order of increasing ci (position)
            for li in local_v2i[v]:
                cm = cand_masks[li]
                if cm & used == 0:
                    path.append(cand_indices[li])
                    res = dfs(used | cm, path)
                    if res is not None:
                        return res
                    path.pop()

            memo_fail.add(used)
            return None

        return dfs(0, [])

    # Strategy 1: binary search smallest k where exists_in_prefix(k) returns solution
    best = None
    best_val = float('inf')

    # Binary search
    lo, hi = 6, m
    best_k_sol = None
    while lo <= hi and time_ok():
        mid = (lo + hi) // 2
        sol = exists_in_prefix(mid)
        if sol is not None:
            best_k_sol = sol
            hi = mid - 1
        else:
            lo = mid + 1

    if best_k_sol is not None:
        triples = [[int(edge_order[i][0]), int(edge_order[i][1]), int(edge_order[i][2])] for i in best_k_sol]
        if valid_match(triples):
            v = value_of(triples)
            if v < best_val:
                best_val = v
                best = triples

    # Strategy 1b: Try to improve by searching slightly below the binary search result
    # The binary search finds the smallest prefix k with ANY solution, but the solution
    # found might use indices close to k. Try a few values below to see if we get a better max position.
    if best_k_sol is not None:
        # Try k-1, k-2, ..., k-5 (if they exist) to see if we can find a solution with smaller max index
        for trial_k in range(max(6, best_k_sol[-1]), max(5, best_k_sol[-1] - 5), -1):
            if not time_ok():
                break
            sol2 = exists_in_prefix(trial_k)
            if sol2 is not None:
                triples2 = [[int(edge_order[i][0]), int(edge_order[i][1]), int(edge_order[i][2])] for i in sol2]
                if valid_match(triples2):
                    v2 = value_of(triples2)
                    if v2 < best_val:
                        best_val = v2
                        best = triples2

    # Strategy 2: greedy by position using only first-occurrence positions up to some bound
    # For each bound B (one-based), consider candidates whose first_pos+1 <= B.
    # But binary search over prefix already does this implicitly if positions align with index.
    # However, positions are by index in edge_order, so first_pos+1 == idx+1 for first occurrence.
    # Since we restrict to first k in order, any triple with index < k is included.
    # Thus value = max(index+1) over chosen = max index + 1 <= k for solution found in prefix k.
    # So the smallest k with a solution gives minimal max position (since any solution in prefix k has all indices < k).
    # But a solution using indices not all < k could have max position smaller than some k' where k' is the smallest prefix with ANY solution? 
    # Actually if there's a solution with max position P (one-based), then prefix P (1-based) = first P triples includes it. So smallest prefix with a solution <= P.
    # Conversely if prefix k has a solution, its max position <= k. So minimal max position = smallest k with solution. 
    # So binary search result is optimal IF positions are exactly the indices (which they are for first occurrence; but a triple may appear later too, but first_pos uses first).
    # Since we use first occurrence position, and the solution we extract uses specific indices from prefix, those indices' one-based positions are exactly index+1 (since they are the first k, each index i has position i+1 regardless of duplicates? The task says "pinned shuffled list of all triples" and "arrival index" - presumably the position in the list. If duplicates exist, the same triple at different positions has different arrival indices. But the task says "six disjoint triples" - they must be distinct as sets? It says vertex order doesn't matter, so [a,b,c] same as [c,b,a]. Duplicates in edge_order with same set would be the same triple at different positions. We need to pick 6 triples (sets) that are disjoint in vertices. If the same set appears twice, picking it once is fine. The arrival index is the position in the list where that triple appears. If we need to use a triple, we can use its earliest occurrence to minimize max position. 
    # In our extraction, we use specific indices from the prefix. If a triple set appears at index i and also at j>i, using i is better. Our search uses the actual index in the prefix, which could be a later duplicate if the earlier was not used? But we include all first k entries, so both occurrences are present if both < k. The search may pick the later one unnecessarily. To minimize max position, for the chosen set of 6 distinct triple-sets, we should assign each to its earliest occurrence. 
    # Let's post-process: given 6 indices from search, convert to sets, then for each set find first_pos (earliest index) and use that. Then max position = max(first_pos(set)+1). This could be smaller than max(index+1) if search picked a later duplicate.
    # Also, the search restricts to first k entries; a triple-set whose earliest occurrence is before k but search picked a later duplicate within k - we can just use earliest.
    # But also, there might be a triple-set whose earliest occurrence is >= k but appears in prefix? No, if earliest >= k, it's not in first k.
    # So post-processing to earliest occurrence is safe and can only improve.

    if best is not None:
        # Post-process to use earliest occurrences
        sets_used = []
        for t in best:
            sets_used.append(tuple(sorted(tuple(int(x) for x in t))))
        # Ensure they are distinct sets (they should be, since disjoint vertices)
        # For each set, use first_pos
        new_triples = []
        for s in sets_used:
            idx = first_pos.get(s, None)
            if idx is not None:
                new_triples.append([int(edge_order[idx][0]), int(edge_order[idx][1]), int(edge_order[idx][2])])
        if valid_match(new_triples):
            v2 = value_of(new_triples)
            if v2 < best_val:
                best_val = v2
                best = new_triples

    # Strategy 4: if best is None, do a full search with node limit
    if best is None:
        # Full existence without prefix restriction, then minimize
        # Just call exists_in_prefix(m)
        sol = exists_in_prefix(m)
        if sol is not None:
            triples = [[int(edge_order[i][0]), int(edge_order[i][1]), int(edge_order[i][2])] for i in sol]
            if valid_match(triples):
                v = value_of(triples)
                # Post-process for earliest
                sets_used = [tuple(sorted(tuple(int(x) for x in t))) for t in triples]
                new_triples = []
                for s in sets_used:
                    idx = first_pos.get(s, None)
                    if idx is not None:
                        new_triples.append([int(edge_order[idx][0]), int(edge_order[idx][1]), int(edge_order[idx][2])])
                if valid_match(new_triples):
                    v2 = value_of(new_triples)
                    if v2 < best_val:
                        best_val = v2
                        best = new_triples
                else:
                    if v < best_val:
                        best_val = v
                        best = triples

    # Final safety: if still none, return a hardcoded valid matching
    if best is None or not valid_match(best):
        # Construct a valid matching from first few triples
        seen = 0
        chosen = []
        for i in range(m):
            mm = masks[i]
            if mm & seen == 0:
                chosen.append(i)
                seen |= mm
                if seen == full_mask:
                    break
        if len(chosen) == 6 and seen == full_mask:
            triples = [[int(edge_order[i][0]), int(edge_order[i][1]), int(edge_order[i][2])] for i in chosen]
            if valid_match(triples):
                best = triples
                best_val = value_of(triples)
        else:
            # Last resort: hardcoded valid matching
            best = [[0, 1, 6], [4, 13, 14], [2, 3, 16], [9, 10, 17], [7, 11, 15], [5, 8, 12]]
            if not valid_match(best):
                best = [[0, 1, 6], [4, 13, 14], [2, 3, 16], [9, 10, 17], [11, 12, 16], [5, 8, 15]]
            if not valid_match(best):
                best = [[0, 1, 6], [4, 13, 14], [2, 3, 16], [9, 10, 17], [7, 11, 15], [5, 8, 12]]

    # Ensure return type: list of lists of Python int
    final = []
    for t in best:
        final.append([int(t[0]), int(t[1]), int(t[2])])
    return final

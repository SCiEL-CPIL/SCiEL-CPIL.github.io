import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = P['n']
    edges = P['edges']

    # Build adjacency matrix (bool) for fast membership
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    # Remaining edges as a set of frozenset pairs (min, max)
    remaining = set()
    for u, v in edges:
        if u < v:
            remaining.add((u, v))
        else:
            remaining.add((v, u))

    # Helper: given L and R (sets), check if every cross-pair edge exists in adj
    def is_valid_biclique(L, R):
        for u in L:
            for v in R:
                if not adj[u][v]:
                    return False
        return True

    # Helper: expand L given fixed R: add all w not in L∪R that connect to every r in R
    def expand_L(L, R):
        new_L = set(L)
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w in new_L or w in R:
                    continue
                ok = True
                for r in R:
                    if not adj[w][r]:
                        ok = False
                        break
                if ok:
                    new_L.add(w)
                    changed = True
        return new_L

    # Helper: expand R given fixed L
    def expand_R(L, R):
        new_R = set(R)
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w in L or w in new_R:
                    continue
                ok = True
                for l in L:
                    if not adj[w][l]:
                        ok = False
                        break
                if ok:
                    new_R.add(w)
                    changed = True
        return new_R

    # Full expansion: iteratively expand both sides
    def full_expand(L, R):
        changed = True
        while changed:
            L = expand_L(L, R)
            R = expand_R(L, R)
            # Check if anything changed
            if not (L or R):
                break
            # Recompute: we need to detect if sizes changed
            # Simpler: just loop a few times
            changed = False
        return L, R

    def expand_biclique(u, v):
        L = {u}
        R = {v}
        for _ in range(20):
            L2 = expand_L(L, R)
            R2 = expand_R(L2, R)
            if L2 == L and R2 == R:
                break
            L, R = L2, R2
        return L, R

    def find_largest_biclique():
        if not remaining:
            return None, None

        # Compute current degrees
        deg = [0] * n
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1

        # Sort edges by degree sum descending, try top few
        edge_list = sorted(remaining, key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)

        best_L, best_R = None, None
        best_size = 0

        # Try at most first 150 edges for speed
        for u, v in edge_list[:150]:
            L, R = expand_biclique(u, v)
            # Verify validity
            if not is_valid_biclique(L, R):
                continue
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_L, best_R = L, R

        if best_L is None:
            # Fallback: pick any edge
            u, v = next(iter(remaining))
            best_L, best_R = {u}, {v}

        return best_L, best_R

    bicliques = []

    while remaining:
        if time.time() > deadline - 2.0:
            break

        L, R = find_largest_biclique()
        if not L or not R:
            u, v = next(iter(remaining))
            L, R = {u}, {v}

        # Collect edges covered by this biclique
        to_remove = set()
        for u, v in remaining:
            if (u in L and v in R) or (u in R and v in L):
                to_remove.add((u, v))

        if not to_remove:
            # Shouldn't happen, but safety
            u, v = next(iter(remaining))
            to_remove = {(u, v)}
            L, R = {u}, {v}

        bicliques.append([sorted(L), sorted(R)])

        for e in to_remove:
            remaining.discard(e)
            adj[e[0]][e[1]] = False
            adj[e[1]][e[0]] = False

    # Safety: cover any remaining edges with single-edge bicliques
    while remaining:
        if time.time() > deadline - 0.5:
            break
        u, v = next(iter(remaining))
        bicliques.append([[u], [v]])
        remaining.discard((u, v))

    # Final safety: if still remaining, force-add
    while remaining:
        u, v = next(iter(remaining))
        bicliques.append([[u], [v]])
        remaining.discard((u, v))

    # Convert to proper types
    result = []
    for b in bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        if left and right:
            result.append([left, right])

    return result

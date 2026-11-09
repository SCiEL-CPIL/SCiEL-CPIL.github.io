"""Biclique partition of a random graph – iterative max-biclique extraction."""
import time
import itertools

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_adj(n, edges):
    """Return adjacency as a set of frozensets for fast membership and a list of edge tuples."""
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    return adj, edge_set


def _find_max_biclique(adj, n, edge_set, deadline):
    """
    Heuristic search for a large biclique in the current residual graph.
    Tries all pairs of vertices as seeds and expands greedily.
    Returns (left, right, edge_count) for the best biclique found.
    """
    best_left = None
    best_right = None
    best_count = 0

    vertices = list(range(n))
    # Only consider vertices with at least one remaining edge
    active = [v for v in vertices if adj[v]]
    if not active:
        return None, None, 0

    # Try seeds: for each vertex u, try each neighbor v as a starting pair
    # and expand to find the largest biclique containing (u,v) on opposite sides
    # We also try single-vertex seeds (u on left, all neighbors on right, then expand)

    # Strategy: enumerate candidate left sets of size 1, 2, 3
    # For each candidate left set L, the right set is the intersection of neighbors of all in L
    # Then we can try to add more left vertices whose neighbors are all in R

    # First pass: single-vertex left seeds
    for u in active:
        if time.time() > deadline:
            break
        # Right side starts as all neighbors of u
        R = set(adj[u])
        L = {u}
        # Expand: add vertices w (w != u) such that all neighbors of w that are in R
        # are exactly R (i.e., adj[w] & R == R) — no, we need adj[w] ⊇ R for w to join left
        # Actually for biclique: every left vertex must connect to every right vertex.
        # So w can join L if R ⊆ adj[w].
        # And a vertex r can stay in R only if it connects to all of L.
        # Let's do iterative expansion:
        # Start with L={u}, R=N(u).
        # Try to grow L: for each w not in L, if R ⊆ adj[w], add w to L.
        # After growing L, shrink R to intersection of adj[w] for w in L.
        # Repeat until stable.
        L_set = {u}
        R_set = set(adj[u])
        changed = True
        while changed:
            changed = False
            # Grow L
            for w in active:
                if w not in L_set and R_set.issubset(adj[w]):
                    L_set.add(w)
                    changed = True
            # Shrink R
            new_R = set(adj[list(L_set)[0]])
            for w in L_set:
                new_R &= adj[w]
            if new_R != R_set:
                R_set = new_R
                changed = True

        count = len(L_set) * len(R_set)
        if count > best_count:
            best_count = count
            best_left = sorted(L_set)
            best_right = sorted(R_set)

    # Second pass: pair seeds (u, v) as left side
    for i, u in enumerate(active):
        if time.time() > deadline:
            break
        for v in active[i+1:]:
            if time.time() > deadline:
                break
            # Left = {u, v}, Right = N(u) ∩ N(v)
            R_set = adj[u] & adj[v]
            if not R_set:
                continue
            L_set = {u, v}
            changed = True
            while changed:
                changed = False
                # Grow L
                for w in active:
                    if w not in L_set and R_set.issubset(adj[w]):
                        L_set.add(w)
                        changed = True
                # Shrink R
                new_R = set(adj[list(L_set)[0]])
                for w in L_set:
                    new_R &= adj[w]
                if new_R != R_set:
                    R_set = new_R
                    changed = True

            count = len(L_set) * len(R_set)
            if count > best_count:
                best_count = count
                best_left = sorted(L_set)
                best_right = sorted(R_set)

    # Third pass: triple seeds
    for i, u in enumerate(active):
        if time.time() > deadline:
            break
        for j, v in enumerate(active[i+1:], i+1):
            if time.time() > deadline:
                break
            for w in active[j+1:]:
                if time.time() > deadline:
                    break
                R_set = adj[u] & adj[v] & adj[w]
                if not R_set:
                    continue
                L_set = {u, v, w}
                changed = True
                while changed:
                    changed = False
                    for x in active:
                        if x not in L_set and R_set.issubset(adj[x]):
                            L_set.add(x)
                            changed = True
                    new_R = set(adj[list(L_set)[0]])
                    for x in L_set:
                        new_R &= adj[x]
                    if new_R != R_set:
                        R_set = new_R
                        changed = True

                count = len(L_set) * len(R_set)
                if count > best_count:
                    best_count = count
                    best_left = sorted(L_set)
                    best_right = sorted(R_set)

    if best_left is None or best_right is None or best_count == 0:
        return None, None, 0

    return best_left, best_right, best_count


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = P['n']
    adj, edge_set = _build_adj(n, P['edges'])

    bicliques = []
    remaining_edges = set(edge_set)

    while remaining_edges:
        if time.time() > deadline:
            break

        left, right, count = _find_max_biclique(adj, n, edge_set, deadline)

        if left is None or right is None or count == 0:
            # Fallback: pick any remaining edge as a 1x1 biclique
            e = next(iter(remaining_edges))
            u, v = tuple(e)
            left = [u]
            right = [v]
            count = 1
        else:
            # Verify that all cross pairs are actually edges (they should be by construction)
            # Remove covered edges from remaining
            covered = set()
            for u in left:
                for v in right:
                    fs = frozenset((u, v))
                    if fs in remaining_edges:
                        covered.add(fs)

            if not covered:
                # Shouldn't happen, but safety
                e = next(iter(remaining_edges))
                u, v = tuple(e)
                left = [u]
                right = [v]
                covered = {frozenset((u, v))}

            remaining_edges -= covered

        # Update adjacency for residual
        for u in left:
            for v in right:
                if v in adj[u]:
                    adj[u].discard(v)
                if u in adj[v]:
                    adj[v].discard(u)

        bicliques.append([list(left), list(right)])

    # Fallback: if we ran out of time with edges remaining, cover them as single edges
    if remaining_edges and time.time() < deadline:
        for e in list(remaining_edges):
            if time.time() > deadline:
                break
            u, v = tuple(e)
            bicliques.append([[u], [v]])
            remaining_edges.discard(e)

    # Ensure types are correct: Python int, list
    result = []
    for bic in bicliques:
        L = [int(x) for x in bic[0]]
        R = [int(x) for x in bic[1]]
        result.append([L, R])

    return result
# EVOLVE-BLOCK-END

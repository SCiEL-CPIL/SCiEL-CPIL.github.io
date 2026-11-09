"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85.0

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    uncovered = set(edge_set)
    if not uncovered:
        return []

    bicliques = []

    def find_maximal_biclique_in_uncovered(start_u, start_v):
        """Find maximal biclique containing (start_u, start_v) in the uncovered subgraph."""
        L = {start_u}
        R = {start_v}
        changed = True
        while changed:
            changed = False
            # Expand L: add w not in R such that w is connected to all of R in uncovered
            for w in range(n):
                if w in R or w in L:
                    continue
                all_conn = True
                for r in R:
                    a, b = min(w, r), max(w, r)
                    if (a, b) not in uncovered:
                        all_conn = False
                        break
                if all_conn:
                    L.add(w)
                    changed = True
            # Expand R: add w not in L such that w is connected to all of L in uncovered
            for w in range(n):
                if w in L or w in R:
                    continue
                all_conn = True
                for l in L:
                    a, b = min(w, l), max(w, l)
                    if (a, b) not in uncovered:
                        all_conn = False
                        break
                if all_conn:
                    R.add(w)
                    changed = True
        return L, R

    def biclique_edges(L, R):
        """Return set of uncovered edges covered by biclique (L, R)."""
        result = set()
        for l in L:
            for r in R:
                a, b = min(l, r), max(l, r)
                if (a, b) in uncovered:
                    result.add((a, b))
        return result

    def score_edge(e):
        """Score an edge by how many common neighbors it has in the uncovered subgraph."""
        u, v = e
        # Common neighbors in uncovered: w such that (u,w) and (v,w) both in uncovered
        common = 0
        for w in range(n):
            if w == u or w == v:
                continue
            a1, b1 = min(u, w), max(u, w)
            a2, b2 = min(v, w), max(v, w)
            if (a1, b1) in uncovered and (a2, b2) in uncovered:
                common += 1
        return common

    while uncovered:
        if time.time() > deadline:
            for e in list(uncovered):
                u, v = e
                bicliques.append([[u], [v]])
            break

        # Pick the edge with the most common neighbors in uncovered subgraph
        best_edge = None
        best_score = -1
        for e in uncovered:
            s = score_edge(e)
            if s > best_score:
                best_score = s
                best_edge = e

        u, v = best_edge

        # Find maximal biclique containing this edge
        L, R = find_maximal_biclique_in_uncovered(u, v)
        R = R - L

        if not L or not R:
            bicliques.append([[u], [v]])
            uncovered.discard((min(u, v), max(u, v)))
            continue

        # Try both orientations to see which covers more edges
        L1, R1 = L, R
        L2, R2 = R, L
        covered1 = biclique_edges(L1, R1)
        covered2 = biclique_edges(L2, R2)

        if len(covered2) > len(covered1):
            L, R = L2, R2
            covered = covered2
        else:
            covered = covered1

        if not covered:
            bicliques.append([[u], [v]])
            uncovered.discard((min(u, v), max(u, v)))
            continue

        bicliques.append([list(L), list(R)])
        uncovered -= covered

    # Post-processing: try to merge bicliques
    # Two bicliques can be merged if their union is also a valid biclique
    # i.e., all cross edges between (L1 ∪ L2) and (R1 ∪ R2) are in the original graph
    # AND the cross-edge sets partition (no edge appears in two bicliques)
    # Actually for partition, we need: the cross edges of biclique1 and biclique2 are disjoint
    # and the union's cross edges = cross1 ∪ cross2 (no new edges needed)
    # This means: every (l in L1, r in R2) and (l in L2, r in R1) must NOT be in the graph
    # (otherwise they'd be extra cross edges not in either biclique)

    def can_merge(b1, b2):
        L1, R1 = set(b1[0]), set(b1[1])
        L2, R2 = set(b2[0]), set(b2[1])
        # Check that no cross edges exist between L1-R2 or L2-R1 in the original graph
        for l in L1:
            for r in R2:
                a, b = min(l, r), max(l, r)
                if (a, b) in edge_set:
                    return False
        for l in L2:
            for r in R1:
                a, b = min(l, r), max(l, r)
                if (a, b) in edge_set:
                    return False
        return True

    def merge_bicliques(b1, b2):
        L1, R1 = set(b1[0]), set(b1[1])
        L2, R2 = set(b2[0]), set(b2[1])
        return [list(L1 | L2), list(R1 | R2)]

    # Greedy merging: repeatedly try to merge pairs
    merged = True
    while merged:
        if time.time() > deadline:
            break
        merged = False
        new_list = []
        used = [False] * len(bicliques)
        for i in range(len(bicliques)):
            if used[i]:
                continue
            best_j = -1
            for j in range(i + 1, len(bicliques)):
                if used[j]:
                    continue
                if can_merge(bicliques[i], bicliques[j]):
                    best_j = j
                    break
            if best_j >= 0:
                new_list.append(merge_bicliques(bicliques[i], bicliques[best_j]))
                used[i] = True
                used[best_j] = True
                merged = True
            else:
                new_list.append(bicliques[i])
                used[i] = True
        bicliques = new_list

    # Convert to proper types
    result = []
    for bic in bicliques:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        if left and right and not (set(left) & set(right)):
            result.append([left, right])

    return result
# EVOLVE-BLOCK-END

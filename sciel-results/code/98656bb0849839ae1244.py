"""Rewritten program: biclique partition via maximal-biclique set cover + merge + local search."""
import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = P['n']
ALL_EDGES = set((min(u, v), max(u, v)) for u, v in P['edges'])

# Precompute adjacency as bitmasks
_adj_mask = [0] * N
for u, v in P['edges']:
    _adj_mask[u] |= (1 << v)
    _adj_mask[v] |= (1 << u)

# Precompute neighbor sets for quick lookup
_adj_set = [set() for _ in range(N)]
for u, v in P['edges']:
    _adj_set[u].add(v)
    _adj_set[v].add(u)


def _all_max_bicliques():
    """
    Enumerate all maximal bicliques in the graph using a Bron-Kerbosch variant
    on the bipartite double cover. Each vertex v is split into v_L and v_R.
    An edge (u,v) in the original graph gives edges u_L--v_R and v_L--u_R in the double cover.
    A maximal clique in the double cover that contains vertices from both sides (and no
    vertex is both L and R of the same original vertex) corresponds to a maximal biclique.
    
    We use a simpler approach: enumerate all maximal bicliques by trying all pairs of
    subsets. For a graph with 30 vertices, we use a branch-and-bound approach.
    """
    bicliques = []
    seen = set()

    # For each pair of vertices that form an edge, try to extend to a maximal biclique
    # Use a recursive approach with pruning
    def extend_biclique(left, right, candidate_left, candidate_right):
        """
        left, right: current biclique sides
        candidate_left: vertices that can be added to left (connected to all of right)
        candidate_right: vertices that can be added to right (connected to all of left)
        """
        if not candidate_left and not candidate_right:
            key = (frozenset(left), frozenset(right))
            if key not in seen:
                seen.add(key)
                bicliques.append((frozenset(left), frozenset(right)))
            return

        # Try adding a vertex from candidate_right
        if candidate_right:
            v = min(candidate_right)
            # Include v in right
            new_right = right | {v}
            new_candidate_left = {u for u in candidate_left if v in _adj_set[u]}
            new_candidate_right = {u for u in candidate_right if u != v}
            # Also, vertices in left that are connected to v stay; vertices not in left but connected to all new_right
            # Actually we need to recompute: new_candidate_left should be vertices connected to all of new_right
            new_candidate_left2 = set()
            for u in range(N):
                if u not in left and u not in right and all(w in _adj_set[u] for w in new_right):
                    new_candidate_left2.add(u)
            extend_biclique(left, new_right, new_candidate_left2, new_candidate_right)

            # Exclude v from right
            extend_biclique(left, right, candidate_left, candidate_right - {v})

        elif candidate_left:
            u = min(candidate_left)
            # Include u in left
            new_left = left | {u}
            new_candidate_right = {v for v in candidate_right if u in _adj_set[v]}
            # Recompute properly
            new_candidate_right2 = set()
            for v in range(N):
                if v not in left and v not in right and all(w in _adj_set[v] for w in new_left):
                    new_candidate_right2.add(v)
            extend_biclique(new_left, right, candidate_left - {u}, new_candidate_right2)

            # Exclude u from left
            extend_biclique(left, right, candidate_left - {u}, candidate_right)

    # For each edge, try to build a maximal biclique containing that edge
    # To avoid duplicates, we use a canonical ordering
    for u, v in P['edges']:
        left = {u}
        right = {v}
        # candidate_left: vertices connected to all of right (={v}), not in left or right
        candidate_left = {x for x in _adj_set[v] if x != u and x != v}
        # candidate_right: vertices connected to all of left (={u}), not in left or right
        candidate_right = {x for x in _adj_set[u] if x != u and x != v}
        extend_biclique(left, right, candidate_left, candidate_right)

    return bicliques


def _biclique_edge_set(left, right):
    """Get the set of edges covered by a biclique (only edges that exist in the graph)."""
    edges = set()
    for u in left:
        for v in right:
            e = (min(u, v), max(u, v))
            if e in ALL_EDGES:
                edges.add(e)
    return edges


def _greedy_set_cover(bicliques, deadline):
    """
    Greedy set cover: repeatedly pick the biclique that covers the most uncovered edges.
    """
    remaining = set(ALL_EDGES)
    selected = []

    # Precompute edge sets for all bicliques
    biclique_edges = []
    for left, right in bicliques:
        biclique_edges.append(_biclique_edge_set(left, right))

    while remaining and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for i, edges in enumerate(biclique_edges):
            if not edges:
                continue
            count = len(edges & remaining)
            if count > best_count:
                best_count = count
                best_idx = i

        if best_idx == -1:
            break

        left, right = bicliques[best_idx]
        selected.append((left, right))
        remaining -= biclique_edges[best_idx]

    # Cover any remaining edges with single-edge bicliques
    for e in sorted(remaining):
        selected.append((frozenset({e[0]}), frozenset({e[1]})))

    return selected


def _try_merge(selected, deadline):
    """Try to merge pairs of bicliques if their union forms a valid biclique."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        i = 0
        while i < len(selected) and time.time() < deadline:
            j = i + 1
            while j < len(selected) and time.time() < deadline:
                li, ri = selected[i]
                lj, rj = selected[j]

                # Try merging: new_left = li ∪ lj, new_right = ri
                # Valid if all of lj connect to all of ri, and li ∩ ri = ∅, lj ∩ ri = ∅
                new_left = li | lj
                new_right = ri
                if not (new_left & new_right):
                    valid = all(w in _adj_set[u] for u in new_left for w in new_right)
                    if valid:
                        # Check that the union of edges is exactly the union of the two bicliques' edges
                        # (i.e., no extra edges are introduced that weren't covered before)
                        old_edges = _biclique_edge_set(li, ri) | _biclique_edge_set(lj, rj)
                        new_edges = _biclique_edge_set(new_left, new_right)
                        if new_edges == old_edges:
                            selected[i] = (new_left, new_right)
                            del selected[j]
                            changed = True
                            break

                # Try: new_left = li, new_right = ri ∪ rj
                new_left2 = li
                new_right2 = ri | rj
                if not (new_left2 & new_right2):
                    valid = all(w in _adj_set[u] for u in new_left2 for w in new_right2)
                    if valid:
                        old_edges = _biclique_edge_set(li, ri) | _biclique_edge_set(lj, rj)
                        new_edges = _biclique_edge_set(new_left2, new_right2)
                        if new_edges == old_edges:
                            selected[i] = (new_left2, new_right2)
                            del selected[j]
                            changed = True
                            break

                # Try: new_left = li ∪ lj, new_right = ri ∪ rj
                new_left3 = li | lj
                new_right3 = ri | rj
                if not (new_left3 & new_right3):
                    valid = all(w in _adj_set[u] for u in new_left3 for w in new_right3)
                    if valid:
                        old_edges = _biclique_edge_set(li, ri) | _biclique_edge_set(lj, rj)
                        new_edges = _biclique_edge_set(new_left3, new_right3)
                        if new_edges == old_edges:
                            selected[i] = (new_left3, new_right3)
                            del selected[j]
                            changed = True
                            break

                j += 1
            i += 1

    return selected


def _local_search(selected, deadline):
    """
    Local search: try to move a vertex from one biclique to another to reduce the total count
    or to enable further merges.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            for j in range(len(selected)):
                if i == j or time.time() > deadline:
                    continue
                li, ri = selected[i]
                lj, rj = selected[j]

                # Try moving a vertex from li to lj (or vice versa)
                for u in list(li):
                    # Move u from li to lj
                    new_li = li - {u}
                    new_lj = lj | {u}
                    if not new_li or not new_lj:
                        continue
                    if new_li & ri or new_lj & rj:
                        continue
                    valid = (all(w in _adj_set[x] for x in new_li for w in ri) and
                             all(w in _adj_set[x] for x in new_lj for w in rj))
                    if valid:
                        # Check that the edge sets still partition ALL_EDGES
                        old_edges = _biclique_edge_set(li, ri) | _biclique_edge_set(lj, rj)
                        new_edges = _biclique_edge_set(new_li, ri) | _biclique_edge_set(new_lj, rj)
                        if new_edges == old_edges:
                            selected[i] = (new_li, ri)
                            selected[j] = (new_lj, rj)
                            improved = True
                            break

                if improved:
                    break

                # Try moving a vertex from ri to rj
                for v in list(ri):
                    new_ri = ri - {v}
                    new_rj = rj | {v}
                    if not new_ri or not new_rj:
                        continue
                    if li & new_rj or lj & new_ri:
                        continue
                    valid = (all(w in _adj_set[x] for x in li for w in new_ri) and
                             all(w in _adj_set[x] for x in lj for w in new_rj))
                    if valid:
                        old_edges = _biclique_edge_set(li, ri) | _biclique_edge_set(lj, rj)
                        new_edges = _biclique_edge_set(li, new_ri) | _biclique_edge_set(lj, new_rj)
                        if new_edges == old_edges:
                            selected[i] = (li, new_ri)
                            selected[j] = (lj, new_rj)
                            improved = True
                            break

                if improved:
                    break

            if improved:
                break

    return selected


def _validate(selected):
    """Validate the partition."""
    covered = set()
    for left, right in selected:
        if not left or not right:
            return False
        if left & right:
            return False
        for u in left:
            for v in right:
                if v not in _adj_set[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    return covered == ALL_EDGES


def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill

    # Step 1: Enumerate all maximal bicliques
    bicliques = _all_max_bicliques()

    # Step 2: Greedy set cover
    selected = _greedy_set_cover(bicliques, deadline)

    # Step 3: Try to merge
    selected = _try_merge(selected, deadline)

    # Step 4: Local search
    if time.time() < deadline:
        selected = _local_search(selected, deadline)

    # Step 5: Validate and fallback
    if not _validate(selected):
        # Fallback: one biclique per edge
        selected = [(frozenset({u}), frozenset({v})) for u, v in P['edges']]

    # Convert to required format
    result = []
    for left, right in selected:
        left_list = [int(x) for x in sorted(left)]
        right_list = [int(x) for x in sorted(right)]
        result.append([left_list, right_list])

    return result

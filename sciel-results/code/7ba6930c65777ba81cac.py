"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edges_to_set(edges):
    s = set()
    for u, v in edges:
        s.add((min(u, v), max(u, v)))
    return s

def _find_biclique_from_anchor(adj, remaining, anchor):
    """Find a biclique anchored at `anchor` using only remaining edges."""
    n = P['n']
    right = set()
    for v in adj[anchor]:
        if (min(anchor, v), max(anchor, v)) in remaining:
            right.add(v)
    if not right:
        return None, None

    left = {anchor}
    # Expand left: any vertex connected (in graph) to all of right and with a remaining edge to right
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if v in left:
                continue
            if not all(v in adj[w] for w in right):
                continue
            has_rem = False
            for w in right:
                if (min(v, w), max(v, w)) in remaining:
                    has_rem = True
                    break
            if has_rem:
                left.add(v)
                changed = True

    # Shrink right to those connected to all of left
    right = {w for w in right if all(w in adj[u] for u in left)}
    # Shrink left to those connected to all of right
    left = {u for u in left if all(u in adj[v] for v in right)}

    if not left or not right:
        return None, None
    return list(left), list(right)

def _greedy_max_biclique_partition(adj, remaining, deadline):
    n = P['n']
    bicliques = []
    while remaining:
        if time.time() > deadline:
            break
        deg = {}
        for u, v in remaining:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        sorted_v = sorted(deg.keys(), key=lambda x: -deg[x])
        best_left, best_right, best_edges = None, None, 0
        for anchor in sorted_v[:30]:
            left, right = _find_biclique_from_anchor(adj, remaining, anchor)
            if left is None:
                continue
            count = 0
            for u in left:
                for v in right:
                    if (min(u, v), max(u, v)) in remaining:
                        count += 1
            if count > best_edges:
                best_edges = count
                best_left = left
                best_right = right
        if best_left is None or best_edges == 0:
            e = next(iter(remaining))
            u, v = e
            bicliques.append([[u], [v]])
            remaining.discard(e)
            continue
        for u in best_left:
            for v in best_right:
                remaining.discard((min(u, v), max(u, v)))
        bicliques.append([list(best_left), list(best_right)])
    for e in list(remaining):
        u, v = e
        bicliques.append([[u], [v]])
        remaining.discard(e)
    return bicliques

def _validate(bicliques, edges_set):
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if u == v:
                    return False
                e = (min(u, v), max(u, v))
                if e not in edges_set:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == edges_set

def _try_merge(bicliques, adj, edges_set, deadline):
    """Merge two bicliques when the new cross-edges are a subset of their union."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        i = 0
        while i < len(bicliques) and time.time() < deadline:
            j = i + 1
            while j < len(bicliques) and time.time() < deadline:
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                configs = [
                    (set(L1) | set(L2), set(R1)),
                    (set(L1) | set(L2), set(R2)),
                    (set(L1), set(R1) | set(R2)),
                    (set(L2), set(R1) | set(R2)),
                    (set(L1) | set(L2), set(R1) | set(R2)),
                ]
                merged = False
                for new_L, new_R in configs:
                    if not new_L or not new_R:
                        continue
                    valid = True
                    for u in new_L:
                        if u in new_R:
                            valid = False
                            break
                        for v in new_R:
                            if v not in adj[u]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    new_edges = set()
                    for u in new_L:
                        for v in new_R:
                            new_edges.add((min(u, v), max(u, v)))
                    old_edges = set()
                    for u in L1:
                        for v in R1:
                            old_edges.add((min(u, v), max(u, v)))
                    for u in L2:
                        for v in R2:
                            old_edges.add((min(u, v), max(u, v)))
                    if new_edges <= old_edges:
                        bicliques[i] = [list(new_L), list(new_R)]
                        del bicliques[j]
                        changed = True
                        merged = True
                        break
                if merged:
                    break
                j += 1
            i += 1
    return bicliques

def _try_absorb(bicliques, adj, deadline):
    """Grow each biclique by adding vertices whose cross-edges are all already covered by it."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            L, R = bicliques[i]
            Ls, Rs = set(L), set(R)
            bic_edges = set()
            for u in L:
                for v in R:
                    bic_edges.add((min(u, v), max(u, v)))
            # Try adding a vertex to L
            for v in range(P['n']):
                if v in Ls or v in Rs:
                    continue
                ok = True
                for w in R:
                    if (min(v, w), max(v, w)) not in bic_edges:
                        ok = False
                        break
                if ok:
                    Ls.add(v)
                    changed = True
                    break
            else:
                # Try adding a vertex to R
                for u in range(P['n']):
                    if u in Ls or u in Rs:
                        continue
                    ok = True
                    for w in Ls:
                        if (min(u, w), max(u, w)) not in bic_edges:
                            ok = False
                            break
                    if ok:
                        Rs.add(u)
                        changed = True
                        break
            if changed:
                bicliques[i] = [list(Ls), list(Rs)]
    return bicliques

def _local_search(bicliques, adj, edges_set, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_bicliques = _try_absorb(bicliques, adj, deadline)
        if len(new_bicliques) < len(bicliques):
            bicliques = new_bicliques
            improved = True
        if time.time() < deadline:
            new_bicliques = _try_merge(bicliques, adj, edges_set, deadline)
            if len(new_bicliques) < len(bicliques):
                bicliques = new_bicliques
                improved = True
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin under 90s
    adj = _build_adj()
    edges_set = _edges_to_set(P['edges'])
    n = P['n']

    def star_partition():
        result = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                result.append([list(neighbors), [u]])
        return result

    remaining1 = set(edges_set)
    result1 = _greedy_max_biclique_partition(adj, remaining1, deadline)
    if not _validate(result1, edges_set):
        result1 = star_partition()
    best = result1

    if time.time() < deadline - 5:
        improved = _local_search(list(best), adj, edges_set, deadline - 5)
        if _validate(improved, edges_set) and len(improved) <= len(best):
            best = improved

    # Strategy 2: anchor by original degree
    if time.time() < deadline - 10:
        degrees = [len(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        remaining2 = set(edges_set)
        bicliques2 = []
        for anchor in order:
            if not remaining2 or time.time() > deadline - 5:
                break
            left, right = _find_biclique_from_anchor(adj, remaining2, anchor)
            if left is None:
                continue
            for u in left:
                for v in right:
                    remaining2.discard((min(u, v), max(u, v)))
            bicliques2.append([list(left), list(right)])
        for e in list(remaining2):
            u, v = e
            bicliques2.append([[u], [v]])
            remaining2.discard(e)
        if _validate(bicliques2, edges_set) and len(bicliques2) < len(best):
            best = bicliques2
            if time.time() < deadline - 5:
                improved = _local_search(list(best), adj, edges_set, deadline - 5)
                if _validate(improved, edges_set) and len(improved) <= len(best):
                    best = improved

    if not _validate(best, edges_set):
        best = star_partition()

    result = []
    for left, right in best:
        result.append([list(left), list(right)])
    return result

# EVOLVE-BLOCK-END

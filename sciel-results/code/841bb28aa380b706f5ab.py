"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = P['n']
ALL_MASK = (1 << N) - 1


def _build_adj():
    n = P['n']
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    return adj


def _build_edge_set():
    es = set()
    for u, v in P['edges']:
        es.add((min(u, v), max(u, v)))
    return es


def _grow_biclique(adj, L_mask, R_mask, edge_mask, deadline):
    """Maximally grow biclique (L,R) using only edges in edge_mask. Bitmask fast."""
    n = N
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        # Add to L: vertex x not in L∪R with x adjacent to all R, and all edges x-r present
        candidates = ALL_MASK & ~(L_mask | R_mask)
        # For each x in candidates, require (adj[x] & R_mask) == R_mask and edge_mask contains those
        # edge_mask is a bitmask over pairs? We'll precompute per-pair? Too big. Instead check via adj & R and edge presence.
        # Since edge_mask is set of edges, checking each pair is expensive; rely on adj only if edge_mask is full.
        # For candidate generation we use full graph (edge_mask full). For greedy partition we restrict.
        # Here we assume edge_mask is full graph adjacency (adj). Use adj.
        for x in range(n):
            if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                continue
            if (adj[x] & R_mask) == R_mask:
                L_mask |= (1 << x)
                changed = True
        for x in range(n):
            if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                continue
            if (adj[x] & L_mask) == L_mask:
                R_mask |= (1 << x)
                changed = True
    return L_mask, R_mask


def _grow_biclique_restricted(adj, edge_set, L_mask, R_mask, deadline):
    """Grow using only edges present in edge_set (for greedy partition)."""
    n = N
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        for x in range(n):
            if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                continue
            ok = True
            for r in range(n):
                if (R_mask >> r) & 1:
                    if (min(x, r), max(x, r)) not in edge_set:
                        ok = False
                        break
            if ok:
                L_mask |= (1 << x)
                changed = True
        for x in range(n):
            if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                continue
            ok = True
            for l in range(n):
                if (L_mask >> l) & 1:
                    if (min(x, l), max(x, l)) not in edge_set:
                        ok = False
                        break
            if ok:
                R_mask |= (1 << x)
                changed = True
    return L_mask, R_mask


def _mask_to_list(mask):
    return [i for i in range(N) if (mask >> i) & 1]


def _find_maximal_bicliques(adj, edge_set, deadline, max_bicliques=20000):
    """Find many (near)maximal bicliques via bitmask expansion from seed edges."""
    bicliques = []
    seen = set()
    seeds = list(edge_set)
    random.shuffle(seeds)
    for u, v in seeds:
        if time.time() > deadline or len(bicliques) >= max_bicliques:
            break
        L_mask = (1 << u)
        R_mask = (1 << v)
        L_mask, R_mask = _grow_biclique(adj, L_mask, R_mask, None, deadline)
        L = _mask_to_list(L_mask)
        R = _mask_to_list(R_mask)
        if L and R:
            key = (frozenset(L), frozenset(R))
            key2 = (frozenset(R), frozenset(L))
            if key not in seen and key2 not in seen:
                seen.add(key)
                seen.add(key2)
                bicliques.append((L, R))
    return bicliques


def _greedy_partition(adj, edge_set, deadline):
    """Greedy: repeatedly pick largest biclique among remaining edges."""
    remaining = set(edge_set)
    bicliques = []
    while remaining:
        if time.time() > deadline:
            break
        deg = {}
        for (u, v) in remaining:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        anchors = sorted(deg.keys(), key=lambda x: -deg[x])
        best = None
        best_size = 0
        for anchor in anchors[:30]:
            if time.time() > deadline:
                break
            # Right candidates: neighbors of anchor with remaining edge
            right_mask = 0
            for (u, v) in remaining:
                if u == anchor:
                    right_mask |= (1 << v)
                elif v == anchor:
                    right_mask |= (1 << u)
            if right_mask == 0:
                continue
            L_mask = (1 << anchor)
            R_mask = right_mask
            L_mask, R_mask = _grow_biclique_restricted(adj, remaining, L_mask, R_mask, deadline)
            size = bin(L_mask).count('1') * bin(R_mask).count('1')
            if size > best_size:
                best_size = size
                best = (L_mask, R_mask)
        if best is None:
            e = next(iter(remaining))
            best = (1 << e[0], 1 << e[1])
        L_mask, R_mask = best
        L = _mask_to_list(L_mask)
        R = _mask_to_list(R_mask)
        bicliques.append((L, R))
        for u in L:
            for v in R:
                remaining.discard((min(u, v), max(u, v)))
    return bicliques


def _merge_pass(bicliques, adj, edge_set, deadline):
    """Merge compatible bicliques using multiple orientations."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(bicliques)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                l1, r1 = bicliques[i]
                l2, r2 = bicliques[j]
                s1, s2, s3, s4 = set(l1), set(r1), set(l2), set(r2)
                merged = None
                for (nl, nr) in [
                    (s1 | s2, s3 | s4),
                    (s1 | s2, s3 & s4),
                    (s1 & s2, s3 | s4),
                    (s1 | s4, s3 & s2),
                    (s2 | s3, s4 & s1),
                ]:
                    if not nl or not nr:
                        continue
                    if nl & nr:
                        continue
                    valid = True
                    for u in nl:
                        for v in nr:
                            if v not in adj[u]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # ensure no extra edges beyond union
                        merged_edges = set()
                        for u in nl:
                            for v in nr:
                                merged_edges.add((min(u, v), max(u, v)))
                        old = set()
                        for u in s1:
                            for v in s3:
                                old.add((min(u, v), max(u, v)))
                        for u in s2:
                            for v in s4:
                                old.add((min(u, v), max(u, v)))
                        if merged_edges == old:
                            merged = (list(nl), list(nr))
                            break
                if merged:
                    bicliques[i] = merged
                    bicliques.pop(j)
                    changed = True
                    break
    return bicliques


def _set_cover(edge_set, candidates, deadline):
    """Exact min set cover over candidate bicliques via backtracking."""
    edge_list = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    num_edges = len(edge_list)

    cand_edges = []
    cand_verts = []
    seen = set()
    for L, R in candidates:
        es = set()
        ok = True
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in edge_to_idx:
                    es.add(edge_to_idx[e])
                else:
                    ok = False
                    break
            if not ok:
                break
        if ok and es:
            key = frozenset(es)
            if key not in seen:
                seen.add(key)
                cand_edges.append(key)
                cand_verts.append((L, R))

    num_cand = len(cand_edges)
    if num_cand == 0:
        return None

    edge_to_cand = [set() for _ in range(num_edges)]
    for i, be in enumerate(cand_edges):
        for e in be:
            edge_to_cand[e].add(i)

    # Greedy upper bound
    def greedy():
        rem = set(range(num_edges))
        chosen = []
        while rem:
            best_i = None
            best_c = 0
            for i in range(num_cand):
                c = len(cand_edges[i] & rem)
                if c > best_c:
                    best_c = c
                    best_i = i
            if best_i is None:
                break
            chosen.append(best_i)
            rem -= cand_edges[best_i]
        return chosen

    g = greedy()
    best_size = len(g)
    best_solution = list(g)

    # Also try greedy with random restarts for a tighter bound
    for t in range(8):
        if time.time() > deadline:
            break
        random.seed(1000 + t)
        rem = set(range(num_edges))
        chosen = []
        while rem:
            best_i = None
            best_c = -1
            for i in range(num_cand):
                c = len(cand_edges[i] & rem)
                # tie-break by random
                score = c * 10000 + random.random()
                if c > best_c or (c == best_c and score > (best_c * 10000)):
                    pass
                if c > best_c:
                    best_c = c
                    best_i = i
            if best_i is None:
                break
            chosen.append(best_i)
            rem -= cand_edges[best_i]
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_solution = list(chosen)

    memo = {}
    nodes = [0]

    def backtrack(uncovered, cur, depth):
        nonlocal best_solution, best_size
        nodes[0] += 1
        if time.time() > deadline or nodes[0] > 2000000:
            return
        if not uncovered:
            if depth < best_size:
                best_size = depth
                best_solution = list(cur)
            return
        if depth >= best_size:
            return
        # lower bound
        max_cov = 0
        for i in range(num_cand):
            if cand_edges[i] & uncovered:
                c = len(cand_edges[i] & uncovered)
                if c > max_cov:
                    max_cov = c
        if max_cov == 0:
            return
        lb = (len(uncovered) + max_cov - 1) // max_cov
        if depth + lb >= best_size:
            return
        state = frozenset(uncovered)
        if state in memo:
            if memo[state] <= depth:
                return
        else:
            memo[state] = depth
        # most constrained edge
        min_c = float('inf')
        min_e = None
        for e in uncovered:
            c = len(edge_to_cand[e])
            if c < min_c:
                min_c = c
                min_e = e
        if min_e is None or min_c == 0:
            return
        cands = sorted(edge_to_cand[min_e],
                       key=lambda bi: (-len(cand_edges[bi] & uncovered), -len(cand_edges[bi])))
        for bi in cands:
            new_unc = uncovered - cand_edges[bi]
            cur.append(bi)
            backtrack(new_unc, cur, depth + 1)
            cur.pop()

    backtrack(frozenset(range(num_edges)), [], 0)

    if best_solution:
        return [cand_verts[i] for i in best_solution]
    return None


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s

    edge_set = _build_edge_set()
    adj = _build_adj()

    # Phase 1: greedy partition for initial bound
    greedy = _greedy_partition(adj, edge_set, deadline)
    greedy = _merge_pass(greedy, adj, edge_set, deadline)
    greedy_count = len(greedy)

    # Phase 2: candidate bicliques
    candidates = _find_maximal_bicliques(adj, edge_set, deadline, max_bicliques=20000)

    # Phase 3: exact set cover
    exact = None
    if time.time() < deadline - 5.0:
        exact = _set_cover(edge_set, candidates, deadline - 2.0)

    # Choose best
    if exact and len(exact) < greedy_count:
        result = exact
    else:
        result = greedy

    # Final merge pass
    result = _merge_pass(result, adj, edge_set, deadline)

    # Convert to proper format
    out = []
    for L, R in result:
        L_list = [int(v) for v in L]
        R_list = [int(v) for v in R]
        if L_list and R_list:
            out.append([L_list, R_list])

    return out
# EVOLVE-BLOCK-END

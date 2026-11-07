"""Exact biclique edge-partition: rich candidate generation + exact-cover + greedy + merge/split/relocate polish."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

adj_global = None


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _edge_set(L, R, edges_set):
    s = set()
    for u in L:
        for v in R:
            if u == v:
                continue
            e = (u, v) if u < v else (v, u)
            if e in edges_set:
                s.add(e)
    return s


def _expand_biclique(seedL, seedR, adj, n):
    """Maximally expand a valid biclique (L,R) by adding vertices adjacent to all of the opposite side."""
    L = set(seedL)
    R = set(seedR)
    if not L or not R:
        return None
    # Quick validity check
    for u in L:
        for v in R:
            if v not in adj[u]:
                return None
    changed = True
    while changed:
        changed = False
        for x in range(n):
            if x in L or x in R:
                continue
            if all(x in adj[r] for r in R):
                L.add(x)
                changed = True
        for x in range(n):
            if x in L or x in R:
                continue
            if all(x in adj[l] for l in L):
                R.add(x)
                changed = True
    if not L or not R:
        return None
    return (sorted(L), sorted(R))


def _find_maximal_bicliques(adj, edges_set, n):
    maximal = []
    seen = set()
    for u, v in edges_set:
        res = _expand_biclique({u}, {v}, adj, n)
        if not res:
            continue
        L, R = res
        k1 = (frozenset(L), frozenset(R))
        k2 = (frozenset(R), frozenset(L))
        if k1 not in seen and k2 not in seen:
            seen.add(k1)
            seen.add(k2)
            maximal.append((L, R))
    return maximal


def _find_pair_bicliques(adj, edges_set, n, max_pairs=600):
    """Bicliques from pairs of edges sharing a vertex (captures small bicliques)."""
    bic = []
    seen = set()
    elist = list(edges_set)
    # Build adjacency: vertex -> list of incident edges
    vert_edges = {v: [] for v in range(n)}
    for e in elist:
        vert_edges[e[0]].append(e)
        vert_edges[e[1]].append(e)
    cnt = 0
    for v in range(n):
        if cnt >= max_pairs:
            break
        inc = vert_edges[v]
        if len(inc) < 2:
            continue
        for i in range(len(inc)):
            if cnt >= max_pairs:
                break
            u1 = inc[i][0] if inc[i][0] != v else inc[i][1]
            for j in range(i + 1, len(inc)):
                cnt += 1
                if cnt >= max_pairs:
                    break
                u2 = inc[j][0] if inc[j][0] != v else inc[j][1]
                res = _expand_biclique({v}, {u1, u2}, adj, n)
                if not res:
                    continue
                L, R = res
                k1 = (frozenset(L), frozenset(R))
                k2 = (frozenset(R), frozenset(L))
                if k1 not in seen and k2 not in seen:
                    seen.add(k1)
                    seen.add(k2)
                    bic.append((L, R))
    return bic


def _find_greedy_bicliques(adj, edges_set, n, time_limit=3.0):
    start = time.time()
    bic = []
    seen = set()
    deg = [0] * n
    for u, v in edges_set:
        deg[u] += 1
        deg[v] += 1
    cands = [v for v in range(n) if deg[v] > 0]
    cands.sort(key=lambda x: -deg[x])
    for anchor in cands[:min(len(cands), 60)]:
        if time.time() - start > time_limit:
            break
        L = {anchor}
        R = set(adj[anchor])
        # Refinement via intersection
        changed = True
        while changed:
            changed = False
            if L:
                nr = set(adj[list(L)[0]])
                for u in list(L)[1:]:
                    nr &= adj[u]
                nr &= R
                if nr != R:
                    R = nr
                    changed = True
            if R:
                nl = set(adj[list(R)[0]])
                for v in list(R)[1:]:
                    nl &= adj[v]
                nl &= L
                if nl != L:
                    L = nl
                    changed = True
        if not L or not R:
            continue
        res = _expand_biclique(L, R, adj, n)
        if not res:
            continue
        L2, R2 = res
        k1 = (frozenset(L2), frozenset(R2))
        k2 = (frozenset(R2), frozenset(L2))
        if k1 not in seen and k2 not in seen:
            seen.add(k1)
            seen.add(k2)
            bic.append((L2, R2))
    return bic


def _solve_exact_cover(bicliques, edges_set, time_limit=15.0):
    start = time.time()
    bic_edges = []
    valid = []
    for L, R in bicliques:
        e = _edge_set(L, R, edges_set)
        if e:
            bic_edges.append(e)
            valid.append((L, R))
    if not bic_edges:
        return None
    all_edges = set(edges_set)
    edge_to = {}
    for i, es in enumerate(bic_edges):
        for e in es:
            edge_to.setdefault(e, []).append(i)
    for e in all_edges:
        if e not in edge_to:
            return None
    # Precompute bitmasks of edges for each biclique
    all_edges_sorted = sorted(all_edges)
    eidx = {e: i for i, e in enumerate(all_edges_sorted)}
    m = len(eidx)
    masks = []
    for es in bic_edges:
        mask = 0
        for e in es:
            mask |= (1 << eidx[e])
        masks.append(mask)
    # Precompute popcount for each mask
    mask_sizes = [bin(mask).count('1') for mask in masks]
    full = (1 << m) - 1
    best = [None]
    best_cnt = [float('inf')]
    nodes = [0]
    # Precompute for each edge index, list of biclique indices that cover it
    edge_cands = [[] for _ in range(m)]
    for i, es in enumerate(bic_edges):
        for e in es:
            edge_cands[eidx[e]].append(i)

    def bt(covered, current):
        nonlocal best, best_cnt
        nodes[0] += 1
        if time.time() - start > time_limit:
            return
        if len(current) >= best_cnt[0]:
            return
        if covered == full:
            if len(current) < best_cnt[0]:
                best_cnt[0] = len(current)
                best[0] = list(current)
            return
        # MRV: pick uncovered edge with fewest non-conflicting candidates
        best_e = None
        best_cands = None
        best_n = float('inf')
        for e in range(m):
            if (covered >> e) & 1:
                continue
            cands = [b for b in edge_cands[e] if not (covered & masks[b])]
            if len(cands) < best_n:
                best_n = len(cands)
                best_e = e
                best_cands = cands
                if best_n == 0:
                    return
        if best_cands is None or best_n == 0:
            return
        # Sort by coverage (descending) for better pruning
        best_cands.sort(key=lambda b: -mask_sizes[b])
        for b in best_cands:
            current.append(b)
            bt(covered | masks[b], current)
            current.pop()

    bt(0, [])
    if best[0] is not None:
        return [valid[i] for i in best[0]]
    return None


def _validate(bicliques, edges_set):
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if u == v:
                    return False
                e = (u, v) if u < v else (v, u)
                if e not in edges_set:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == edges_set


def _greedy_partition(adj, edges_set, n, time_limit=10.0):
    start = time.time()
    remaining = set(edges_set)
    bic = []
    while remaining:
        if time.time() - start > time_limit:
            break
        rem_deg = [0] * n
        for u, v in remaining:
            rem_deg[u] += 1
            rem_deg[v] += 1
        rem_cands = [v for v in range(n) if rem_deg[v] > 0]
        rem_cands.sort(key=lambda x: -rem_deg[x])
        rem_set = set()
        for u, v in remaining:
            rem_set.add((u, v))
            rem_set.add((v, u))
        best_L = None
        best_R = None
        best_cnt = 0
        for anchor in rem_cands[:min(len(rem_cands), 30)]:
            if time.time() - start > time_limit:
                break
            neigh = set()
            for u, v in remaining:
                if u == anchor:
                    neigh.add(v)
                elif v == anchor:
                    neigh.add(u)
            if not neigh:
                continue
            L = {anchor}
            R = set(neigh)
            changed = True
            while changed:
                changed = False
                if L:
                    nr = set(adj[list(L)[0]])
                    for u in list(L)[1:]:
                        nr &= adj[u]
                    nr &= R
                    if nr != R:
                        R = nr
                        changed = True
                if R:
                    nl = set(adj[list(R)[0]])
                    for v in list(R)[1:]:
                        nl &= adj[v]
                    nl &= L
                    if nl != L:
                        L = nl
                        changed = True
            if not L or not R:
                continue
            # expand within remaining
            expanded = True
            while expanded:
                expanded = False
                for u in rem_cands:
                    if u not in L and u not in R:
                        if all((u, v) in rem_set for v in R):
                            L.add(u)
                            expanded = True
                for v in rem_cands:
                    if v not in L and v not in R:
                        if all((u, v) in rem_set for u in L):
                            R.add(v)
                            expanded = True
            if not L or not R:
                continue
            cnt = sum(1 for u in L for v in R if (u, v) in rem_set)
            if cnt > best_cnt:
                best_cnt = cnt
                best_L = sorted(L)
                best_R = sorted(R)
        if best_L is None or best_R is None:
            u, v = next(iter(remaining))
            bic.append([[u], [v]])
            remaining.discard((u, v) if u < v else (v, u))
            continue
        for u in best_L:
            for v in best_R:
                remaining.discard((u, v) if u < v else (v, u))
        bic.append([best_L, best_R])
    for e in remaining:
        bic.append([[e[0]], [e[1]]])
    return bic


def _polish_merge(bicliques, edges_set, time_limit=6.0):
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        nb = len(bicliques)
        for i in range(nb):
            if time.time() - start > time_limit:
                break
            li, ri = bicliques[i][0], bicliques[i][1]
            for j in range(i + 1, nb):
                if time.time() - start > time_limit:
                    break
                lj, rj = bicliques[j][0], bicliques[j][1]
                nL = set(li) | set(lj)
                nR = set(ri) | set(rj)
                if nL & nR:
                    continue
                ok = True
                for u in nL:
                    for v in nR:
                        if (u, v) if u < v else (v, u) not in edges_set:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                ci = _edge_set(li, ri, edges_set)
                cj = _edge_set(lj, rj, edges_set)
                cn = _edge_set(sorted(nL), sorted(nR), edges_set)
                if cn == ci | cj:
                    bicliques[i] = [sorted(nL), sorted(nR)]
                    del bicliques[j]
                    improved = True
                    break
            if improved:
                break
    return bicliques


def _polish_split_merge(bicliques, edges_set, time_limit=8.0):
    """Try splitting one biclique's side into two parts so that each part can merge with a different partner,
    reducing the total count. Specifically: for a biclique B=(L,R), try splitting L into L1,L2 and check if
    (L1,R) merges with A1 and (L2,R) merges with A2 where A1,A2 are other bicliques, net reducing count."""
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        nb = len(bicliques)
        for i in range(nb):
            if time.time() - start > time_limit:
                break
            li, ri = bicliques[i][0], bicliques[i][1]
            if len(li) < 2:
                continue
            # Try all 2-way splits of L
            L = list(li)
            for a in range(len(L)):
                if time.time() - start > time_limit:
                    break
                for b in range(a + 1, len(L)):
                    if time.time() - start > time_limit:
                        break
                    L1 = [x for k, x in enumerate(L) if k == a or (k == b)]
                    # This is getting complex; do a simpler split: pick one vertex to move
                    pass
            # Simpler: try moving one vertex from L to another biclique's L or R
            for u in li:
                if time.time() - start > time_limit:
                    break
                for j in range(nb):
                    if i == j:
                        continue
                    lj, rj = bicliques[j][0], bicliques[j][1]
                    # Move u to j's left
                    if u not in lj:
                        test = [list(x) for x in bicliques]
                        test[i][0] = [x for x in li if x != u]
                        test[j][0] = lj + [u]
                        if not test[i][0] or not test[i][1]:
                            test.pop(i)
                            if j > i:
                                j -= 1
                        if _validate(test, edges_set) and len(test) < nb:
                            bicliques = test
                            improved = True
                            break
                    # Move u to j's right
                    if u not in rj:
                        test = [list(x) for x in bicliques]
                        test[i][0] = [x for x in li if x != u]
                        test[j][1] = rj + [u]
                        if not test[i][0] or not test[i][1]:
                            test.pop(i)
                            if j > i:
                                j -= 1
                        if _validate(test, edges_set) and len(test) < nb:
                            bicliques = test
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return bicliques


def _polish_relocate(bicliques, edges_set, time_limit=6.0):
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        nb = len(bicliques)
        for i in range(nb):
            if time.time() - start > time_limit:
                break
            li, ri = bicliques[i][0], bicliques[i][1]
            for j in range(nb):
                if i == j or time.time() - start > time_limit:
                    continue
                lj, rj = bicliques[j][0], bicliques[j][1]
                for side_src in ['L', 'R']:
                    if improved:
                        break
                    src = li if side_src == 'L' else ri
                    for u in src:
                        if improved:
                            break
                        for side_dst in ['L', 'R']:
                            dst = lj if side_dst == 'L' else rj
                            if u in dst:
                                continue
                            if not all(u in adj_global[v] for v in dst):
                                continue
                            test = [list(x) for x in bicliques]
                            if side_src == 'L':
                                test[i][0] = [x for x in li if x != u]
                            else:
                                test[i][1] = [x for x in ri if x != u]
                            if side_dst == 'L':
                                test[j][0] = lj + [u]
                            else:
                                test[j][1] = rj + [u]
                            if not test[i][0] or not test[i][1]:
                                test.pop(i)
                                if j > i:
                                    j -= 1
                            if _validate(test, edges_set) and len(test) < nb:
                                bicliques = test
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
    return bicliques


def construct():
    global adj_global
    start = time.time()
    deadline = start + 85
    n = P['n']
    adj = _build_adjacency()
    adj_global = adj
    edges_set = set((u, v) if u < v else (v, u) for u, v in P['edges'])

    def star_partition():
        res = []
        for u in range(n):
            nb = sorted(adj[u])
            if nb:
                res.append([nb, [u]])
        return res

    fallback = star_partition()
    best = fallback
    best_cnt = len(fallback)

    candidates = []
    if time.time() < deadline - 30:
        candidates.extend(_find_maximal_bicliques(adj, edges_set, n))
    if time.time() < deadline - 25:
        pb = _find_pair_bicliques(adj, edges_set, n, max_pairs=600)
        ex = set()
        for L, R in candidates:
            ex.add((frozenset(L), frozenset(R)))
            ex.add((frozenset(R), frozenset(L)))
        for L, R in pb:
            k1 = (frozenset(L), frozenset(R))
            k2 = (frozenset(R), frozenset(L))
            if k1 not in ex and k2 not in ex:
                ex.add(k1)
                ex.add(k2)
                candidates.append((L, R))
    if time.time() < deadline - 20:
        gb = _find_greedy_bicliques(adj, edges_set, n, time_limit=3.0)
        ex = set()
        for L, R in candidates:
            ex.add((frozenset(L), frozenset(R)))
            ex.add((frozenset(R), frozenset(L)))
        for L, R in gb:
            k1 = (frozenset(L), frozenset(R))
            k2 = (frozenset(R), frozenset(L))
            if k1 not in ex and k2 not in ex:
                ex.add(k1)
                ex.add(k2)
                candidates.append((L, R))

    if candidates and time.time() < deadline - 15:
        sol = _solve_exact_cover(candidates, edges_set, time_limit=min(15.0, deadline - 15 - time.time()))
        if sol is not None and _validate(sol, edges_set) and len(sol) < best_cnt:
            best = sol
            best_cnt = len(sol)

    if time.time() < deadline - 10:
        gs = _greedy_partition(adj, edges_set, n, time_limit=min(8.0, deadline - 10 - time.time()))
        if _validate(gs, edges_set) and len(gs) < best_cnt:
            best = gs
            best_cnt = len(gs)

    if time.time() < deadline - 8:
        best = _polish_merge([list(x) for x in best], edges_set, time_limit=min(5.0, deadline - 8 - time.time()))
        if _validate(best, edges_set) and len(best) < best_cnt:
            best_cnt = len(best)
    if time.time() < deadline - 6:
        best = _polish_split_merge([list(x) for x in best], edges_set, time_limit=min(4.0, deadline - 6 - time.time()))
        if _validate(best, edges_set) and len(best) < best_cnt:
            best_cnt = len(best)
    if time.time() < deadline - 4:
        best = _polish_relocate([list(x) for x in best], edges_set, time_limit=min(3.0, deadline - 4 - time.time()))
        if _validate(best, edges_set) and len(best) < best_cnt:
            best_cnt = len(best)
    # Rebuild: try removing one biclique and re-covering its edges with other candidates
    if time.time() < deadline - 3 and candidates:
        for rm in range(len(best)):
            if time.time() > deadline - 3:
                break
            rm_edges = _edge_set(best[rm][0], best[rm][1], edges_set)
            remaining_edges = edges_set - rm_edges
            # Try to cover remaining_edges with fewer than len(best)-1 bicliques from candidates
            sub_cands = []
            for L, R in candidates:
                e = _edge_set(L, R, edges_set)
                # Only keep candidates that don't use edges from rm_edges (to avoid overlap)
                if e and not (e & rm_edges) and e <= remaining_edges:
                    sub_cands.append((L, R))
            if not sub_cands:
                continue
            sol = _solve_exact_cover(sub_cands, remaining_edges, time_limit=min(3.0, deadline - 3 - time.time()))
            if sol is not None and _validate(sol, edges_set) and len(sol) < best_cnt:
                best = sol
                best_cnt = len(sol)
                break
    if time.time() < deadline - 2:
        best = _polish_merge([list(x) for x in best], edges_set, time_limit=min(2.0, deadline - 2 - time.time()))
        if _validate(best, edges_set) and len(best) < best_cnt:
            best_cnt = len(best)

    if not _validate(best, edges_set):
        best = fallback

    result = []
    for bic in best:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])
    return result

# EVOLVE-BLOCK-END

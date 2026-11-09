import time
import random
import itertools

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before the 90s kill

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        a, b = (u, v) if u < v else (v, u)
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add((a, b))

    def mk(u, v):
        return (u, v) if u < v else (v, u)

    def expand(L, R, remaining):
        L, R = set(L), set(R)
        changed = True
        while changed:
            changed = False
            newL = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for r in R:
                    if mk(w, r) not in remaining:
                        ok = False
                        break
                if ok:
                    newL.add(w)
            if newL:
                L |= newL
                changed = True
            newR = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for l in L:
                    if mk(w, l) not in remaining:
                        ok = False
                        break
                if ok:
                    newR.add(w)
            if newR:
                R |= newR
                changed = True
        return L, R

    def valid_biclique(L, R, remaining):
        for l in L:
            for r in R:
                if l == r:
                    return False
                if mk(l, r) not in remaining:
                    return False
        return True

    def find_biclique(remaining, adj, n):
        best = None
        best_size = 0
        # 1) edge-seed expansion
        for (u, v) in remaining:
            L, R = expand({u}, {v}, remaining)
            if not valid_biclique(L, R, remaining):
                continue
            s = len(L) * len(R)
            if s > best_size:
                best_size = s
                best = (L, R)
        # 2) common-neighbor seeding
        for u in range(n):
            nbrs = set()
            for v in adj[u]:
                if mk(u, v) in remaining:
                    nbrs.add(v)
            if not nbrs:
                continue
            L, R = expand({u}, nbrs, remaining)
            if not valid_biclique(L, R, remaining):
                continue
            s = len(L) * len(R)
            if s > best_size:
                best_size = s
                best = (L, R)
        return best

    def greedy(remaining, adj, n, deadline):
        bicliques = []
        while remaining:
            if time.time() > deadline:
                break
            b = find_biclique(remaining, adj, n)
            if b is None:
                u, v = next(iter(remaining))
                bicliques.append([u, v])
                remaining.discard(mk(u, v))
                continue
            L, R = b
            if not valid_biclique(L, R, remaining):
                u, v = next(iter(remaining))
                bicliques.append([u, v])
                remaining.discard(mk(u, v))
                continue
            bicliques.append([list(L), list(R)])
            for u in L:
                for v in R:
                    remaining.discard(mk(u, v))
        return bicliques

    def try_move(bicliques, edge_set, n, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    return
                L, R = bicliques[i]
                for side in (0, 1):
                    if side == 0:
                        source, target = L, R
                    else:
                        source, target = R, L
                    for v in source:
                        # candidate new host side = target ∪ {v}
                        host = list(target) + [v]
                        # the other side becomes source without v
                        other = [x for x in source if x != v]
                        # new biclique = [other, host]
                        newL = other
                        newR = host
                        if not newL or not newR:
                            continue
                        # every cross pair must be a host edge
                        ok = True
                        for a in newL:
                            for b in newR:
                                if a == b or mk(a, b) not in edge_set:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        # the move must not change the covered edge set
                        old_edges = set()
                        for a in L:
                            for b in R:
                                old_edges.add(mk(a, b))
                        new_edges = set()
                        for a in newL:
                            for b in newR:
                                new_edges.add(mk(a, b))
                        if new_edges != old_edges:
                            continue
                        # apply
                        if side == 0:
                            bicliques[i] = [newL, newR]
                        else:
                            bicliques[i] = [newR, newL]
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break

    def try_merge(bicliques, edge_set, n, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    return
                for j in range(i + 1, len(bicliques)):
                    if time.time() > deadline:
                        return
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    # try the four natural merges
                    candidates = [
                        (set(L1) | set(L2), set(R1) | set(R2)),
                        (set(L1) | set(R2), set(R1) | set(L2)),
                        (set(L1) | set(L2), set(R1) | set(R2)),
                    ]
                    # de-duplicate
                    seen = set()
                    for Lnew, Rnew in candidates:
                        key = (frozenset(Lnew), frozenset(Rnew))
                        if key in seen:
                            continue
                        seen.add(key)
                        if not Lnew or not Rnew:
                            continue
                        # cross pairs must be host edges
                        ok = True
                        for a in Lnew:
                            for b in Rnew:
                                if a == b or mk(a, b) not in edge_set:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        merged_edges = set()
                        for a in Lnew:
                            for b in Rnew:
                                merged_edges.add(mk(a, b))
                        orig_edges = set()
                        for a in L1:
                            for b in R1:
                                orig_edges.add(mk(a, b))
                        for a in L2:
                            for b in R2:
                                orig_edges.add(mk(a, b))
                        if merged_edges == orig_edges:
                            bicliques[i] = [sorted(Lnew), sorted(Rnew)]
                            bicliques.pop(j)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    def reseed(remaining, adj, n, deadline):
        bicliques = []
        rem = set(remaining)
        while rem:
            if time.time() > deadline:
                break
            # pick a random edge as seed, expand
            e = next(iter(rem))
            L, R = expand({e[0]}, {e[1]}, rem)
            if not valid_biclique(L, R, rem) or not L or not R:
                u, v = next(iter(rem))
                bicliques.append([u, v])
                rem.discard(mk(u, v))
                continue
            bicliques.append([list(L), list(R)])
            for u in L:
                for v in R:
                    rem.discard(mk(u, v))
        return bicliques

    # initial greedy partition
    remaining = set(edge_set)
    bicliques = greedy(remaining, adj, n, deadline - 8)
    if remaining:
        for e in remaining:
            u, v = e
            bicliques.append([u, v])

    # local refinement: move then merge, a few passes
    for _ in range(3):
        if time.time() > deadline - 4:
            break
        try_move(bicliques, edge_set, n, deadline - 4)
        try_merge(bicliques, edge_set, n, deadline - 4)

    # occasional reseed to escape local minima (only if clearly beneficial)
    if time.time() < deadline - 6 and len(bicliques) > 3:
        for _ in range(2):
            if time.time() > deadline - 6:
                break
            alt = reseed(set(edge_set), adj, n, deadline - 6)
            if len(alt) < len(bicliques):
                bicliques = alt
                try_move(bicliques, edge_set, n, deadline - 4)
                try_merge(bicliques, edge_set, n, deadline - 4)

    # final safety: ensure full coverage by appending any missed edges
    covered = set()
    for L, R in bicliques:
        for a in L:
            for b in R:
                covered.add(mk(a, b))
    for e in edge_set:
        if e not in covered:
            bicliques.append([e[0], e[1]])

    # convert to exact Python types
    result = []
    for L, R in bicliques:
        Lc = [int(x) for x in L]
        Rc = [int(x) for x in R]
        if Lc and Rc:
            result.append([Lc, Rc])
    return result
# EVOLVE-BLOCK-END

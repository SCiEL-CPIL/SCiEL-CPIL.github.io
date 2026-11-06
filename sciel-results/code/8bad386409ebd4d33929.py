import time

N = 31
E = 91
DEADLINE = 110.0

def _pg25_polarity():
    F = 5
    pts = []
    seen = set()
    for a in range(F):
        for b in range(F):
            for c in range(F):
                if a == b == c == 0:
                    continue
                if a != 0:
                    rep = (1, (b * pow(a, -1, F)) % F, (c * pow(a, -1, F)) % F)
                elif b != 0:
                    rep = (0, 1, (c * pow(b, -1, F)) % F)
                else:
                    rep = (0, 0, 1)
                if rep not in seen:
                    seen.add(rep)
                    pts.append(rep)
    assert len(pts) == 31

    def dot(p, q):
        return (p[0]*q[0] - p[1]*q[1] - p[2]*q[2]) % F

    adj = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if dot(pts[i], pts[j]) == 0:
                adj[i][j] = adj[j][i] = 1
    return adj

def _edge_count(adj):
    return sum(adj[i][j] for i in range(N) for j in range(i+1, N))

def _count_c4(adj):
    total = 0
    for u in range(N):
        for v in range(u+1, N):
            c = 0
            for w in range(N):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    c += 1
            total += c*(c-1)//2
    return total

def _build_initial():
    adj = _pg25_polarity()
    ec = _edge_count(adj)
    assert ec == 90, f"polarity graph has {ec} edges, expected 90"
    best = None
    for u in range(N):
        for v in range(u+1, N):
            if adj[u][v]:
                continue
            adj[u][v] = adj[v][u] = 1
            cnt = _count_c4(adj)
            adj[u][v] = adj[v][u] = 0
            if best is None or cnt < best[0]:
                best = (cnt, u, v)
    assert best is not None
    adj[best[1]][best[2]] = adj[best[2]][best[1]] = 1
    assert _edge_count(adj) == E
    return adj

def construct():
    start = time.time()
    adj = _build_initial()
    best_c4 = _count_c4(adj)

    def time_left():
        return DEADLINE - (time.time() - start)

    # Incremental C4 via common-neighbor counts.
    # common[u][v] = number of w with adj[u][w] and adj[v][w].
    # C4 = sum_{u<v} C(common[u][v], 2).
    def build_common(adj):
        com = [[0]*N for _ in range(N)]
        for w in range(N):
            for u in range(N):
                if not adj[u][w]:
                    continue
                for v in range(N):
                    if v != w and adj[v][w]:
                        com[u][v] += 1
        return com

    def c4_from_common(com):
        total = 0
        for u in range(N):
            for v in range(u+1, N):
                c = com[u][v]
                total += c*(c-1)//2
        return total

    com = build_common(adj)
    assert c4_from_common(com) == best_c4

    improved = True
    while improved and time_left() > 4.0:
        improved = False
        cur_c4 = best_c4
        for u in range(N):
            for v in range(u+1, N):
                if not adj[u][v]:
                    continue
                # Try removing {u,v} and adding {a,b}
                # Remove {u,v}: for each w in N(u)\{v}, common[u][w]-=1, common[v][w]-=1
                # and common counts involving u,v with each other unaffected.
                # We'll simulate by updating com in place and reverting.
                # Collect neighbors
                Nu = [w for w in range(N) if w != v and adj[u][w]]
                Nv = [w for w in range(N) if w != u and adj[v][w]]
                # remove edge u-v
                for w in Nu:
                    com[u][w] -= 1
                    com[w][u] -= 1
                for w in Nv:
                    com[v][w] -= 1
                    com[w][v] -= 1
                com[u][v] -= 0  # common(u,v) unaffected by removing their own edge
                adj[u][v] = adj[v][u] = 0
                base_c4 = cur_c4
                # recompute base_c4 after removal quickly: subtract old contributions
                # Easier: recompute from com for the changed pairs only.
                # Changed common pairs: (u,w) for w in Nu, (v,w) for w in Nv.
                # We'll just track delta.
                delta = 0
                for w in Nu:
                    c_old = com[u][w] + 1
                    c_new = com[u][w]
                    delta += c_new*(c_new-1)//2 - c_old*(c_old-1)//2
                for w in Nv:
                    c_old = com[v][w] + 1
                    c_new = com[v][w]
                    delta += c_new*(c_new-1)//2 - c_old*(c_old-1)//2
                c4_after_remove = cur_c4 + delta
                # Now try adding each non-edge {a,b}
                best_add = None
                for a in range(N):
                    for b in range(a+1, N):
                        if adj[a][b]:
                            continue
                        # Adding {a,b}: common[a][b] unchanged (a,b not neighbors of each other)
                        # but common[a][w] +=1 for w in N(b)\{a}, common[b][w] +=1 for w in N(a)\{b}
                        Na = [w for w in range(N) if w != b and adj[a][w]]
                        Nb = [w for w in range(N) if w != a and adj[b][w]]
                        d2 = 0
                        for w in Na:
                            c_old = com[a][w]
                            c_new = c_old + 1
                            d2 += c_new*(c_new-1)//2 - c_old*(c_old-1)//2
                        for w in Nb:
                            c_old = com[b][w]
                            c_new = c_old + 1
                            d2 += c_new*(c_new-1)//2 - c_old*(c_old-1)//2
                        cand = c4_after_remove + d2
                        if cand < base_c4:
                            if best_add is None or cand < best_add[0]:
                                best_add = (cand, a, b, Na, Nb)
                # Revert removal
                adj[u][v] = adj[v][u] = 1
                for w in Nu:
                    com[u][w] += 1
                    com[w][u] += 1
                for w in Nv:
                    com[v][w] += 1
                    com[w][v] += 1
                if best_add is not None:
                    cand, a, b, Na, Nb = best_add
                    # Apply: remove {u,v}, add {a,b}
                    adj[u][v] = adj[v][u] = 0
                    for w in Nu:
                        com[u][w] -= 1
                        com[w][u] -= 1
                    for w in Nv:
                        com[v][w] -= 1
                        com[w][v] -= 1
                    adj[a][b] = adj[b][a] = 1
                    for w in Na:
                        com[a][w] += 1
                        com[w][a] += 1
                    for w in Nb:
                        com[b][w] += 1
                        com[w][b] += 1
                    best_c4 = cand
                    cur_c4 = cand
                    improved = True
                    break
            if improved:
                break
            if time_left() < 4.0:
                break

    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                edges.append([i, j])
    assert len(edges) == E, f"got {len(edges)} edges"
    return edges

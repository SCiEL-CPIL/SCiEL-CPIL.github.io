import time
import itertools

def construct():
    deadline = time.time() + 140

    # Q3 edges: 8 vertices (0-7), edges between Hamming distance 1
    Q3E = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                Q3E.append((i, j))
    q3a = [0] * 8
    for u, v in Q3E:
        q3a[u] |= 1 << v
        q3a[v] |= 1 << u

    def has_q3(adj, verts):
        """Check if induced subgraph on verts contains Q3 as a subgraph."""
        v = list(verts)
        # Quick filter: each vertex needs >= 3 neighbors in the set
        for i in range(8):
            d = 0
            for j in range(8):
                if i != j and (adj[v[i]] >> v[j]) & 1:
                    d += 1
            if d < 3:
                return False
        # Build local adjacency
        la = [0] * 8
        for i in range(8):
            for j in range(i + 1, 8):
                if (adj[v[i]] >> v[j]) & 1:
                    la[i] |= 1 << j
                    la[j] |= 1 << i
        # Backtracking to find Q3 embedding
        asg = [-1] * 8
        used = [False] * 8

        def bt(q):
            if q == 8:
                return True
            for l in range(8):
                if used[l]:
                    continue
                ok = True
                for p in range(q):
                    if asg[p] >= 0 and (q3a[q] >> p) & 1:
                        if not (la[l] >> asg[p]) & 1:
                            ok = False
                            break
                if ok:
                    asg[q] = l
                    used[l] = True
                    if bt(q + 1):
                        return True
                    used[l] = False
                    asg[q] = -1
            return False

        return bt(0)

    def q3free(adj):
        for c in itertools.combinations(range(16), 8):
            if has_q3(adj, c):
                return False
        return True

    def ec(adj):
        return sum(bin(adj[v]).count('1') for v in range(16)) // 2

    def elist(adj):
        e = []
        for u in range(16):
            for v in range(u + 1, 16):
                if (adj[u] >> v) & 1:
                    e.append([u, v])
        return e

    # Start: two disjoint K8's (56 edges, Q3-free since Q3 is connected and not a clique)
    adj = [0] * 16
    for u in range(8):
        for v in range(u + 1, 8):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    for u in range(8, 16):
        for v in range(u + 1, 16):
            adj[u] |= 1 << v
            adj[v] |= 1 << u

    best_ec = ec(adj)
    best_adj = adj[:]

    # Try to add cross edges between the two cliques
    while time.time() < deadline - 5:
        changed = False
        for u in range(8):
            for v in range(8, 16):
                if (adj[u] >> v) & 1:
                    continue
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                if q3free(adj):
                    changed = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        if not changed:
            break

    c = ec(adj)
    if c > best_ec:
        best_ec = c
        best_adj = adj[:]

    return elist(best_adj)

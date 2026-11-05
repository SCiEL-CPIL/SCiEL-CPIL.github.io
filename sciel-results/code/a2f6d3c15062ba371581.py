import time

def construct():
    deadline = time.time() + 145.0
    n = 18

    # ---- Build initial K4-free graph: blow-up of K6 with 6 groups of 3 ----
    # Complete between groups, no edges inside groups.
    # K4-free: any K4 needs 4 distinct groups (no intra-group edges), impossible.
    # Largest triangle-free induced subgraph = 6 (one vertex per group).
    def group_of(v):
        return v // 3

    adj = [0] * n  # bitsets
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if group_of(i) != group_of(j):
                edges.append([i, j])
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # ---- Exact evaluation: largest triangle-free induced subgraph ----
    def eval_max_tf(adj):
        best = 0
        for mask in range(1, 1 << n):
            m = mask
            tri = False
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                m ^= lb
                rest = m & adj[v]
                if rest & (rest >> 1):
                    # has an edge among rest -> triangle
                    tri = True
                    break
            if not tri:
                c = mask.bit_count()
                if c > best:
                    best = c
        return best

    current = eval_max_tf(adj)

    # ---- Cheap local polish: only add edges (safe: no K4, no new triangles) ----
    # A candidate edge (u,v) with u,v in different groups can be added iff it does not
    # create a triangle, i.e. (adj[u] & adj[v]) == 0.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if group_of(i) == group_of(j):
                    continue
                if adj[i] & (1 << j):
                    continue
                if adj[i] & adj[j]:
                    continue  # would form a triangle
                # add edge
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
                newc = eval_max_tf(adj)
                if newc < current:
                    current = newc
                    improved = True
                else:
                    # undo
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                    edges.pop()
        if current <= 5:
            break

    # ---- Deadline guard ----
    if time.time() > deadline - 3.0:
        return edges

    return edges

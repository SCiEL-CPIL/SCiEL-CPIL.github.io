def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    P = 101
    inv = [0] + [pow(x, -1, P) for x in range(1, 11)]

    # Enumerate all non-empty subsets
    subsets = []  # list of (residue, size, terms)
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
        subsets.append((s, len(terms), terms))

    n_subsets = len(subsets)

    # For each L, try to find a perfect bipartite matching
    # Left: residues 0..100, Right: subsets with size <= L
    # Edge: residue r to subset i if subsets[i][0] == r and subsets[i][1] <= L
    for L in range(1, 11):
        if time.time() - start_time > DEADLINE:
            break

        # Build adjacency: adj[r] = list of subset indices that produce residue r with size <= L
        adj = [[] for _ in range(P)]
        for i in range(n_subsets):
            res, sz, terms = subsets[i]
            if sz <= L:
                adj[res].append(i)

        # Hopcroft-Karp algorithm for bipartite matching
        INF = float('inf')
        pair_u = [-1] * P       # pair_u[r] = subset index matched to residue r
        pair_v = [-1] * n_subsets  # pair_v[j] = residue matched to subset j
        dist = [0] * P

        from collections import deque

        def bfs():
            q = deque()
            found = False
            for u in range(P):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = INF
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == INF:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = INF
            return False

        matching = 0
        while time.time() - start_time < DEADLINE:
            if not bfs():
                break
            for u in range(P):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1

        if matching == P:
            result = [subsets[pair_u[r]][2] for r in range(P)]
            return result

    # Fallback: greedy
    assignment = {}
    for L in range(1, 11):
        for res, sz, terms in subsets:
            if sz == L and res not in assignment:
                assignment[res] = terms
        if len(assignment) == P:
            return [assignment[r] for r in range(P)]

    return [assignment.get(r, [1]) for r in range(P)]

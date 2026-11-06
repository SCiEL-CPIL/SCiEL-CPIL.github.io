import random, time

def construct():
    deadline = time.time() + 120.0
    rng = random.Random(12345)
    best_n = 0
    best_edges = []

    def valid(n, edges):
        adj = [0] * n
        for a, b in edges:
            adj[a] |= 1 << b
            adj[b] |= 1 << a
        for i in range(n):
            if adj[i] & (1 << i):
                return False
        # K4 check
        for i in range(n):
            ni = adj[i]
            for j in range(i + 1, n):
                if not (ni >> j) & 1:
                    continue
                nij = ni & adj[j]
                for k in range(j + 1, n):
                    if not (nij >> k) & 1:
                        continue
                    if (nij & adj[k]) >> (k + 1):
                        return False
        # independent 6-set
        for i in range(n):
            ni = adj[i]
            for j in range(i + 1, n):
                if (ni >> j) & 1:
                    continue
                nij = ~ni & ~adj[j]
                for k in range(j + 1, n):
                    if not (nij >> k) & 1:
                        continue
                    nk = nij & ~adj[k]
                    for l in range(k + 1, n):
                        if not (nk >> l) & 1:
                            continue
                        if (nk & ~adj[l]) & ((1 << n) - 1) & (1 << (l + 1)):
                            return False
        return True

    def extend(n, edges):
        adj = [0] * n
        for a, b in edges:
            adj[a] |= 1 << b
            adj[b] |= 1 << a
        for _ in range(4000):
            if time.time() > deadline:
                return n, edges
            v = n
            # candidate neighbors: must avoid K4 and indep-6
            cand = []
            for x in range(v):
                # K4: no two neighbors of x are adjacent to each other... i.e. N(x) must be independent
                ok = True
                nx = adj[x]
                for y in range(x):
                    if (nx >> y) & 1 and (nx & adj[y]):
                        ok = False
                        break
                if not ok:
                    continue
                # indep6: N(x) must not contain a 5-clique of non-neighbors... i.e. no 5-set in N(x) pairwise non-adjacent
                # check: among neighbors of x, is there an independent 5-set?
                nn = nx
                found5 = False
                # quick: count independent set of size 5 within nn
                for y in range(v):
                    if not (nn >> y) & 1:
                        continue
                    ny = nn & ~adj[y]
                    for z in range(y + 1, v):
                        if not (ny >> z) & 1:
                            continue
                        nyz = ny & ~adj[z]
                        for w in range(z + 1, v):
                            if not (nyz >> w) & 1:
                                continue
                            nyzw = nyz & ~adj[w]
                            for t in range(w + 1, v):
                                if not (nyzw >> t) & 1:
                                    continue
                                if (nyzw & ~adj[t]) & ((1 << v) - 1) & (1 << (t + 1)):
                                    found5 = True
                                    break
                            if found5:
                                break
                        if found5:
                            break
                    if found5:
                        break
                if found5:
                    continue
                cand.append(x)
            if not cand:
                break
            x = cand[rng.randrange(len(cand))]
            adj.append(adj[x] | (1 << x))
            adj[x] |= 1 << v
            edges.append([x, v])
            n += 1
        return n, edges

    # multiple seeds with random restarts
    for attempt in range(400):
        if time.time() > deadline - 3.0:
            break
        # seed with a small random K4-free, indep6-free graph
        n0 = rng.randint(5, 10)
        edges = []
        for i in range(n0):
            for j in range(i + 1, n0):
                if rng.random() < 0.4:
                    edges.append([i, j])
        if not valid(n0, edges):
            continue
        n, edges = extend(n0, edges)
        if valid(n, edges) and n > best_n:
            best_n = n
            best_edges = edges[:]
        # also try to push a bit more with different randomization
        if time.time() > deadline - 3.0:
            break

    if best_n == 0:
        # fallback: known Paley 17
        squares = {x * x % 17 for x in range(1, 17)}
        best_n = 17
        best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]

    return {'n': best_n, 'edges': best_edges}

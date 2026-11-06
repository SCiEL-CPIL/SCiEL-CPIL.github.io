import time

N = 40

def construct():
    start_time = time.time()
    deadline = 55.0
    
    base_edges = [(i, (i+1) % N) for i in range(N)]
    chords_init = [(0, j) for j in range(2, 21)]
    
    def dist(a, b):
        return (b - a) % N
    
    def get_all_cycle_lengths(cl):
        lengths = {N}
        n = len(cl)
        for i in range(n):
            u, v = cl[i]
            d1 = dist(u, v)
            d2 = dist(v, u)
            if 1 <= d1 <= N-1:
                lengths.add(d1 + 1)
            if 1 <= d2 <= N-1:
                lengths.add(d2 + 1)
        for i in range(n):
            for j in range(i+1, n):
                u1, v1 = cl[i]
                u2, v2 = cl[j]
                l1 = 2 + dist(v1, u2) + dist(v2, u1)
                if 3 <= l1 <= N:
                    lengths.add(l1)
                l2 = 2 + dist(v1, v2) + dist(u2, u1)
                if 3 <= l2 <= N:
                    lengths.add(l2)
        if n >= 3:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        u1, v1 = cl[i]
                        u2, v2 = cl[j]
                        u3, v3 = cl[k]
                        l1 = 3 + dist(v1, u2) + dist(v2, u3) + dist(v3, u1)
                        if 3 <= l1 <= N:
                            lengths.add(l1)
                        l2 = 3 + dist(v1, v2) + dist(u2, u3) + dist(v3, u1)
                        if 3 <= l2 <= N:
                            lengths.add(l2)
                        l3 = 3 + dist(v1, v2) + dist(u2, v3) + dist(u3, u1)
                        if 3 <= l3 <= N:
                            lengths.add(l3)
        return lengths
    
    def is_pancyclic(cl):
        lengths = get_all_cycle_lengths(cl)
        return all(L in lengths for L in range(3, N+1))
    
    current = list(chords_init)
    changed = True
    while changed and (time.time() - start_time) < deadline - 15:
        changed = False
        for i in range(len(current)):
            test = current[:i] + current[i+1:]
            if is_pancyclic(test):
                current.pop(i)
                changed = True
                break
    
    edges = base_edges + current
    edge_set = set((min(e), max(e)) for e in edges)
    
    adj = {i: set() for i in range(N)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_cycle(L):
        if L == N:
            return list(range(N))
        for u, v in current:
            d1 = dist(u, v)
            if d1 + 1 == L:
                cyc = []
                cur = u
                for _ in range(d1):
                    cyc.append(cur)
                    cur = (cur + 1) % N
                cyc.append(v)
                return cyc
            d2 = dist(v, u)
            if d2 + 1 == L:
                cyc = []
                cur = u
                for _ in range(d2):
                    cyc.append(cur)
                    cur = (cur - 1) % N
                cyc.append(v)
                return cyc
        for i in range(len(current)):
            for j in range(i+1, len(current)):
                u1, v1 = current[i]
                u2, v2 = current[j]
                l1 = 2 + dist(v1, u2) + dist(v2, u1)
                if l1 == L:
                    cyc = [u1, v1]
                    cur = v1
                    for _ in range(dist(v1, u2)):
                        cur = (cur + 1) % N
                        cyc.append(cur)
                    cyc.append(v2)
                    cur = v2
                    for _ in range(dist(v2, u1)):
                        cur = (cur + 1) % N
                        cyc.append(cur)
                    return cyc
                l2 = 2 + dist(v1, v2) + dist(u2, u1)
                if l2 == L:
                    cyc = [u1, v1]
                    cur = v1
                    for _ in range(dist(v1, v2)):
                        cur = (cur + 1) % N
                        cyc.append(cur)
                    cyc.append(u2)
                    cur = u2
                    for _ in range(dist(u2, u1)):
                        cur = (cur + 1) % N
                        cyc.append(cur)
                    return cyc
        if len(current) >= 3:
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    for k in range(j+1, len(current)):
                        u1, v1 = current[i]
                        u2, v2 = current[j]
                        u3, v3 = current[k]
                        l1 = 3 + dist(v1, u2) + dist(v2, u3) + dist(v3, u1)
                        if l1 == L:
                            cyc = [u1, v1]
                            cur = v1
                            for _ in range(dist(v1, u2)):
                                cur = (cur + 1) % N
                                cyc.append(cur)
                            cyc.append(v2)
                            cur = v2
                            for _ in range(dist(v2, u3)):
                                cur = (cur + 1) % N
                                cyc.append(cur)
                            cyc.append(v3)
                            cur = v3
                            for _ in range(dist(v3, u1)):
                                cur = (cur + 1) % N
                                cyc.append(cur)
                            return cyc
                        l2 = 3 + dist(v1, v2) + dist(u2, u3) + dist(v3, u1)
                        if l2 == L:
                            cyc = [u1, v1]
                            cur = v1
                            for _ in range(dist(v1, v2)):
                                cur = (cur + 1) % N
                                cyc.append(cur)
                            cyc.append(u2)
                            cur = u2
                            for _ in range(dist(u2, u3)):
                                cur = (cur + 1) % N
                                cyc.append(cur)
                            cyc.append(v3)
                            cur = v3
                            for _ in range(dist(v3, u1)):
                                cur = (cur + 1) % N
                                cyc.append(cur)
                            return cyc
        def dfs(cur, vis, path, rem):
            if rem == 0:
                return path + [path[0]] if path[0] in adj[cur] else None
            if rem < 1:
                return None
            for nb in sorted(adj[cur]):
                if nb not in vis:
                    vis.add(nb)
                    path.append(nb)
                    r = dfs(nb, vis, path, rem-1)
                    if r:
                        return r
                    path.pop()
                    vis.remove(nb)
            return None
        for s in range(N):
            r = dfs(s, {s}, [s], L-1)
            if r:
                return r
        return None
    
    def fallback():
        e = base_edges + [(0, j) for j in range(2, 21)]
        cs = []
        for k in range(3, 22):
            cs.append(list(range(k)))
        for L2 in range(22, 40):
            j = 41 - L2
            cs.append([0] + list(range(j, N)))
        cs.append(list(range(N)))
        return {'edges': e, 'cycles': cs}
    
    cycles = []
    for L in range(3, N+1):
        if time.time() - start_time > deadline - 2:
            return fallback()
        c = find_cycle(L)
        if c is None:
            return fallback()
        cycles.append(c)
    
    return {'edges': edges, 'cycles': cycles}

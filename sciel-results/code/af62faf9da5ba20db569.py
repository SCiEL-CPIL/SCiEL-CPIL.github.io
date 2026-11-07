import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    t0 = time.time()
    deadline = t0 + 85
    n = P['n']
    
    base = set()
    for e in P['edges']:
        base.add((e[0], e[1]))
        base.add((e[1], e[0]))
    
    # Build base adjacency
    base_adj = [[] for _ in range(n)]
    for u, v in base:
        if u < v:
            base_adj[u].append(v)
            base_adj[v].append(u)
    
    # Candidate edges: same parity, dist >= 4
    cands = []
    for u in range(n):
        for v in range(u + 1, n):
            d = v - u
            if d >= 4 and d % 2 == 0:
                cands.append((u, v))
    
    def get_adj(added):
        adj = [list(x) for x in base_adj]
        for u, v in added:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def is_valid(added):
        if time.time() > deadline:
            return False
        adj = get_adj(added)
        # Check triangle-free
        for u in range(n):
            for i, v in enumerate(adj[u]):
                for w in adj[u][i+1:]:
                    if w in adj[v]:
                        return False
        # Check connected and diameter
        max_d = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            for u in queue:
                for v in adj[u]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if -1 in dist:
                return False
            max_d = max(max_d, max(dist))
        return max_d <= 4
    
    best = None
    best_len = float('inf')
    
    import random
    random.seed(42)
    
    for attempt in range(10):
        if time.time() > deadline - 3:
            break
        order = cands[:]
        random.shuffle(order)
        
        added = []
        while not is_valid(added) and time.time() < deadline - 3:
            best_e, best_score = None, float('inf')
            for e in order:
                if e in added:
                    continue
                trial = added + [e]
                # Quick check: does this create a triangle?
                # e = (u,v), check if any w adjacent to both u and v in trial
                u, v = e
                adj_u = set(base_adj[u])
                adj_v = set(base_adj[v])
                for a, b in trial:
                    if a == u:
                        adj_u.add(b)
                    elif b == u:
                        adj_u.add(a)
                    if a == v:
                        adj_v.add(b)
                    elif b == v:
                        adj_v.add(a)
                if adj_u & adj_v:
                    continue  # Would create triangle
                # Compute diameter
                adj = get_adj(trial)
                max_d = 0
                for s in range(n):
                    dist = [-1] * n
                    dist[s] = 0
                    q = [s]
                    for x in q:
                        for y in adj[x]:
                            if dist[y] < 0:
                                dist[y] = dist[x] + 1
                                q.append(y)
                    max_d = max(max_d, max(dist))
                if max_d < best_score:
                    best_score = max_d
                    best_e = e
            if best_e is None:
                break
            added.append(best_e)
        
        if is_valid(added) and len(added) < best_len:
            best_len = len(added)
            best = added[:]
            # Local search
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(len(added)):
                    if time.time() > deadline - 2:
                        break
                    trial = added[:i] + added[i+1:]
                    if is_valid(trial):
                        added = trial
                        improved = True
                        best_len = len(added)
                        best = added[:]
                        break
    
    if best is None:
        # Fallback: need a valid solution
        # Try all same-parity edges distance 4 only (triangle-free? (0,4),(4,8) -> (0,8) not added, so no triangle)
        # Actually distance 4 edges: (u,u+4). Two such edges sharing vertex: (u,u+4) and (u+4,u+8). Triangle? (u,u+4),(u+4,u+8), need (u,u+8). Not added. So no triangle.
        # But (u,u+4) and (u+2,u+6) share no vertex. (u,u+4) and (u-4,u) share u. (u-4,u) and (u,u+4): triangle? (u-4,u),(u,u+4), need (u-4,u+4). Distance 8, not added. No triangle.
        # So all distance-4 same-parity edges are triangle-free!
        best = [(u, u+4) for u in range(n-4)]
        # Verify
        if not is_valid(best):
            # Add more edges
            for e in cands:
                if e not in best and is_valid(best + [e]):
                    best.append(e)
    
    return [[u, v] for u, v in best]

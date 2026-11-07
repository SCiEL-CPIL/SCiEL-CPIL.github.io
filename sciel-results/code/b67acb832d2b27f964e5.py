import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base = P['edges']
    base_set = {tuple(sorted(e)) for e in base}
    
    candidates = [(u, v) for u in range(n) for v in range(u+1, n) if (u,v) not in base_set]
    
    def check(added):
        all_e = base + added
        es = set()
        for u,v in all_e:
            k = (min(u,v), max(u,v))
            if k in es: return False
            es.add(k)
        # Triangle-free
        adj = [set() for _ in range(n)]
        for u,v in all_e:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common = adj[u] & adj[v]
                    if common:
                        return False
        # Diameter
        for src in range(n):
            dist = [-1]*n
            dist[src] = 0
            q = [src]
            for nd in q:
                for nb in adj[nd]:
                    if dist[nb] == -1:
                        dist[nb] = dist[nd]+1
                        q.append(nb)
            if max(dist) > 4:
                return False
        return True
    
    deadline = time.time() + 80
    
    # Try increasing k
    for k in range(1, len(candidates)+1):
        if time.time() > deadline:
            break
        # For small k, try combinations
        # This is expensive for large k, so use greedy for larger k
        if k <= 3:
            for combo in combinations(candidates, k):
                if time.time() > deadline:
                    break
                if check(list(combo)):
                    return [[int(u), int(v)] for u,v in combo]
        else:
            break
    
    # Greedy fallback
    current = list(base)
    cur_set = set(tuple(sorted(e)) for e in base)
    added = []
    
    def diam(edges):
        adj = [set() for _ in range(n)]
        for u,v in edges:
            adj[u].add(v)
            adj[v].add(u)
        md = 0
        for src in range(n):
            dist = [-1]*n
            dist[src]=0
            q=[src]
            for nd in q:
                for nb in adj[nd]:
                    if dist[nb]==-1:
                        dist[nb]=dist[nd]+1
                        q.append(nb)
            md = max(md, max(dist))
        return md
    
    def tri_free(edges):
        adj = [set() for _ in range(n)]
        for u,v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if v > u and (adj[u] & adj[v]):
                    return False
        return True
    
    cur_d = diam(current)
    while cur_d > 4 and time.time() < deadline:
        best_e = None
        best_d = cur_d + 1
        for (u,v) in candidates:
            if (u,v) in cur_set: continue
            ne = current + [(u,v)]
            if not tri_free(ne): continue
            d = diam(ne)
            if d < best_d:
                best_d = d
                best_e = (u,v)
        if best_e is None: break
        current.append(best_e)
        cur_set.add(best_e)
        added.append(best_e)
        cur_d = best_d
    
    if cur_d <= 4:
        return [[int(u), int(v)] for u,v in added]
    
    return []

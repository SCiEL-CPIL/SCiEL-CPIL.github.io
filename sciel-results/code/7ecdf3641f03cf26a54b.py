"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 85.0

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_dist(adj, n, start):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _diameter(adj, n):
    max_d = 0
    for s in range(n):
        d = _bfs_dist(adj, n, s)
        if -1 in d:
            return -1
        m = max(d)
        if m > max_d:
            max_d = m
    return max_d

def _check(n, base, added):
    es = set()
    for u,v in base:
        es.add((min(u,v), max(u,v)))
    for u,v in added:
        k = (min(u,v), max(u,v))
        if k in es:
            return False
        es.add(k)
    adj = [set() for _ in range(n)]
    for u,v in es:
        adj[u].add(v)
        adj[v].add(u)
    # triangle-free
    for i in range(n):
        for j in adj[i]:
            for k in adj[i]:
                if j < k and k in adj[j]:
                    return False
    # connected + diameter
    for s in range(n):
        d = _bfs_dist(adj, n, s)
        if -1 in d or max(d) > 4:
            return False
    return True

def _can_add(adj, n, u, v, edge_set):
    if u == v:
        return False
    k = (min(u,v), max(u,v))
    if k in edge_set:
        return False
    # triangle-free check
    for w in adj[u]:
        if w in adj[v]:
            return False
    return True

def construct():
    n = P['n']
    base = P['edges']
    base_set = set((min(u,v), max(u,v)) for u,v in base)
    
    # Initial adjacency
    adj = [set() for _ in range(n)]
    for u,v in base:
        adj[u].add(v)
        adj[v].add(u)
    
    added = []
    edge_set = set(base_set)
    
    # Phase 1: Try structured patterns
    patterns = [
        # Single strides
        [5], [6], [7], [8], [9], [10], [11], [12],
        # Pairs of strides
        [5, 7], [5, 8], [5, 9], [6, 8], [6, 9], [7, 9], [7, 10], [8, 10],
        [5, 10], [6, 10], [7, 11], [8, 11], [9, 12],
        # Triples
        [5, 7, 10], [6, 8, 10], [5, 8, 11], [6, 9, 12],
    ]
    
    best_result = None
    best_len = float('inf')
    
    for strides in patterns:
        if not _time_ok():
            break
        # Build candidate
        cand_added = []
        cand_es = set(base_set)
        for s in strides:
            for i in range(n - s):
                j = i + s
                k = (i, j)
                if k not in cand_es:
                    cand_added.append([i, j])
                    cand_es.add(k)
        if len(cand_added) < best_len and _check(n, base, cand_added):
            best_result = [list(e) for e in cand_added]
            best_len = len(cand_added)
    
    # Phase 2: Greedy add to get a valid solution
    if best_result is None:
        added = []
        edge_set = set(base_set)
        adj = [set() for _ in range(n)]
        for u,v in base:
            adj[u].add(v)
            adj[v].add(u)
        
        while _time_ok():
            d = _diameter(adj, n)
            if d <= 4:
                break
            
            # Find all pairs with dist > 4
            far = []
            for u in range(n):
                dist = _bfs_dist(adj, n, u)
                for v in range(u+1, n):
                    if dist[v] > 4:
                        far.append((u, v))
            
            if not far:
                break
            
            best_edge = None
            best_score = -1
            
            for u in range(n):
                for v in range(u+1, n):
                    if not _can_add(adj, n, u, v, edge_set):
                        continue
                    # Temporarily add
                    adj[u].add(v)
                    adj[v].add(u)
                    nd = _diameter(adj, n)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
                    if nd == -1:
                        continue
                    
                    # Score: prefer lower diameter, then fewer far pairs
                    score = -nd * 1000
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            edge_set.add((min(u,v), max(u,v)))
            adj[u].add(v)
            adj[v].add(u)
            added.append([u, v])
        
        if _check(n, base, added):
            best_result = added
            best_len = len(added)
    
    if best_result is None:
        return []
    
    # Phase 3: Greedy removal to minimize edges
    result = [list(e) for e in best_result]
    changed = True
    while changed and _time_ok():
        changed = False
        for i in range(len(result)-1, -1, -1):
            if not _time_ok():
                break
            # Try removing result[i]
            trial = result[:i] + result[i+1:]
            if _check(n, base, trial):
                result = trial
                changed = True
    
    # Phase 4: Local search - try swapping edges
    for _ in range(3):
        if not _time_ok():
            break
        improved = True
        while improved and _time_ok():
            improved = False
            # Try removing one edge and adding another
            for i in range(len(result)):
                if not _time_ok():
                    break
                for u in range(n):
                    for v in range(u+1, n):
                        if (min(u,v), max(u,v)) in base_set:
                            continue
                        # Check if (u,v) is already in result
                        if any(min(a,b)==min(u,v) and max(a,b)==max(u,v) for a,b in result):
                            continue
                        # Try: remove result[i], add (u,v)
                        trial = result[:i] + result[i+1:] + [[u,v]]
                        if _check(n, base, trial):
                            # Check if this allows further removal
                            result = trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            # After swap, try removal again
            changed = True
            while changed and _time_ok():
                changed = False
                for i in range(len(result)-1, -1, -1):
                    if not _time_ok():
                        break
                    trial = result[:i] + result[i+1:]
                    if _check(n, base, trial):
                        result = trial
                        changed = True
    
    # Ensure types are correct
    final = []
    for u, v in result:
        final.append([int(u), int(v)])
    return final
# EVOLVE-BLOCK-END

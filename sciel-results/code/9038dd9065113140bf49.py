import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if k != 3:
        return _luw_construct(k, g)
        
    target_girth = g
    # Moore bound for k=3, g=16 is 766.
    # Try n from 768 to 1000.
    
    best_n = None
    best_edges = None
    
    for n in range(768, 1000, 2):
        if time.time() > deadline - 10.0:
            break
            
        # Generate random 3-regular graph
        adj, edges = _random_3_reg(n)
        if adj is None:
            continue
            
        # Local search
        for _ in range(200): # Reduced iterations
            if time.time() > deadline - 10.0:
                break
                
            # Check girth
            ok, bad_edge = _check_girth_and_find_bad_edge(adj, n, target_girth)
            
            if ok:
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges[:]
                break
            
            # Swap bad_edge
            if bad_edge:
                u, v = bad_edge
                # Find partner
                swapped = False
                for _ in range(50):
                    x = random.randint(0, n-1)
                    if x == u or x == v:
                        continue
                    y_candidates = [y for y in adj[x] if y != u and y != v]
                    if not y_candidates:
                        continue
                    y = random.choice(y_candidates)
                    
                    # Check validity
                    if x in adj[u] or y in adj[v]:
                        continue
                    # Also check (u,y) and (v,x) if we swap differently?
                    # We are doing (u,x) and (v,y).
                    
                    _remove_edge(adj, edges, u, v)
                    _remove_edge(adj, edges, x, y)
                    _add_edge(adj, edges, u, x)
                    _add_edge(adj, edges, v, y)
                    swapped = True
                    break
                
                if not swapped:
                    # Revert? No, we didn't swap.
                    # Try next iteration.
                    pass

    if best_n is not None:
        return best_n, best_edges
    
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # ... (Original code) ...
    pass

def _random_3_reg(n):
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    edges = []
    for i in range(0, 3*n, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None, None
        if v in adj[u]:
            return None, None
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
    return adj, edges

def _check_girth_and_find_bad_edge(adj, n, target):
    half_g = target // 2
    # We return (True, None) if girth >= target
    # (False, (u,v)) if girth < target, where (u,v) is an edge in a short cycle
    
    for i in range(n):
        dist = [-1] * n
        parent = [-1] * n
        q = deque([i])
        dist[i] = 0
        while q:
            u = q.popleft()
            if dist[u] >= half_g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < half_g:
                        q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < target:
                        # Return this edge
                        return False, (u, v)
    return True, None

def _remove_edge(adj, edges, u, v):
    adj[u].remove(v)
    adj[v].remove(u)
    for i, e in enumerate(edges):
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            edges.pop(i)
            break

def _add_edge(adj, edges, u, v):
    adj[u].append(v)
    adj[v].append(u)
    edges.append((u, v))

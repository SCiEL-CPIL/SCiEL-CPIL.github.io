import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 3:
        return _lnw_construct(k, g)
        
    # Try to find a small graph with girth >= g
    # Known best is 936. Try 940.
    n = 940
    
    # Generate random 3-regular graph
    edges = None
    for _ in range(10):
        edges = _generate_random_3reg(n)
        if edges is not None:
            break
            
    if edges is None:
        return _lnw_construct(k, g)
        
    adj = _edges_to_adj(edges, n)
    
    best_n = n
    best_edges = edges[:]
    best_girth = 0
    
    while time.time() < deadline - 5:
        girth = _calculate_girth_bounded(adj, n, g)
        
        if girth >= g:
            return n, edges
            
        if girth > best_girth:
            best_girth = girth
            best_n = n
            best_edges = edges[:]
            
        # Find a shortest cycle
        cycle = _find_shortest_cycle_bounded(adj, n, g)
        if not cycle:
            break
            
        # Try to break the cycle
        broken = False
        for _ in range(100):
            u, v = random.choice(cycle)
            w, x = _pick_random_disjoint_edge(adj, edges, u, v)
            if w is None:
                continue
            if _try_switch(adj, edges, u, v, w, x):
                broken = True
                break
                
        if not broken:
            u, v = random.choice(edges)
            w, x = _pick_random_disjoint_edge(adj, edges, u, v)
            if w is not None:
                _try_switch(adj, edges, u, v, w, x)
                
    # If we didn't find a valid graph, return the best we found?
    # No, must be valid.
    # So return LUW fallback.
    return _lnw_construct(k, g)

def _generate_random_3reg(n):
    half_edges = []
    for i in range(n):
        for _ in range(3):
            half_edges.append(i)
    random.shuffle(half_edges)
    
    edges = set()
    i = 0
    while i < len(half_edges):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            return None # Loop
        if (min(u,v), max(u,v)) in edges:
            return None # Multi-edge
        edges.add((min(u,v), max(u,v)))
        i += 2
        
    # Check connectivity
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    count = 0
    while queue:
        u = queue.popleft()
        count += 1
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
                
    if count != n:
        return None # Disconnected
        
    return list(edges)

def _edges_to_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _calculate_girth_bounded(adj, n, g):
    min_cycle = g
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= min_cycle // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < g:
                                return min_cycle
    return min_cycle

def _find_shortest_cycle_bounded(adj, n, g):
    min_cycle_len = g
    best_cycle_edges = None
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= min_cycle_len // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle_len:
                            min_cycle_len = cycle_len
                            cycle_edges = []
                            p1 = u
                            while p1 != start:
                                p0 = parent[p1]
                                cycle_edges.append((min(p0,p1), max(p0,p1)))
                                p1 = p0
                            p2 = v
                            while p2 != start:
                                p0 = parent[p2]
                                cycle_edges.append((min(p0,p2), max(p0,p2)))
                                p2 = p0
                            cycle_edges.append((min(u,v), max(u,v)))
                            
                            best_cycle_edges = cycle_edges
                            if min_cycle_len < g:
                                return best_cycle_edges
    return best_cycle_edges

def _pick_random_disjoint_edge(adj, edges, u, v):
    for _ in range(100):
        w, x = random.choice(edges)
        if w in (u, v) or x in (u, v):
            continue
        if w in adj[u] or x in adj[v]:
            continue
        return w, x
    return None, None

def _try_switch(adj, edges, u, v, w, x):
    edges.remove((u,v))
    edges.remove((w,x))
    
    adj[u].remove(v)
    adj[v].remove(u)
    adj[w].remove(x)
    adj[x].remove(w)
    
    adj[u].add(w)
    adj[w].add(u)
    adj[v].add(x)
    adj[x].add(v)
    
    edges.append((min(u,w), max(u,w)))
    edges.append((min(v,x), max(v,x)))
    return True

def _lnw_construct(k, g):
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)

def _lnw_construct_params(k, d):
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

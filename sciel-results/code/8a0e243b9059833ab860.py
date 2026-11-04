import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if k == 4 and g == 10:
        # Try to find the (4,10)-cage (n=320)
        # Or at least a small graph
        for n_target in [320, 322, 324, 326, 328, 330]:
            if time.time() > deadline - 10:
                break
            result = find_small_graph(k, g, n_target, deadline)
            if result:
                return result
        
        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)

def find_small_graph(k, g, n, deadline):
    # Simulated Annealing
    # 1. Generate random k-regular graph
    edges = random_k_regular(n, k)
    if not edges:
        return None
        
    adj = [[] for _ in range(n)]
    adj_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        if u > v:
            u, v = v, u
        adj_set.add((u, v))
        
    current_girth = get_girth(adj, g)
    if current_girth >= g:
        return n, edges
        
    best_girth = current_girth
    best_edges = edges[:]
    
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute edge list for fast access
    edge_list = list(adj_set)
    
    while time.time() < deadline - 5:
        # Pick random edges
        e1 = random.choice(edge_list)
        e2 = random.choice(edge_list)
        if e1 == e2:
            continue
            
        u, v = e1
        x, y = e2
        
        # Try swap 1: (u,x), (v,y)
        if u != x and v != y:
            ux = (min(u,x), max(u,x))
            vy = (min(v,y), max(v,y))
            if ux not in adj_set and vy not in adj_set:
                # Apply
                adj_set.remove(e1)
                adj_set.remove(e2)
                adj[u].remove(v); adj[v].remove(u)
                adj[x].remove(y); adj[y].remove(x)
                
                adj_set.add(ux); adj_set.add(vy)
                adj[u].append(x); adj[x].append(u)
                adj[v].append(y); adj[y].append(v)
                
                edge_list.remove(e1)
                edge_list.remove(e2)
                edge_list.append(ux)
                edge_list.append(vy)
                
                new_girth = get_girth(adj, g)
                
                if new_girth >= g:
                    return n, list(adj_set)
                    
                delta = new_girth - current_girth
                if delta > 0:
                    current_girth = new_girth
                    if new_girth > best_girth:
                        best_girth = new_girth
                        best_edges = list(adj_set)
                else:
                    # Accept with probability exp(delta/T)
                    if random.random() < math.exp(delta / T):
                        current_girth = new_girth
                    else:
                        # Revert
                        adj_set.remove(ux); adj_set.remove(vy)
                        adj[u].remove(x); adj[x].remove(u)
                        adj[v].remove(y); adj[y].remove(v)
                        
                        adj_set.add(e1); adj_set.add(e2)
                        adj[u].append(v); adj[v].append(u)
                        adj[x].append(y); adj[y].append(x)
                        
                        edge_list.remove(ux)
                        edge_list.remove(vy)
                        edge_list.append(e1)
                        edge_list.append(e2)
                        
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart if stuck?
            # For now, just keep going
            
    return None

def get_girth(adj, g_limit):
    n = len(adj)
    # BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g_limit // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Cycle found
                    # Length is dist[u] + dist[v] + 1
                    # But we need to be careful: this is a cycle in the BFS tree
                    # It might not be the girth, but if it's < g_limit, we return it
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g_limit:
                        return cycle_len
                        
    return g_limit

def random_k_regular(n, k):
    # Simple stub matching
    for _ in range(1000):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                valid = False
                break
            edges.add((u, v))
            
        if valid:
            return list(edges)
    return []

def luw_construct(k, g):
    # Same as seed, but ensure it works
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
        
    n = len(order)
    return n, edges

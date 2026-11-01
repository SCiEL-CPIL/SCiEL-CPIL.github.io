import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Moore bound for k=4, g=10 is 161
    # We try to find a graph with n close to 161
    # If k != 4 or g != 10, fall back to LUW
    if k != 4 or g != 10:
        return _luw_construct(k, g)
        
    # Try local search for n from 161 to 200
    best_n = 320
    best_edges = None
    
    for n in range(161, 201):
        if time.time() > deadline - 10.0:
            break
            
        # Try multiple restarts for each n
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
                
            graph = _generate_random_4_reg(n)
            if graph is None:
                continue
                
            # Local search to increase girth
            success = _local_search_girth(graph, n, g, deadline)
            if success:
                # Verify girth
                if _check_girth(graph, n, g):
                    edges = _get_edges(graph)
                    return n, edges
                    
    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original LUW implementation
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def _generate_random_4_reg(n):
    # Generate a random 4-regular graph
    # Using configuration model
    stubs = [i for i in range(n) for _ in range(4)]
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    used = [False] * n  # Not enough for multi-edges, need edge set
    
    edges = set()
    # Pair stubs
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop
        if (u, v) in edges or (v, u) in edges:
            return None # Multi-edge
        edges.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return None
            
    return adj

def _check_girth(adj, n, g):
    # BFS from each vertex to find shortest cycle
    # Non-backtracking BFS
    for start in range(n):
        # dist[v][prev] = distance
        # We want to find a cycle
        # Use BFS tree, if we find a back edge, cycle length = dist[u] + dist[v] + 1
        # But for girth, we need shortest.
        
        # Simpler: BFS from start, tracking parent
        # If we reach a visited node, cycle found.
        # This finds the shortest cycle *containing start*.
        # Global girth is min over all starts.
        
        visited = [-1] * n
        parent = [-1] * n
        dist = [0] * n
        queue = deque([start])
        visited[start] = start
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if visited[v] == -1:
                    visited[v] = u
                    parent[v] = u
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Cycle found
                    # Length = dist[u] + dist[v] + 1
                    # This is a cycle, but not necessarily the shortest one in the graph
                    # However, BFS guarantees shortest path to v.
                    # The cycle length is dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _local_search_girth(adj, n, g, deadline):
    # Try to increase girth
    # Find shortest cycle
    # Perform swap
    
    for _ in range(100):
        if time.time() > deadline:
            return False
            
        # Find shortest cycle
        min_cycle_len = float('inf')
        cycle_edges = None
        
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            dist = [0] * n
            queue = deque([start])
            visited[start] = start
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if visited[v] == -1:
                        visited[v] = u
                        parent[v] = u
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle_len:
                            min_cycle_len = cycle_len
                            # Reconstruct cycle edges
                            # This is expensive, skip for now, just check length
                            
        if min_cycle_len >= g:
            return True
            
        # Perform random edge swap
        # Pick a random edge
        u = random.randint(0, n-1)
        v = random.choice(adj[u])
        
        # Pick another random edge
        x = random.randint(0, n-1)
        y = random.choice(adj[x])
        
        # Ensure (u,v) and (x,y) are distinct and not sharing vertices in a way that creates loops
        if u == x or u == y or v == x or v == y:
            continue
        if (u, v) == (x, y) or (u, v) == (y, x):
            continue
            
        # Swap
        # Remove (u,v) and (x,y)
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        # Add (u,x) and (v,y)
        # Check for multi-edges
        if x in adj[u] or y in adj[v]:
            # Revert
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
            continue
            
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(y)
        adj[y].append(v)
        
    return False

def _get_edges(adj):
    edges = []
    for u in range(len(adj)):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

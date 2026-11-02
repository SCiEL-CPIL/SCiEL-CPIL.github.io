import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0 # 5s margin
    
    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))
    
    def seed_construct(k, g):
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

    seed_n, seed_edges = seed_construct(k, g)
    
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def heuristic_girth_check(n, adj, g, sample_size=10):
        # Check only sample_size random sources
        sources = random.sample(range(n), min(sample_size, n))
        for s in sources:
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def generate_simple_regular(n, k):
        # Configuration model with repair
        max_attempts = 200
        for attempt in range(max_attempts):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            for i in range(0, n*k, 2):
                u, v = stubs[i], stubs[i+1]
                edges.append((u, v))
            
            # Repair: fix self-loops and multi-edges by swapping
            # Build adjacency count
            edge_count = {}
            for u, v in edges:
                key = (min(u,v), max(u,v))
                edge_count[key] = edge_count.get(key, 0) + 1
            
            # Check for self-loops and multi-edges
            has_issues = False
            for u, v in edges:
                if u == v:
                    has_issues = True
                    break
                key = (min(u,v), max(u,v))
                if edge_count[key] > 1:
                    has_issues = True
                    break
            
            if not has_issues:
                return edges
            
            # Try to repair by swapping endpoints
            # Find a bad edge and a good edge, swap one endpoint
            max_swaps = n * k
            for swap_iter in range(max_swaps):
                # Find a bad edge
                bad_idx = None
                for i, (u, v) in enumerate(edges):
                    if u == v:
                        bad_idx = i
                        break
                    key = (min(u,v), max(u,v))
                    if edge_count[key] > 1:
                        bad_idx = i
                        break
                
                if bad_idx is None:
                    break
                
                u1, v1 = edges[bad_idx]
                # Find a good edge to swap with
                found_swap = False
                for j in range(n*k):
                    if j == bad_idx:
                        continue
                    u2, v2 = edges[j]
                    if u2 == v2:
                        continue
                    key2 = (min(u2,v2), max(u2,v2))
                    if edge_count[key2] > 1:
                        continue
                    
                    # Try swapping v1 with v2
                    new_edges = list(edges)
                    new_edges[bad_idx] = (u1, v2)
                    new_edges[j] = (u2, v1)
                    
                    # Check if this creates new issues
                    valid = True
                    new_count = dict(edge_count)
                    # Remove old edges
                    for idx in [bad_idx, j]:
                        a, b = edges[idx]
                        if a != b:
                            key_old = (min(a,b), max(a,b))
                            new_count[key_old] -= 1
                            if new_count[key_old] == 0:
                                del new_count[key_old]
                    
                    # Add new edges
                    for idx in [bad_idx, j]:
                        a, b = new_edges[idx]
                        if a == b:
                            valid = False
                            break
                        key_new = (min(a,b), max(a,b))
                        if key_new in new_count and new_count[key_new] >= 1:
                            valid = False
                            break
                        new_count[key_new] = new_count.get(key_new, 0) + 1
                    
                    if valid:
                        edges = new_edges
                        edge_count = new_count
                        found_swap = True
                        break
                
                if not found_swap:
                    break
            
            # Final check
            edge_count = {}
            for u, v in edges:
                if u == v:
                    continue
                key = (min(u,v), max(u,v))
                edge_count[key] = edge_count.get(key, 0) + 1
            
            is_simple = True
            for u, v in edges:
                if u == v:
                    is_simple = False
                    break
                key = (min(u,v), max(u,v))
                if edge_count.get(key, 0) > 1:
                    is_simple = False
                    break
            
            if is_simple:
                return edges
        
        return None

    # Search
    best_n = seed_n
    best_edges = seed_edges
    
    # For k=5, g=9: Moore bound is 428, seed is 1116
    # Random 5-regular graphs on ~428-1116 vertices almost certainly have short cycles
    # The probability of girth >= 9 is extremely low for random graphs
    # So this search is unlikely to succeed, but let's try a few values
    
    # Focus on a smaller range near the Moore bound where we might have better luck
    # But realistically, random graphs won't work here
    
    # Better approach: try a few n values with more attempts
    # But limit total time
    
    search_start = max(moore_bound, 500)  # Start a bit above Moore bound
    search_end = min(seed_n, 700)  # Don't go too high
    
    if search_start < search_end:
        # Try a few n values
        n_values = list(range(search_start, min(search_start + 20, search_end), 3))
        
        for n in n_values:
            if time.time() > deadline - 15:
                break
            
            # Try up to 5 random graphs for this n
            for attempt in range(5):
                if time.time() > deadline - 15:
                    break
                edges = generate_simple_regular(n, k)
                if edges is None:
                    continue
                
                adj = build_adj(edges, n)
                
                # Quick heuristic check
                if heuristic_girth_check(n, adj, g, sample_size=3):
                    # Full check
                    if exact_girth_check(n, adj, g):
                        return n, edges
    
    return seed_n, seed_edges

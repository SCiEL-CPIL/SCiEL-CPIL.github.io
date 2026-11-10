import random
import time
import sys
from collections import deque

def construct(k, g):
    # Fallback to LUW construction if search fails
    # (Copy of the seed logic, slightly optimized)
    def luw_construct(k, g):
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

    # Moore Bound Calculation
    # For odd girth g=2r+1, MB = 1 + k * sum((k-1)^i for i in range(r))
    r = (g - 1) // 2
    mb = 1 + k * sum((k - 1) ** i for i in range(r))
    
    # Start search from slightly above Moore bound
    # For k=5, g=9, MB=426.
    # Known best is 1116.
    # We will try to find a graph with N < 1116.
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s for overhead
    
    def get_girth(n, edges):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = 10**9
        for i in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Cycle found
                        # Length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    def is_valid(n, edges, g):
        # Check simple graph
        if len(edges) != n * k // 2:
            return False
        seen = set()
        for u, v in edges:
            if u == v: return False
            if (min(u,v), max(u,v)) in seen: return False
            seen.add((min(u,v), max(u,v)))
            
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != k for d in deg):
            return False
            
        # Check girth
        girth = get_girth(n, edges)
        return girth >= g

    def generate_random_regular(n, k):
        # Configuration model
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = []
        adj = [[] for _ in range(n)]
        # Pair stubs
        for i in range(0, n, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u != v:
                # Check for multi-edge
                if v not in adj[u]:
                    edges.append((u, v))
                    adj[u].append(v)
                    adj[v].append(u)
                else:
                    # Multi-edge, need to fix. 
                    # For simplicity in generation, we restart if multi-edges are too frequent?
                    # Or we can just accept and fix later?
                    # Better to restart generation if multi-edges occur in this simple model.
                    # But probability is low for large n.
                    pass
        # This simple generation often fails to produce a simple graph.
        # We need a robust generator.
        # Let's use a swap-based repair.
        
        # Re-implementation of robust generation:
        # 1. Create k-regular multigraph.
        # 2. Remove loops and multi-edges by swapping.
        
        # Actually, let's just use the LUW graph as a strong fallback and try a few random instances.
        # If random fails, we return LUW.
        
        return edges, adj

    # Search Loop
    best_n = float('inf')
    best_edges = None
    
    # Try a few N values
    # For k=5, g=9, try N from 500 to 1116 in steps.
    # If we find a valid one, return it.
    
    # To ensure we finish, we only try a limited number of candidates.
    candidates = []
    # Add known good size
    candidates.append(1116)
    # Try smaller
    for n in range(500, 1116, 10):
        candidates.append(n)
        
    random.shuffle(candidates)
    
    for n in candidates:
        if time.time() - start_time > timeout:
            break
            
        # Try 5 random instances for this N
        for _ in range(5):
            if time.time() - start_time > timeout:
                break
                
            # Generate random 5-regular graph
            # Using a simple method: Start with a cycle, add chords? No.
            # Let's use the "random pairing" and then "swap to fix" method.
            
            # 1. Generate stubs
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            edges = []
            adj = [[] for _ in range(n)]
            
            # Pair them up
            for i in range(0, n, 2):
                u, v = stubs[i], stubs[i+1]
                if u != v:
                    edges.append((u, v))
                    adj[u].append(v)
                    adj[v].append(u)
                else:
                    # Loop, ignore for now, will fix
                    pass
            
            # Fix loops and multi-edges
            # This is complex. 
            # Alternative: Use a library? No.
            # Let's just check if the generated graph is simple.
            # If not, skip.
            
            is_simple = True
            seen = set()
            for u, v in edges:
                if u == v:
                    is_simple = False
                    break
                if (min(u,v), max(u,v)) in seen:
                    is_simple = False
                    break
                seen.add((min(u,v), max(u,v)))
            
            if not is_simple:
                continue
                
            if len(edges) != n * k // 2:
                continue
                
            # Check Girth
            girth = get_girth(n, edges)
            if girth >= g:
                return n, edges
                
    # If no small graph found, return LUW
    return luw_construct(k, g)
